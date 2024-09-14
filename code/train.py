import config
import torch
from torch.amp import GradScaler
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import wandb
wandb.require("core")
import pathlib as Path
import json

from model import YOLOv3
from loss import YOLOLoss
from utils import (
    calc_mAP, 
    cells_to_boxes,
    get_eval_boxes,
    check_model_accuracy,
    save_checkpoint, 
    load_checkpoint,
    get_loaders
)
import config

import tqdm

def choose_hyperparameter_config():
    return {
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]), 
        "num_epochs": 100
    }

def train_one_epoch(train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors):
    loop = tqdm(train_loader, leave = True)
    losses = [] # we have losses for each scale
    model.train()
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (  # y0, y1, y2 are targets for each scale
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast(device_type = "cuda"): #Darknet uses mixed precision training
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        
        losses.append(loss.item())
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()    #adjust scaling factor if underflow/overflow occurs
        loop.set_postfix(loss = loss.item())
    return sum(losses) / len(losses)

def val_one_epoch(val_loader, model, loss_fn, scaled_anchors, epoch):
    loop = tqdm(val_loader, leave = True)
    obj_losses, no_obj_losses, box_losses, class_losses = [], [], [], []
    losses = []
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in loop:
            x = x.to(config.DEVICE)
            y0, y1, y2 = (
                y[0].to(config.DEVICE),
                y[1].to(config.DEVICE),
                y[2].to(config.DEVICE)
            )

            targets.append([y0, y1, y2])

            out = model(x)

            losses = torch.tensor([loss_fn(out[0], y0, scaled_anchors[0]),
                                   loss_fn(out[1], y1, scaled_anchors[1]),
                                   loss_fn(out[2], y2, scaled_anchors[2])])
            
            box_loss = losses[:, 0].sum().item()
            obj_loss = losses[:, 1].sum().item()
            no_obj_loss = losses[:, 2].sum().item()
            class_loss = losses[:, 3].sum().item()

            loss = box_loss + obj_loss + no_obj_loss + class_loss

            box_losses.append(box_loss)
            obj_losses.append(obj_loss)
            no_obj_losses.append(no_obj_loss)
            class_losses.append(class_loss)
            losses.append(loss.item())

            predictions.append(out)
            loop.set_postfix(loss = loss.item())

    wandb.log({"box_loss": sum(box_losses) / len(box_losses)})
    wandb.log({"obj_loss": sum(obj_losses) / len(obj_losses)})
    wandb.log({"no_obj_loss": sum(no_obj_losses) / len(no_obj_losses)})
    wandb.log({"class_loss": sum(class_losses) / len(class_losses)})

    if (epoch + 1) % 5 == 0:
        check_model_accuracy(predictions, targets, object_threshold = config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_eval_boxes(predictions, targets, 
                                                iou_threshold = config.NMS_IOU_THRESHOLD,
                                                anchors = config.ANCHORS, 
                                                obj_threshold = config.CONF_THRESHOLD)
        mAP = calc_mAP(pred_boxes, true_boxes, iou_threshold = config.MAP_IOU_THRESHOLD, 
                    num_classes = config.NUM_TURBINE_CLASSES)
        wandb.log({"mAP": mAP.item()})
        print(f"MAP: {mAP.item()}")

    return sum(losses) / len(losses)

def train(hyperparam_config, csv_folder_path, model_folder_path):

    wandb.init(
      project=f"YOLOv3_Turbine_Detection",
      config = config
      )

    wandb.log(config)

    model = YOLOv3(num_classes = config.NUM_TURBINE_CLASSES).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.LR)
    loss_fn = YOLOLoss()
    grad_scaler = torch.GradScaler(device_type = config.DEVICE)

    train_loader, val_loader, _ = get_loaders(csv_folder_path, batch_size = hyperparam_config["batch_size"])

    scaled_anchors = torch.tensor(config.ANCHORS) * (
                    torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2))
    
    min_val_loss = float("inf")
    epoch = 0
    early_stop = int(0.05 * hyperparam_config["num_epochs"])
    best_model = None

    while epoch < hyperparam_config["num_epochs"] and early_stop != 0:
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors)
        
        print(f"Train loss at epoch {epoch}: {train_loss}")
        wandb.log({"train_loss": train_loss})

        val_loss = val_one_epoch(val_loader, model, loss_fn, scaled_anchors, epoch)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model
            early_stop = int(0.05 * hyperparam_config["num_epochs"])
        else:
            early_stop -= 1

        print(f"Val loss at epoch {epoch}: {val_loss}")
        wandb.log({"val_loss": val_loss})
        wandb.log({"lowest_val_loss": min_val_loss})
        session.report({"min_val_loss": min_val_loss})

        epoch += 1
    save_checkpoint(best_model, optimizer, filename = Path(model_folder_path) / "best_model.pth")
    wandb.log_model(Path(model_folder_path) / "best_model.pth", name = "best_model")
    wandb.finish()

def tune_model(csv_folder_path, model_folder_path):
    wandb.login()
    hyperparam_config = choose_hyperparameter_config()

    scheduler = ASHAScheduler(
        metric = "min_val_loss", 
        mode = "min", 
        max_t = config["num_epochs"],
        grace_period = 2,
        brakets = 2, 
        reduction_factor = 4
    )

    tuner = tune.Tuner(
        tune.with_resources(train, hyperparam_config = hyperparam_config, 
                            model_folder_path = model_folder_path),
        param_space = config,
        tune_config = tune.TuneConfig(
            scheduler = scheduler, 
            num_samples = 50,
            max_concurrent_trials= config.NUM_PROCESSES
        ),
        run_config = ray.air.config.RunConfig(
            name = "tune_model", 
            verbose = 1
        )
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="mean_val_loss", mode="min")
    best_config = best_result.config
    best_metrics = best_result.metrics
    print("Best hyperparameters found were: ", best_config)
    print("Best validation loss found was: ", best_metrics['mean_val_loss'])

    best_settings_map = {
        "config": best_config,
        "loss": best_metrics['mean_val_loss']
    }
    with open (f'{model_folder_path}/best_config.json', 'w') as f:
        json.dump(best_settings_map, f)

def main():
    model_folder_path = Path("../models")
    csv_folder_path = Path("../data")
    tune_model(csv_folder_path, model_folder_path)

if __name__ == "__main__":
    main()