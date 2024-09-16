import config
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import wandb
wandb.require("core")
from pathlib import Path
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
    get_loaders,
    seed_everything
)
import config
import os

from tqdm import tqdm

def choose_hyperparameter_config():
    return {
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": 0.0005,
        "batch_size": 64, 
        "momentum": 0.9,
        "max_num_steps": 10000
    }

def train_one_epoch(train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)

    model.train()
    tot_box_loss, tot_obj_loss, tot_no_obj_loss, tot_class_loss = 0, 0, 0, 0
    tot_loss = 0

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)

        y0, y1, y2 = (  # y0, y1, y2 are targets for each scale
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.amp.autocast(device_type=config.DEVICE):  # Mixed precision training
            out = model(x)

            box_loss_0, obj_loss_0, no_obj_loss_0, class_loss_0 = loss_fn(out[0], y0, scaled_anchors[0])
            box_loss_1, obj_loss_1, no_obj_loss_1, class_loss_1 = loss_fn(out[1], y1, scaled_anchors[1])
            box_loss_2, obj_loss_2, no_obj_loss_2, class_loss_2 = loss_fn(out[2], y2, scaled_anchors[2])

            box_loss = box_loss_0 + box_loss_1 + box_loss_2
            obj_loss = obj_loss_0 + obj_loss_1 + obj_loss_2
            no_obj_loss = no_obj_loss_0 + no_obj_loss_1 + no_obj_loss_2
            class_loss = class_loss_0 + class_loss_1 + class_loss_2

            loss = box_loss + obj_loss + no_obj_loss + class_loss

        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        tot_box_loss += box_loss.item()
        tot_obj_loss += obj_loss.item()
        tot_no_obj_loss += no_obj_loss.item()
        tot_class_loss += class_loss.item()
        tot_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    wandb.log({"train_box_loss": tot_box_loss / len(train_loader)})
    wandb.log({"train_obj_loss": tot_obj_loss / len(train_loader)})
    wandb.log({"train_no_obj_loss": tot_no_obj_loss / len(train_loader)})
    wandb.log({"train_class_loss": tot_class_loss / len(train_loader)})

    return tot_loss / len(train_loader)

def val_one_epoch(val_loader, model, loss_fn, scaled_anchors, epoch):
    loop = tqdm(val_loader, leave=True)
    tot_box_loss, tot_obj_loss, tot_no_obj_loss, tot_class_loss = 0, 0, 0, 0
    tot_loss = 0
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
            
            targets.append([y0.clone(), y1.clone(), y2.clone()])

            out = model(x)

            box_loss_0, obj_loss_0, no_obj_loss_0, class_loss_0 = loss_fn(out[0], y0, scaled_anchors[0])
            box_loss_1, obj_loss_1, no_obj_loss_1, class_loss_1 = loss_fn(out[1], y1, scaled_anchors[1])
            box_loss_2, obj_loss_2, no_obj_loss_2, class_loss_2 = loss_fn(out[2], y2, scaled_anchors[2])

            box_loss = box_loss_0 + box_loss_1 + box_loss_2
            obj_loss = obj_loss_0 + obj_loss_1 + obj_loss_2
            no_obj_loss = no_obj_loss_0 + no_obj_loss_1 + no_obj_loss_2
            class_loss = class_loss_0 + class_loss_1 + class_loss_2

            loss = box_loss + obj_loss + no_obj_loss + class_loss

            tot_box_loss += box_loss.item()
            tot_obj_loss += obj_loss.item()
            tot_no_obj_loss += no_obj_loss.item()
            tot_class_loss += class_loss.item()
            tot_loss += loss.item()

            predictions.append(out)
            loop.set_postfix(loss=loss.item())

    wandb.log({"val_box_loss": tot_box_loss / len(val_loader)})
    wandb.log({"val_obj_loss": tot_obj_loss / len(val_loader)})
    wandb.log({"val_no_obj_loss": tot_no_obj_loss / len(val_loader)})
    wandb.log({"val_class_loss": tot_class_loss / len(val_loader)})

    val_loss = tot_loss / len(val_loader)
    session.report({"val_loss": val_loss})

    if (epoch + 1) % 5 == 0:
        class_accuracy, noobj_accuracy, obj_accuracy = check_model_accuracy(predictions, targets, object_threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_eval_boxes(predictions, targets, 
                                                iou_threshold=config.NMS_IOU_THRESHOLD,
                                                anchors=config.ANCHORS, 
                                                obj_threshold=config.CONF_THRESHOLD)
        mAP = calc_mAP(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESHOLD, 
                       num_classes=config.NUM_TURBINE_CLASSES)
        wandb.log({"class accuracy: ": class_accuracy})
        wandb.log({"noobj accuracy: ": noobj_accuracy})
        wandb.log({"obj accuracy: ": obj_accuracy})
        wandb.log({"mAP": mAP.item()})
        print(f"MAP: {mAP.item()}")

    return val_loss

def train(hyperparam_config, csv_folder_path, model_folder_path, identifier):

    wandb.init(
    #   project=f"YOLOv3_Turbine_Detection_Test",
    #   config = hyperparam_config
    #   )

    wandb.log(hyperparam_config)

    model = YOLOv3(num_classes = config.NUM_TURBINE_CLASSES, 
                    weights_path = Path(config.WEIGHTS_FOLDER) / "darknet53.conv.74").to(config.DEVICE)
    model.load_weights()
    optimizer = torch.optim.SGD(model.parameters(), lr = hyperparam_config["lr"], 
                                momentum = hyperparam_config["momentum"], weight_decay = hyperparam_config["weight_decay"])
    # optimizer = torch.optim.AdamW(model.parameters(), lr = hyperparam_config["lr"], weight_decay = hyperparam_config["weight_decay"])
    loss_fn = YOLOLoss()
    grad_scaler = torch.amp.GradScaler()
    # warmup_scheduler = LinearLR(optimizer, start_factor = 1e-6, end_factor = 1, steps = 1000)

    train_loader, val_loader, _ = get_loaders(csv_folder_path, batch_size = hyperparam_config["batch_size"])

    scaled_anchors = (torch.tensor(config.ANCHORS) 
                      * torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(config.DEVICE)

    min_val_loss = float("inf")
    epoch = 0
    
    num_epochs = hyperparam_config["max_num_steps"] // len(train_loader)
    early_stop = int(0.05 * num_epochs)
    best_model = None

    while epoch < num_epochs and early_stop != 0:
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors)
        
        print(f"Train loss at epoch {epoch}: {train_loss}")
        wandb.log({"train_loss": train_loss})

        val_loss = val_one_epoch(val_loader, model, loss_fn, scaled_anchors, epoch)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = model
            early_stop = int(0.05 * num_epochs)
        else:
            early_stop -= 1

        print(f"Val loss at epoch {epoch}: {val_loss}")
        wandb.log({"val_loss": val_loss})
        wandb.log({"lowest_val_loss": min_val_loss})

        epoch += 1
    save_checkpoint(best_model, optimizer, filename = Path(model_folder_path) / f"best_model_{identifier}.pth")
    wandb.log_model(Path(model_folder_path) / "best_model.pth", name = f"best_model_{identifier}")
    wandb.finish()

def tune_model(csv_folder_path, model_folder_path, identifier):
    wandb.login()
    hyperparam_config = choose_hyperparameter_config()

    scheduler = ASHAScheduler(
        metric = "val_loss", 
        mode = "min", 
        grace_period = 2,
        brackets = 2, 
        reduction_factor = 2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, csv_folder_path = csv_folder_path,
                            model_folder_path = model_folder_path, identifier = identifier),
            resources = {"cpu": config.NUM_WORKERS, "gpu": config.NUM_GPUS/config.NUM_PROCESSES}
        ),
        param_space = hyperparam_config,
        tune_config = tune.TuneConfig(
            scheduler = scheduler, 
            num_samples = 20,
            max_concurrent_trials= config.NUM_PROCESSES
        ),
        run_config = ray.air.config.RunConfig(
            name = "tune_YOLOv3", 
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
    seed_everything()
    # tune_model(config.CSV_FOLDER, config.MODEL_FOLDER, 'LR')
    hyperparam_config = choose_hyperparameter_config()
    train(hyperparam_config, config.CSV_FOLDER, config.MODEL_FOLDER, 'test')

if __name__ == "__main__":
    main()