import config
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
import wandb
wandb.require("core")
import time  

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

def train_one_epoch(train_dataset, train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors, warmup_scheduler):
    loop = tqdm(train_loader, leave=True)

    model.train()
    tot_box_loss, tot_obj_loss, tot_no_obj_loss, tot_class_loss = 0, 0, 0, 0
    tot_loss = 0

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        if (batch_idx + 1) % 10 == 0:  # Use batch_idx directly
            train_dataset.change_scale()
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

        if warmup_scheduler is not None:
            lr = optimizer.param_groups[0]['lr']
            wandb.log({"lr": lr})
            warmup_scheduler.step()  

        tot_box_loss += box_loss.item()
        tot_obj_loss += obj_loss.item()
        tot_no_obj_loss += no_obj_loss.item()
        tot_class_loss += class_loss.item()
        tot_loss += loss.item()

        loop.set_postfix(loss=loss.item())

    if torch.sum(torch.isnan(loss)) > 0:
        raise ValueError("Nan loss")

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

    with torch.no_grad():  
        for x, y in loop:
            x = x.to(config.DEVICE)
            y0, y1, y2 = (
                y[0].to(config.DEVICE),
                y[1].to(config.DEVICE),
                y[2].to(config.DEVICE)
            )
            
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

            loop.set_postfix(loss=loss.item())

    if torch.sum(torch.isnan(loss)) > 0:
        raise ValueError("Nan loss")

    wandb.log({"val_box_loss": tot_box_loss / len(val_loader)})
    wandb.log({"val_obj_loss": tot_obj_loss / len(val_loader)})
    wandb.log({"val_no_obj_loss": tot_no_obj_loss / len(val_loader)})
    wandb.log({"val_class_loss": tot_class_loss / len(val_loader)})

    val_loss = tot_loss / len(val_loader)

    mAP = None
    if (epoch + 1) % 10 == 0:
        class_accuracy, noobj_accuracy, obj_accuracy = check_model_accuracy(model, val_loader, object_threshold=config.CONF_THRESHOLD)
        pred_boxes, true_boxes = get_eval_boxes(val_loader, model, 
                                                iou_threshold=config.NMS_IOU_THRESHOLD,
                                                anchors=config.ANCHORS, 
                                                obj_threshold=config.CONF_THRESHOLD)
        mAP = calc_mAP(pred_boxes, true_boxes, iou_threshold=config.MAP_IOU_THRESHOLD, 
                       num_classes=config.NUM_TURBINE_CLASSES).item()
        wandb.log({"class accuracy: ": class_accuracy})
        wandb.log({"noobj accuracy: ": noobj_accuracy})
        wandb.log({"obj accuracy: ": obj_accuracy})
        wandb.log({"mAP": mAP})
        session.report({"mAP": mAP})
        print(f"MAP: {mAP}")

    return val_loss, mAP

def train(hyperparam_config, csv_folder_path, model_folder_path, identifier, checkpoint_name = None):
    wandb.init(
      project=f"YOLOv3_Turbine_Detection_{identifier}",
      config = hyperparam_config
      )

    wandb.log(hyperparam_config)
    
    model = YOLOv3(num_classes = config.NUM_TURBINE_CLASSES, 
                    activation = hyperparam_config["activation"],
                    weights_path = Path(config.WEIGHTS_FOLDER) / "darknet53.conv.74", 
                    freeze = config.FREEZE_BACKBONE).to(config.DEVICE,)

    optimizer = torch.optim.SGD(model.parameters(), lr = hyperparam_config["lr"], 
                                momentum = hyperparam_config["momentum"], weight_decay = hyperparam_config["weight_decay"])

    if config.LOAD_CHECKPOINT:
        load_checkpoint(model, optimizer, lr = hyperparam_config["lr"], filename = checkpoint_name)
    elif config.LOAD_WEIGHTS:
        model.load_weights()

    loss_fn = YOLOLoss()
    grad_scaler = torch.amp.GradScaler()
    start_factor = 1e-6

    warmup_scheduler = None
    decay_scheduler = None

    if config.WARMUP:
        warmup_scheduler = LinearLR(optimizer, start_factor = start_factor, end_factor = 1, 
                                    total_iters = hyperparam_config["max_num_steps"] * hyperparam_config["warmup"])
        warmup_scheduler.step()
    # if config.DECAY_LR:
    #     decay_scheduler = CosineAnnealingLR(optimizer, T_max = hyperparam_config[""])

    train_loader, val_loader, train_dataset = get_loaders(csv_folder_path, batch_size = hyperparam_config["batch_size"], train = True)

    scaled_anchors = (torch.tensor(config.ANCHORS) 
                      * torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2)
    ).to(config.DEVICE)

    best_mAP = 0
    epoch = 0
    
    num_epochs = hyperparam_config["max_num_steps"] // len(train_loader)
    print(f"Num epochs: {num_epochs}")
    best_model = None

    start_time = time.time()
    while epoch < num_epochs:
        train_loss = train_one_epoch(train_dataset, train_loader, model, optimizer, loss_fn, grad_scaler, scaled_anchors, warmup_scheduler)
        
        print(f"Train loss at epoch {epoch}: {train_loss}")
        wandb.log({"train_loss": train_loss})

        val_loss, mAP = val_one_epoch(val_loader, model, loss_fn, scaled_anchors, epoch)
        if mAP is not None and mAP > best_mAP:
            best_mAP = mAP
            best_model = model

        print(f"Val loss at epoch {epoch}: {val_loss}")
        wandb.log({"val_loss": val_loss})

        epoch += 1
        
        if (epoch + 1) % int(0.25 * num_epochs) == 0:
            save_checkpoint(best_model, optimizer, filename = Path(model_folder_path) / f"best_model_{identifier}.pth")
            wandb.log_model(str(Path(model_folder_path) / f"best_model_{identifier}.pth"), name=f"best_model_{identifier}")
        
        elapsed_time = time.time() - start_time
        wandb.log({"time_elapsed_in_hours": elapsed_time/3600})

    save_checkpoint(best_model, optimizer, filename = Path(model_folder_path) / f"best_model_{identifier}.pth")
    wandb.log_model(str(Path(model_folder_path) / f"best_model_{identifier}.pth"), name=f"best_model_{identifier}")
    
    wandb.finish()

def tune_model(csv_folder_path, model_folder_path, hyperparam_config, num_samples, identifier, checkpoint_name):
    wandb.login()
    
    scheduler = ASHAScheduler(
        metric = "mAP", 
        mode = "max", 
        grace_period = 5,
        brackets = 2, 
        reduction_factor = 2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train, csv_folder_path = csv_folder_path,
                            model_folder_path = model_folder_path, identifier = identifier,
                            checkpoint_name = checkpoint_name),
            resources = {"cpu": config.NUM_WORKERS, "gpu": config.NUM_GPUS/config.NUM_PROCESSES}
        ),
        param_space = hyperparam_config,
        tune_config = tune.TuneConfig(
            scheduler = scheduler, 
            num_samples = num_samples,
            max_concurrent_trials= config.NUM_PROCESSES
        ),
        run_config = ray.air.config.RunConfig(
            name = "tune_YOLOv3", 
            verbose = 1
        )
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="mAP", mode="max")
    best_config = best_result.config
    best_metrics = best_result.metrics
    print("Best hyperparameters found were: ", best_config)
    print("Last mAP found was: ", best_metrics['mAP'])

    best_settings_map = {
        "config": best_config,
        "mAP": best_metrics['mAP']
    }
    with open (f'{model_folder_path}/best_config.json', 'w') as f:
        json.dump(best_settings_map, f)

def load_config(model_folder, config_name):
    with open (f'{model_folder}/{config_name}', 'r') as f:
        hyperparam_config = json.load(f)
    return hyperparam_config["config"]

def main():
    seed_everything()
    identifier = 'MOSAIC'
    num_samples = 20
    model_folder = config.MODEL_FOLDER
    csv_folder = config.CSV_FOLDER

    hyperparam_config = load_config(model_folder, f"best_config_LR.json")
    hyperparam_config["activation"] = "mish"
    # hyperparam_config["warmup"] = tune.grid_search([0.01 * i for i in range(1, 11)])
    hyperparam_config["warmup"] = 0.05
    
    # tune_model(csv_folder, model_folder, hyperparam_config, num_samples, identifier, checkpoint_name = "best_model_91824.pth")

    train(hyperparam_config, csv_folder, model_folder, identifier = identifier, checkpoint_name = "best_model_mish_latest.pth")


if __name__ == "__main__":
    main()