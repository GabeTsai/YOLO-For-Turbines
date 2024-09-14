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
        print(f"MAP: {mAP.item()}")

    return sum(losses) / len(losses)

def train(model, optimizer, loss_fn, train_loader, val_loader, grad_scaler, num_epochs, model_folder_path):
    scaled_anchors = torch.tensor(config.ANCHORS) * (
                    torch.tensor(config.GRID_SIZES).unsqueeze(1).unsqueeze(1).repeat(1,3,2))
    min_val_loss = float("inf")
    epoch = 0
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
        session.report({"min_val_loss": min_val_loss})
        epoch += 1
    wandb.finish()
    save_checkpoint(best_model, optimizer, filename = Path(model_folder_path) / "best_model.pth")
    return best_model

def tune_model(csv_folder_path, model_folder_path):
    config = choose_hyperparameter_config()
    model = YOLOv3(num_classes = config.NUM_TURBINE_CLASSES).to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.LR)
    loss_fn = YOLOLoss()
    grad_scaler = torch.GradScaler(device_type = config.DEVICE)
    train_loader, val_loader = get_loaders(csv_folder_path)
    best_model = train(model, optimizer, loss_fn, train_loader, val_loader, grad_scaler, num_epochs = config["num_epochs"], model_folder_path = model_folder_path)
    return best_model

def main():
    wandb.login()
    model_folder_path = Path("../models")
    csv_folder_path = Path("../data")
    best_model = tune_model(csv_folder_path, model_folder_path)

if __name__ == "__main__":
    main()