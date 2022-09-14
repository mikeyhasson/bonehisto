import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from . import utils
import wandb
from .coco_utils import  get_batch_statistics, ap_per_class

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.wandb("train")
    wandb.log({"lr": optimizer.param_groups[0]["lr"],"epoch":epoch})
    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def valid_one_epoch(model, data_loader, device, scaler):
    metric_logger = utils.MetricLogger(delimiter="  ")

    for images, targets in metric_logger.log_every(data_loader, 100, "Test:"):
        with torch.no_grad():
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                loss_dict = model(images, targets)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    metric_logger.wandb("valid")

    return metric_logger


#https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch4-RetinaNet.html
def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds

@torch.inference_mode()
def evaluate(model, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")

    labels = []
    preds_adj_all = []
    annot_all = []

    for images, annot in metric_logger.log_every(data_loader, 100, "Test:"):
        images = list(image.to(device) for image in images)
        # annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

        for t in annot:
            labels += t['labels']

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()

        with torch.no_grad():
            preds_adj = make_prediction(model, images, 0.5)
            preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
            preds_adj_all.append(preds_adj)
            annot_all.append(annot)

        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    sample_metrics = []
    for batch_i in range(len(preds_adj_all)):
        sample_metrics += get_batch_statistics(preds_adj_all[batch_i], annot_all[batch_i], iou_threshold=0.5)

    true_positives, pred_scores, pred_labels = [torch.cat(x, 0) for x in list(zip(*sample_metrics))]  # 배치가 전부 합쳐짐
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels,
                                                             torch.tensor(labels))
    mAP = torch.mean(AP)

    metric_logger.synchronize_between_processes()
    print("Stats:", metric_logger)

    metric_logger = utils.MetricLogger(delimiter="  ")
    d = {'recall': recall, 'precision': precision, 'f1': f1, 'mAP':mAP}
    print(' '.join([f'{k} = {v}' for k,v in d]))
    wandb.log(d)