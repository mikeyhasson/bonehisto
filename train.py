# https://github.com/pytorch/vision/blob/main/references/detection/train.py
r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.

the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import wandb
from sahi.model import TorchVisionDetectionModel

import tools.utils as utils
from retinanet import retinanet_resnet18_fpn_v2
from tools import presets
from tools.coco_utils import get_coco
from tools.engine import train_one_epoch, valid_one_epoch, inference
from tools.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler


def get_dataset(image_set, transform, data_path):
    ds = get_coco(data_path, image_set=image_set, transforms=transform)
    return ds


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrain(data_augmentation=args.data_augmentation
                                            , crop_size=args.crop_size)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()
        return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEval(args.crop_size)


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="/home/paperspace/bonehisto/rfimg", type=str, help="dataset path")
    parser.add_argument("--dataset", default="bonecell", type=str, help="dataset name")
    parser.add_argument("--model", default="retinanet_resnet18_fpn", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--crop-size", default="512", type=int, help="model name")
    parser.add_argument("--opt", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )


    parser.add_argument(
        "--lr-scheduler", default="reducelronplateau", type=str, help="name of lr scheduler (default: multisteplr)"
    )

    parser.add_argument("--scheduler-factor", default=0.7, type=float, help="for ReduceLROnPlateau")
    parser.add_argument("--scheduler-patience", default=6, type=int, help="for ReduceLROnPlateau")

    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[50, 80],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--max-lr", default=0.1, type=float, help="maximal lr (onecyclelr scheduler only)"
    )
    parser.add_argument("--name", default="run", type=str, help="name run for wandb")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="out_weights", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--trainable-backbone-layers", default=5, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="bonecell", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default='tenpercent_resnet18.pth', type=str,
                        help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")
    return parser


def init_wandb(args):
    config_dic = {"data_augmentation": args.data_augmentation, "epochs": args.epochs, "lr0": args.lr,
                  "lr_gamma": args.lr_gamma, "lr_scheduler": args.lr_scheduler, "opt": args.opt,
                  "weight_decay": args.weight_decay, "batch_size": args.batch_size}

    wandb.init(project=args.dataset, config=config_dic, name=args.name)
    wandb.define_metric("epoch")
    for name in ["train/*", "valid/*", "recall", 'ap_0.5:0.95', 'ap_0.5', 'lr']:
        wandb.define_metric(name, step_metric="epoch")

    wandb.log({"lr": args.lr, "epoch": 0})


def get_eval_model(model, image_size, device):
    return TorchVisionDetectionModel(
        model=model,
        image_size=image_size,
        device=device,
        load_at_init=True,
    )


def main(args):
    init_wandb(args)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.use_deterministic_algorithms(True)

    # Data loading code
    print("Loading data")

    num_classes = 4 + 1
    dataset = get_dataset("train", get_transform(True, args), args.data_path)
    dataset_test = get_dataset("val", get_transform(False, args), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    train_collate_fn = utils.collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )

    print("Creating model")
    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]  # https://github.com/ozanciga/self-supervised-histopathology/issues/2
    model = retinanet_resnet18_fpn_v2(weights=args.weights, weights_backbone=args.weights_backbone,
                                      num_classes=num_classes, image_mean=mean, image_std=std,
                                      min_size=args.crop_size, max_size=args.crop_size, **kwargs)
    model.to(device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    parameters = [p for p in model.parameters() if p.requires_grad]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, betas=(0.9, 0.999), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()

    if args.lr_scheduler == "multisteplr":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == "onecyclelr":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,steps_per_epoch = args.batch_size,
                                                           max_lr = args.max_lr, epochs=args.epochs)
    elif args.lr_scheduler == "reducelronplateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=args.scheduler_factor,
                                                                  patience=args.scheduler_patience)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR, OneCycleLR and ReduceLROnPlateau are supported."
        )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cuda")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    eval_model = get_eval_model(model=model, image_size=args.crop_size, device=args.device)

    if args.test_only:
        torch.backends.cudnn.deterministic = True
        inference(eval_model, data_loader_test)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
        loss = valid_one_epoch(model, data_loader, device, scaler)
        lr_scheduler.step(loss)
        wandb.log({"lr": optimizer.param_groups[0]["lr"], "epoch": epoch})

        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
            if args.amp:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

'''
Ratios: [0.316, 0.602, 1.0, 1.661, 3.164]
Scales: [0.4, 0.46, 0.527, 0.605, 0.695]

Ratios: [0.317, 0.603, 1.0, 1.658, 3.151]
Scales: [0.4, 0.46, 0.528, 0.607, 0.698]


'''
