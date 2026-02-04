"""Mask RCNN Instance Segmentation Training.
Modified from here
https://github.com/pytorch/vision/blob/master/references/detection/train.py
"""

import datetime
import os
import time
import torch
import torchvision
import torch.utils.data
from annolid.utils.config import get_config
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from annolid.segmentation.maskrcnn.group_by_aspect_ratio import (
    GroupedBatchSampler,
    create_aspect_ratio_groups,
)
from annolid.segmentation.maskrcnn.engine import train_one_epoch, evaluate
from annolid.segmentation.maskrcnn import transforms as T
from annolid.segmentation.maskrcnn.coco_utils import get_coco, get_coco_kp
from annolid.segmentation.maskrcnn import utils
from annolid.segmentation.maskrcnn.model import get_maskrcnn_model


def get_transform(train):
    transforms = []
    # convert PIL image to pytorch tensor
    transforms.append(T.ToTensor())

    if train:
        # random flip the images
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_dataset(name, image_set, transform, data_path, num_classes=91):
    paths = {
        "coco": (data_path, get_coco, num_classes),
        "coco_kp": (data_path, get_coco_kp, 2),
    }

    p, dataset_func, num_classes = paths[name]
    dataset = dataset_func(p, image_set=image_set, transforms=transform)
    config_file = os.path.join(data_path, "data.yaml")

    if os.path.isfile(config_file):
        custom_config = get_config(config_file)
        num_classes = len(custom_config.DATASET.class_names) + 1

    return dataset, num_classes


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading
    print("Loading dataset")
    dataset, num_classes = get_dataset(
        args.dataset, "train", get_transform(train=True), args.data_path
    )

    dataset_test, _ = get_dataset(
        args.dataset, "valid", get_transform(train=False), args.data_path
    )

    print("Create data loaders")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(
            dataset, k=args.aspect_ratio_group_factor
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, args.batch_size
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True
        )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=train_batch_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        sampler=test_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
    )

    print("Creating model")
    if "mask" in args.model:
        model = get_maskrcnn_model(num_classes)
    else:
        model = torchvision.models.detection.__dict__[args.model](
            num_classes=num_classes, pretrained=args.pretrained
        )

    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model.torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_steps, gamma=args.lr_gamma
    )

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)

    print("Start training")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()

        if args.output_dir:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"model_{epoch}.pth"),
            )
        # evaluate after each epoch
        evaluate(model, data_loader_test, device=device)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data-path", default="/content/datasets/", help="dataset")
    parser.add_argument("--dataset", default="coco", help="dataset")
    parser.add_argument("--model", default="maskrcnn_resnet50_fpn", help="model")
    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument(
        "-b", "--batch_size", default=2, type=int, help="images per gpu"
    )
    parser.add_argument(
        "--epochs", default=26, type=int, metavar="N", help="number of totoal epochs"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers",
    )
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value "
        "on 8 gpus and 2 images per gpu ",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="Momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay",
        dest="weight_decay",
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step size epochs"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.1,
        type=float,
        help="decrease lr by a factor of lr-gamma",
    )
    parser.add_argument(
        "--print-freq",
        default=10,
        type=int,
        help="print frequency",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        dest="output_dir",
        help="folder to save",
    )
    parser.add_argument(
        "--resume",
        default="",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="start epoch",
    )
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=True,
        help="Use pre-trained models",
        action="store_true",
    )

    # distributed training params
    parser.add_argument(
        "--world-size", default=1, type=int, help="num of distributed processes"
    )
    parser.add_argument(
        "--dist-url", default="env://", help="url to setup distributed training"
    )

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
