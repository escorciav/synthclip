# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from collections import OrderedDict
import json
import math
import os
import sys
import time
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import ImageFolder

import datasets
import models
from tokenizer import SimpleTokenizer
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="SLIP training and evaluation", add_help=False
    )
    # Data
    parser.add_argument(
        "--dataset",
        default="synthclip",
        type=str,
        choices=["cc3m", "cc12m", "synthclip"],
    )
    parser.add_argument(
        "--dataset-type", default="csv", type=str, choices=["csv", "webdataset"]
    )
    parser.add_argument("--train-data", default=None, type=str)
    parser.add_argument("--csv-img-key", default="image_path", type=str)
    parser.add_argument("--csv-caption-key", default="caption", type=str)
    parser.add_argument("--csv_separator", default="\t", type=str)
    parser.add_argument("--csv_prefix", default=None, type=Path)
    parser.add_argument("--train-num-samples", default=2576776, type=int)
    parser.add_argument("--train-data-upsampling-factors", default=None, type=int)
    parser.add_argument(
        "--imagenet-root",
        default="/path/to/ImageNet",
        type=str,
        help="path to ImageNet root",
    )
    parser.add_argument(
        "--imagenet-val-dir",
        default="/path/to/ImageNet",
        type=str,
        help="path to ImageNet val root",
    )
    parser.add_argument(
        "--output-dir", default="vitb16-synthclip", type=str, help="output dir"
    )

    # Model
    parser.add_argument("--model", default="CLIP_VITB16", type=str)
    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    parser.add_argument("--resume-only-weights", default=0, choices=[0, 1], type=int,
                        help="0: weights, epochs, optim; 1: only weights")

    # Training
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=512,
        type=int,
        help="number of samples per-device/per-gpu",
    )
    parser.add_argument("--lr", default=3e-3, type=float)
    parser.add_argument(
        "--lr-start", default=1e-6, type=float, help="initial warmup lr"
    )
    parser.add_argument("--lr-end", default=1e-5, type=float, help="minimum final lr")
    parser.add_argument(
        "--update-freq",
        default=1,
        type=int,
        help="optimizer update frequency (i.e. gradient accumulation steps)",
    )
    parser.add_argument("--wd", default=0.1, type=float)
    parser.add_argument("--betas", default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--eval-freq", default=1, type=int)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="disable mixed-precision training (requires more memory and compute)",
    )
    parser.add_argument(
        "--gather-with-grad", action="store_true", help="gather features with gradients"
    )
    # System
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument(
        "-j",
        "--workers",
        default=6,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument("--evaluate", action="store_true", help="eval only")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--save-images", action="store_true", help="Save images during training")
    return parser


best_acc1 = 0


def main(args):

    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    utils.init_distributed_mode(args)
    global best_acc1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = getattr(models, args.model)()

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], bucket_cap_mb=200
        )

    # define loss function (criterion) and optimizer
    criterion = models.get_loss(args.gather_with_grad).cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [
        {"params": p_wd, "weight_decay": args.wd},
        {"params": p_non_wd, "weight_decay": 0},
    ]

    optimizer = torch.optim.AdamW(
        optim_params, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.wd
    )
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cpu")
            if args.resume_only_weights == 0:
                epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
                args.start_epoch = epoch
                result = model.load_state_dict(checkpoint["state_dict"], strict=False)
                print(result)
                (
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    if "optimizer" in checkpoint
                    else ()
                )
                (
                    scaler.load_state_dict(checkpoint["scaler"])
                    if "scaler" in checkpoint
                    else ()
                )
                best_acc1 = checkpoint["best_acc1"]
                print(
                    "=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch)
                )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, "checkpoint.pt")
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location="cpu")
            args.start_epoch = latest_checkpoint["epoch"]
            model.load_state_dict(latest_checkpoint["state_dict"])
            optimizer.load_state_dict(latest_checkpoint["optimizer"])
            scaler.load_state_dict(latest_checkpoint["scaler"])
            best_acc1 = latest_checkpoint["best_acc1"]
            print(
                "=> loaded latest checkpoint '{}' (epoch {})".format(
                    latest, latest_checkpoint["epoch"]
                )
            )

    cudnn.benchmark = True

    # Data loading code
    print("=> creating dataset")
    tokenizer = SimpleTokenizer()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.get_train_data(
        args, train_transform, args.start_epoch, tokenizer
    )

    if os.path.exists(args.imagenet_root):
        imagenet_val_dir = os.path.join(args.imagenet_root, "val")
    elif os.path.exists(args.imagenet_val_dir):
        imagenet_val_dir = args.imagenet_val_dir
    val_dataset = ImageFolder(imagenet_val_dir, val_transform)

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=(val_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    if args.evaluate:
        if args.model.startswith("SIMCLR"):
            print("zero-shot evaluation not supported with ssl-only model.")
            return

        zero_stats = validate_zeroshot(val_loader, model, tokenizer, args)
        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "eval_log.txt"), "a") as f:
                f.write(json.dumps(zero_stats) + "\n")
        return

    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.lr_end,
        args.epochs,
        train_dataset.dataloader.num_batches // args.update_freq,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.lr_start,
    )

    if utils.is_main_process() and args.wandb:
        wandb_id = os.path.split(args.output_dir)[-1]
        wandb.init(project="slip", id=wandb_id, config=args, resume="allow")

    print(args)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        train_dataset.set_epoch(epoch)
        train_loader = train_dataset.dataloader

        # train for one epoch
        train_stats = train(
            train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args
        )

        if (epoch + 1) % args.eval_freq != 0:
            continue

        if args.model.startswith("SIMCLR"):
            val_stats = {"acc1": -1}
            acc1 = -1
        else:
            val_stats = validate_zeroshot(val_loader, model, tokenizer, args)
            acc1 = val_stats["acc1"]

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        print("=> saving checkpoint")
        utils.save_on_master(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc1": best_acc1,
                "args": args,
            },
            is_best,
            args.output_dir,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in val_stats.items()},
            "epoch": epoch,
        }

        if utils.is_main_process():
            if args.wandb:
                wandb.log(log_stats)
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")


def train(train_loader, model, criterion, optimizer, scaler, epoch, lr_schedule, args):
    batch_time = AverageMeter("Time", ":6.2f")
    data_time = AverageMeter("Data", ":6.2f")
    mem = AverageMeter("Mem (GB)", ":6.1f")
    metric_names = models.get_metric_names()
    iters_per_epoch = train_loader.num_batches // args.update_freq

    metrics = OrderedDict([(name, AverageMeter(name, ":.2e")) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]

        if len(inputs) == 3:
            *inputs, raw_text = inputs
        inputs = [tensor.cuda(args.gpu, non_blocking=True) for tensor in inputs]

        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(*inputs)
            loss_dict = criterion(outputs)
            loss = loss_dict["loss"]
            loss /= args.update_freq

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        utils.get_model(model).logit_scale.data.clamp_(0, 4.6052)
        logit_scale = utils.get_model(model).logit_scale.exp().item()

        for k in loss_dict:
            metrics[k].update(loss_dict[k].item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        if optim_iter % args.print_freq == 0:
            if utils.is_main_process() and args.wandb:
                wandb.log(
                    {
                        **{k: v.item() for k, v in loss_dict.items()},
                        "scaler": scaler.get_scale(),
                        "logit": logit_scale,
                    }
                )
            if utils.is_main_process() and (data_iter == 0 or args.save_images):
                # Save a handful of images from the first batch of each epoch
                save_images(inputs[0], raw_text, epoch, args.output_dir)
            progress.display(optim_iter)

    progress.synchronize()
    # import ipdb; ipdb.set_trace()
    return {
        **{k: v.avg for k, v in metrics.items()},
        "lr": optimizer.param_groups[0]["lr"],
        "logit_scale": logit_scale,
    }


def validate_zeroshot(val_loader, model, tokenizer, args):
    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    print("=> encoding captions")
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, "templates.json")) as f:
        templates = json.load(f)["imagenet"]

    with open(os.path.join(cwd, "labels.json")) as f:
        labels = json.load(f)["imagenet"]

    with torch.no_grad():
        text_features = []
        for l in labels:
            texts = [t.format(l) for t in templates]
            texts = tokenizer(texts).cuda(args.gpu, non_blocking=True)
            class_embeddings = utils.get_model(model).encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # encode images
            image_features = utils.get_model(model).encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(logits_per_image, target, topk=(1, 5))
            acc1, acc5 = utils.scaled_all_reduce([acc1, acc5])
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.synchronize()
    print(
        "0-shot * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
    )
    return {"acc1": top1.avg, "acc5": top5.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_images(inputs, raw_text, epoch, output_dir, num_images=5, font_size=12):
    # Create a grid of images
    grid = vutils.make_grid(inputs[:num_images], nrow=num_images, 
                           normalize=True, scale_each=True)
    # Convert the grid to a numpy array 
    np_grid = grid.cpu().numpy().transpose((1, 2, 0)) * 255
    # Convert the numpy array to an image
    img = Image.fromarray(np_grid.astype('uint8'))
    
    # Create a new image with extra height for the text
    extra_height = int(font_size * 1.2 * num_images)
    final_img = Image.new('RGB', (img.width, img.height + extra_height), (255, 255, 255))
    # Paste the image grid
    final_img.paste(img, (0, 0))
    
    # Add the text below the images
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype('arial.ttf', font_size)
    except OSError:
        font = ImageFont.load_default()
        
    captions = raw_text[:num_images]
    caption_text = '\n'.join(str(c) for c in captions)
        
    draw.text((10, img.height + 5), caption_text, fill=(0, 0, 0), font=font)
    
    # Save the final image
    filename = f'images-epoch_{epoch}.jpg'
    final_img.save(os.path.join(output_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SLIP training and evaluation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
