import contextlib
import datetime
import hashlib
import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817 — PyTorch standard alias
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiplicativeLR, OneCycleLR, PolynomialLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    RandomGrayscale,
    RandomPhotometricDistort,
)

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import transforms as T  # noqa: N812 — doctr library conventional alias
from doctr.datasets import DetectionDataset
from doctr.models import detection, login_to_hub, push_to_hf_hub
from doctr.utils.metrics import LocalizationConfusion

from .utils import EarlyStopper, plot_recorder, plot_samples

ProgressHook = Callable[[dict[str, Any]], None]


def _emit_progress(progress_hook: ProgressHook | None, **payload: Any) -> None:
    if progress_hook is None:
        return
    with contextlib.suppress(Exception):
        # Progress reporting must never break training.
        progress_hook(payload)


def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
):
    """Gridsearch the optimal learning rate for the training."""
    if num_it > len(train_loader):
        raise ValueError(
            "the value of `num_it` needs to be lower than the number of available batches"
        )

    model = model.train()
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = torch.amp.GradScaler("cuda")

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast("cuda"):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()

        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


def fit_one_epoch(
    model,
    device,
    train_loader,
    batch_transforms,
    optimizer,
    scheduler,
    amp=False,
    log=None,
    rank=0,
    progress_hook: ProgressHook | None = None,
):
    if amp:
        scaler = torch.amp.GradScaler("cuda")

    model.train()
    epoch_train_loss, batch_cnt = 0, 0
    iterator = (
        tqdm(train_loader, dynamic_ncols=True, disable=(rank != 0))
        if progress_hook is None
        else train_loader
    )
    total_batches = len(train_loader)
    for batch_idx, (images, targets) in enumerate(iterator, start=1):
        if torch.cuda.is_available():
            images = images.to(device)
        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast("cuda"):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
        last_lr = scheduler.get_last_lr()[0]

        if progress_hook is None:
            iterator.set_description(f"Training loss: {train_loss.item():.6} | LR: {last_lr:.6}")
        else:
            _emit_progress(
                progress_hook,
                event="train_batch",
                loss=float(train_loss.item()),
                lr=float(last_lr),
                batch=batch_idx,
                total_batches=total_batches,
            )
        if log:
            log(train_loss=train_loss.item(), lr=last_lr)

        epoch_train_loss += train_loss.item()
        batch_cnt += 1

    epoch_train_loss /= batch_cnt
    return epoch_train_loss, last_lr


@torch.no_grad()
def evaluate(model, device, val_loader, batch_transforms, val_metric, amp=False, log=None):
    model.eval()
    val_metric.reset()
    val_loss, batch_cnt = 0, 0
    pbar = tqdm(val_loader, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device)
        images = batch_transforms(images)
        if amp:
            with torch.amp.autocast("cuda"):
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)

        loc_preds = out["preds"]
        for target, loc_pred in zip(targets, loc_preds, strict=False):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values(), strict=False):
                if (
                    isinstance(boxes_pred, np.ndarray)
                    and boxes_pred.ndim == 2
                    and boxes_pred.shape[1] == 5
                ):
                    boxes_pred = boxes_pred[:, :4]
                val_metric.update(
                    gts=boxes_gt,
                    preds=boxes_pred if len(boxes_pred) else np.zeros((0, 4)),
                )

        pbar.set_description(f"Validation loss: {out['loss'].item():.6}")
        if log:
            log(val_loss=out["loss"].item())

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


@torch.no_grad()
def evaluate_with_progress(
    model,
    device,
    val_loader,
    batch_transforms,
    val_metric,
    amp=False,
    log=None,
    progress_hook: ProgressHook | None = None,
):
    model.eval()
    val_metric.reset()
    val_loss, batch_cnt = 0, 0
    iterator = tqdm(val_loader, dynamic_ncols=True) if progress_hook is None else val_loader
    total_batches = len(val_loader)
    for batch_idx, (images, targets) in enumerate(iterator, start=1):
        images = images.to(device)
        images = batch_transforms(images)
        if amp:
            with torch.amp.autocast("cuda"):
                out = model(images, targets, return_preds=True)
        else:
            out = model(images, targets, return_preds=True)

        loc_preds = out["preds"]
        for target, loc_pred in zip(targets, loc_preds, strict=False):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values(), strict=False):
                if (
                    isinstance(boxes_pred, np.ndarray)
                    and boxes_pred.ndim == 2
                    and boxes_pred.shape[1] == 5
                ):
                    boxes_pred = boxes_pred[:, :4]
                val_metric.update(
                    gts=boxes_gt,
                    preds=boxes_pred if len(boxes_pred) else np.zeros((0, 4)),
                )

        if progress_hook is None:
            iterator.set_description(f"Validation loss: {out['loss'].item():.6}")
        else:
            _emit_progress(
                progress_hook,
                event="val_batch",
                loss=float(out["loss"].item()),
                batch=batch_idx,
                total_batches=total_batches,
            )
        if log:
            log(val_loss=out["loss"].item())

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args, progress_hook: ProgressHook | None = None):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    if distributed:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=args.backend)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        rank = 0
        if isinstance(args.device, int):
            if not torch.cuda.is_available():
                raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
            if args.device >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            device = torch.device("cuda", args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda", 0)
        else:
            logging.warning("No accessible GPU, target device set to CPU.")
            device = torch.device("cpu")

    slack_token = os.getenv("TQDM_SLACK_TOKEN")
    slack_channel = os.getenv("TQDM_SLACK_CHANNEL")

    pbar = None
    if progress_hook is None:
        pbar = tqdm(disable=not ((slack_token and slack_channel) and (rank == 0)))
        if slack_token and slack_channel:
            pbar.write = lambda msg: pbar.sio.client.chat_postMessage(
                channel=slack_channel, text=msg
            )

    def log_line(message: str) -> None:
        if pbar is not None:
            pbar.write(message)
        else:
            logging.info(message)
        _emit_progress(progress_hook, event="log", message=message)

    log_line(str(args))

    if rank == 0 and args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, multiprocessing.cpu_count())

    torch.backends.cudnn.benchmark = True

    if rank == 0:
        st = time.time()
        val_set = DetectionDataset(
            img_folder=os.path.join(args.val_path, "images"),
            label_path=os.path.join(args.val_path, "labels.json"),
            sample_transforms=T.SampleCompose(
                [
                    T.Resize(
                        (args.input_size, args.input_size),
                        preserve_aspect_ratio=True,
                        symmetric_pad=True,
                    ),
                ]
                if not args.rotation
                else [
                    T.Resize(args.input_size, preserve_aspect_ratio=True),
                    T.RandomApply(T.RandomRotate(90, expand=True), 0.5),
                    T.Resize(
                        (args.input_size, args.input_size),
                        preserve_aspect_ratio=True,
                        symmetric_pad=True,
                    ),
                ]
            ),
            use_polygons=args.rotation,
        )
        with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
            val_hash = hashlib.sha256(f.read()).hexdigest()

        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.workers,
            sampler=SequentialSampler(val_set),
            pin_memory=torch.cuda.is_available(),
            collate_fn=val_set.collate_fn,
        )
        log_line(
            f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in {len(val_loader)} batches)"
        )

        class_names = val_set.class_names

    else:
        val_hash = None
        class_names = ["words"]

    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))

    model = detection.__dict__[args.arch](
        pretrained=args.pretrained,
        assume_straight_pages=not args.rotation,
        class_names=class_names,
    )

    if isinstance(args.resume, str):
        log_line(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        val_metric = LocalizationConfusion(use_polygons=args.rotation)

    if rank == 0 and args.test_only:
        log_line("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate_with_progress(
            model,
            device,
            val_loader,
            batch_transforms,
            val_metric,
            amp=args.amp,
            progress_hook=progress_hook,
        )
        log_line(
            f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
        )
        return

    st = time.time()
    with open(os.path.join(args.train_path, "labels.json"), "rb") as f:
        train_hash = hashlib.sha256(f.read()).hexdigest()

    img_transforms = T.OneOf(
        [
            Compose(
                [
                    T.RandomApply(T.ColorInversion(), 0.3),
                    T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.2),
                ]
            ),
            Compose(
                [
                    T.RandomApply(T.RandomShadow(), 0.3),
                    T.RandomApply(T.GaussianNoise(), 0.1),
                    T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                    RandomGrayscale(p=0.15),
                ]
            ),
            RandomPhotometricDistort(p=0.3),
            lambda x: x,
        ]
    )

    sample_transforms = T.SampleCompose(
        [
            T.RandomHorizontalFlip(0.15),
            T.OneOf(
                [
                    T.RandomApply(T.RandomCrop(ratio=(0.6, 1.33)), 0.25),
                    T.RandomResize(
                        scale_range=(0.4, 0.9), preserve_aspect_ratio=0.5, symmetric_pad=0.5, p=0.25
                    ),
                ]
            ),
            T.Resize(
                (args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True
            ),
        ]
        if not args.rotation
        else [
            T.RandomHorizontalFlip(0.15),
            T.OneOf(
                [
                    T.RandomApply(T.RandomCrop(ratio=(0.6, 1.33)), 0.25),
                    T.RandomResize(
                        scale_range=(0.4, 0.9), preserve_aspect_ratio=0.5, symmetric_pad=0.5, p=0.25
                    ),
                ]
            ),
            T.Resize(args.input_size, preserve_aspect_ratio=True),
            T.RandomApply(T.RandomRotate(90, expand=True), 0.5),
            T.Resize(
                (args.input_size, args.input_size), preserve_aspect_ratio=True, symmetric_pad=True
            ),
        ]
    )

    train_set = DetectionDataset(
        img_folder=os.path.join(args.train_path, "images"),
        label_path=os.path.join(args.train_path, "labels.json"),
        img_transforms=img_transforms,
        sample_transforms=sample_transforms,
        use_polygons=args.rotation,
    )

    if distributed:
        sampler = DistributedSampler(train_set, rank=rank, shuffle=True, drop_last=True)
    else:
        sampler = RandomSampler(train_set)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        collate_fn=train_set.collate_fn,
    )
    if rank == 0:
        log_line(
            f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in {len(train_loader)} batches)"
        )

    if rank == 0 and args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target)
        return

    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.95, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            args.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=args.weight_decay or 1e-4,
        )

    if rank == 0 and args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4
        )
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))
    elif args.sched == "poly":
        scheduler = PolynomialLR(optimizer, args.epochs * len(train_loader))

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    if rank == 0:
        config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "architecture": args.arch,
            "input_size": args.input_size,
            "optimizer": args.optim,
            "framework": "pytorch",
            "scheduler": args.sched,
            "rotation": args.rotation,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "pretrained": args.pretrained,
            "amp": args.amp,
        }

    global global_step
    global_step = 0
    if rank == 0 and args.wb:
        import wandb

        run = wandb.init(
            name=exp_name,
            project="text-detection",
            config=config,
        )

        def wandb_log_at_step(train_loss=None, val_loss=None, lr=None):
            wandb.log(
                {
                    **({"train_loss_step": train_loss} if train_loss is not None else {}),
                    **({"val_loss_step": val_loss} if val_loss is not None else {}),
                    **({"step_lr": lr} if lr is not None else {}),
                }
            )

    if rank == 0 and args.clearml:
        from clearml import Logger, Task

        task = Task.init(
            project_name="docTR/text-detection", task_name=exp_name, reuse_last_task_id=False
        )
        task.upload_artifact("config", config)

        def clearml_log_at_step(train_loss=None, val_loss=None, lr=None):
            logger = Logger.current_logger()
            if train_loss is not None:
                logger.report_scalar(
                    title="Training Step Loss",
                    series="train_loss_step",
                    iteration=global_step,
                    value=train_loss,
                )
            if val_loss is not None:
                logger.report_scalar(
                    title="Validation Step Loss",
                    series="val_loss_step",
                    iteration=global_step,
                    value=val_loss,
                )
            if lr is not None:
                logger.report_scalar(
                    title="Step Learning Rate", series="step_lr", iteration=global_step, value=lr
                )

    def log_at_step(train_loss=None, val_loss=None, lr=None):
        global global_step
        if args.wb:
            wandb_log_at_step(train_loss, val_loss, lr)
        if args.clearml:
            clearml_log_at_step(train_loss, val_loss, lr)
        global_step += 1

    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(
            patience=args.early_stop_epochs, min_delta=args.early_stop_delta
        )

    for epoch in range(args.epochs):
        train_loss, actual_lr = fit_one_epoch(
            model,
            device,
            train_loader,
            batch_transforms,
            optimizer,
            scheduler,
            amp=args.amp,
            log=log_at_step,
            rank=rank,
            progress_hook=progress_hook,
        )

        if rank == 0:
            log_line(
                f"Epoch {epoch + 1}/{args.epochs} - Training loss: {train_loss:.6} | LR: {actual_lr:.6}"
            )

            val_loss, recall, precision, mean_iou = evaluate_with_progress(
                model,
                device,
                val_loader,
                batch_transforms,
                val_metric,
                amp=args.amp,
                log=log_at_step,
                progress_hook=progress_hook,
            )
            if val_loss < min_loss:
                log_line(
                    f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state..."
                )
                params = model.module if hasattr(model, "module") else model
                torch.save(params.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
                # Persist the detection architecture name so downstream
                # consumers can instantiate a matching model without guessing.
                try:
                    arch_path = Path(args.output_dir) / f"{exp_name}.arch"
                    arch_path.write_text(args.arch, encoding="utf-8")
                except Exception as exc:  # pragma: no cover - best effort
                    log_line(f"Warning: failed to write arch sidecar: {exc}")
                min_loss = val_loss
            log_line(
                f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})"
            )
            _emit_progress(
                progress_hook,
                event="epoch_end",
                epoch=epoch + 1,
                total_epochs=args.epochs,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                lr=float(actual_lr),
                recall=float(recall),
                precision=float(precision),
                mean_iou=float(mean_iou),
            )

            if args.wb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": actual_lr,
                        "recall": recall,
                        "precision": precision,
                        "mean_iou": mean_iou,
                    }
                )

            if args.clearml:
                from clearml import Logger

                logger = Logger.current_logger()
                logger.report_scalar(
                    title="Training Loss", series="train_loss", value=train_loss, iteration=epoch
                )
                logger.report_scalar(
                    title="Validation Loss", series="val_loss", value=val_loss, iteration=epoch
                )
                logger.report_scalar(
                    title="Learning Rate", series="lr", value=actual_lr, iteration=epoch
                )
                logger.report_scalar(title="Recall", series="recall", value=recall, iteration=epoch)
                logger.report_scalar(
                    title="Precision", series="precision", value=precision, iteration=epoch
                )
                logger.report_scalar(
                    title="Mean IoU", series="mean_iou", value=mean_iou, iteration=epoch
                )

            if args.early_stop and early_stopper.early_stop(val_loss):
                log_line("Training halted early due to reaching patience limit.")
                break

    if rank == 0:
        if args.wb:
            run.finish()

        if args.push_to_hub:
            push_to_hf_hub(model, exp_name, task="detection", run_config=args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="PGDP OCR training script for text detection (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--backend", default="nccl", type=str, help="Backend to use for torch.distributed"
    )
    parser.add_argument("arch", type=str, help="text-detection model to train")
    parser.add_argument(
        "--output_dir", type=str, default=".", help="path to save checkpoints and final model"
    )
    parser.add_argument("--train_path", type=str, required=True, help="path to train data folder")
    parser.add_argument("--val_path", type=str, required=True, help="path to val data folder")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to train the model on"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument(
        "--input_size", type=int, default=1024, help="input size H (and W) for the model"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=int,
        help="Specify gpu device for single-gpu training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate for the optimizer")
    parser.add_argument(
        "--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay"
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=None, help="number of workers used for dataloading"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument(
        "--rotation",
        dest="rotation",
        action="store_true",
        help="Use rotated bounding boxes (polygons) and disable assume_straight_pages",
    )
    parser.add_argument(
        "--test-only", dest="test_only", action="store_true", help="Run the validation loop"
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="freeze model backbone for fine-tuning",
    )
    parser.add_argument(
        "--show-samples",
        dest="show_samples",
        action="store_true",
        help="Display unormalized training samples",
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--clearml", dest="clearml", action="store_true", help="Log to ClearML")
    parser.add_argument(
        "--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub"
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    parser.add_argument(
        "--optim", type=str, default="adam", choices=["adam", "adamw"], help="optimizer to use"
    )
    parser.add_argument(
        "--sched",
        type=str,
        default="poly",
        choices=["cosine", "onecycle", "poly"],
        help="scheduler to use",
    )
    parser.add_argument(
        "--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true"
    )
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument(
        "--early-stop-epochs", type=int, default=5, help="Patience for early stopping"
    )
    parser.add_argument(
        "--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping"
    )
    args = parser.parse_args()

    return args


def detect_from_config(
    train_path: str | Path,
    val_path: str | Path,
    arch: str = "db_resnet50",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 0.002,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    scheduler: str = "poly",
    input_size: int = 1024,
    rotation: bool = False,
    workers: int = 4,
    amp: bool = False,
    early_stop: bool = False,
    early_stop_epochs: int = 5,
    early_stop_delta: float = 0.01,
    output_dir: str = ".",
    device: int | None = None,
    pretrained: bool = True,
    name: str | None = None,
    progress_hook: ProgressHook | None = None,
) -> None:
    """Run detection fine-tuning with simplified configuration, suitable for UI calls.

    Args:
        train_path: Path to training data folder (must contain images/ and labels.json)
        val_path: Path to validation data folder (must contain images/ and labels.json)
        arch: Detection model architecture name (e.g. "db_resnet50")
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay
        optimizer: Optimizer type ("adam" or "adamw")
        scheduler: LR scheduler type ("cosine", "onecycle", or "poly")
        input_size: Input image height and width (square)
        rotation: Whether to use rotated bounding boxes / polygons
        workers: Number of data loading workers
        amp: Whether to use automatic mixed precision
        early_stop: Whether to enable early stopping
        early_stop_epochs: Patience for early stopping
        early_stop_delta: Minimum delta for early stopping
        output_dir: Directory to save model checkpoints
        device: GPU device index
        pretrained: Whether to initialize from pretrained weights
        name: Experiment/model name used for output checkpoint naming
    """
    import argparse

    args = argparse.Namespace(
        arch=arch,
        train_path=str(train_path),
        val_path=str(val_path),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        optim=optimizer,
        sched=scheduler,
        input_size=input_size,
        rotation=rotation,
        workers=workers if isinstance(workers, int) else None,
        amp=amp,
        early_stop=early_stop,
        early_stop_epochs=early_stop_epochs,
        early_stop_delta=early_stop_delta,
        output_dir=output_dir,
        device=device,
        backend="nccl",
        name=name,
        resume=None,
        freeze_backbone=False,
        test_only=False,
        show_samples=False,
        wb=False,
        clearml=False,
        push_to_hub=False,
        pretrained=pretrained,
        find_lr=False,
    )

    main(args, progress_hook=progress_hook)


if __name__ == "__main__":
    args = parse_args()
    main(args)
