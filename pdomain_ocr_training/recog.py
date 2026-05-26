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
    RandomPerspective,
    RandomPhotometricDistort,
)

if os.getenv("TQDM_SLACK_TOKEN") and os.getenv("TQDM_SLACK_CHANNEL"):
    from tqdm.contrib.slack import tqdm
else:
    from tqdm.auto import tqdm

from doctr import datasets
from doctr import transforms as T  # noqa: N812 — doctr library conventional alias
from doctr.datasets import VOCABS, RecognitionDataset, WordGenerator
from doctr.models import login_to_hub, push_to_hf_hub, recognition
from doctr.utils.metrics import TextMatch

from .utils import EarlyStopper, plot_recorder, plot_samples

ProgressHook = Callable[[dict[str, Any]], None]


def _emit_progress(progress_hook: ProgressHook | None, **payload: Any) -> None:
    if progress_hook is None:
        return
    with contextlib.suppress(Exception):
        # Progress reporting must never break training.
        progress_hook(payload)


def resolve_vocab(vocab_arg: str) -> str:
    """Resolve a vocab argument to a character string.

    Supports:
      - Named built-in vocabs (e.g. "french", "english")
      - Custom vocabs via "CUSTOM:<chars>" prefix (e.g. "CUSTOM:abc0123")
    """
    if vocab_arg.startswith("CUSTOM:"):
        custom_chars = vocab_arg.split(":", 1)[1]
        if not custom_chars:
            raise ValueError("Custom vocab cannot be empty. Use CUSTOM:<characters>")
        return "".join(sorted(set(custom_chars)))
    if vocab_arg not in VOCABS:
        raise ValueError(
            f"Unknown vocab '{vocab_arg}'. "
            f"Available built-in vocabs: {sorted(VOCABS.keys())}. "
            "Or use CUSTOM:<characters> to define your own."
        )
    return VOCABS[vocab_arg]


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
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if num_it > len(train_loader):
        raise ValueError(
            "the value of `num_it` needs to be lower than the number of available batches"
        )

    model = model.train()
    # Update param groups & LR
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

        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with torch.amp.autocast("cuda"):
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
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
            # Update LR
            scheduler.step()

        # Record
        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        # Stop after the number of iterations
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
    # Iterate over the batches of the dataset
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
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
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
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
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
        # Compute metric
        if len(out["preds"]):
            words, _ = zip(*out["preds"], strict=False)
        else:
            words = []
        val_metric.update(targets, words)

        pbar.set_description(f"Validation loss: {out['loss'].item():.6}")
        if log:
            log(val_loss=out["loss"].item())

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


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
        if len(out["preds"]):
            words, _ = zip(*out["preds"], strict=False)
        else:
            words = []
        val_metric.update(targets, words)

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
    result = val_metric.summary()
    return val_loss, result["raw"], result["unicase"]


def main(args, progress_hook: ProgressHook | None = None):
    # Detect distributed setup
    # variable is set by torchrun
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    # GPU setup
    if distributed:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend=args.backend)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        # single process
        rank = 0
        if isinstance(args.device, int):
            if not torch.cuda.is_available():
                raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
            if args.device >= torch.cuda.device_count():
                raise ValueError("Invalid device index")
            device = torch.device("cuda", args.device)
        # Silent default switch to GPU if available
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
            # Monkey patch tqdm write method to send messages directly to Slack
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

    vocab = resolve_vocab(args.vocab)
    fonts = args.font.split(",")

    if rank == 0:
        # Load val data generator
        st = time.time()
        if isinstance(args.val_path, str):
            with open(os.path.join(args.val_path, "labels.json"), "rb") as f:
                val_hash = hashlib.sha256(f.read()).hexdigest()

            val_set = RecognitionDataset(
                img_folder=os.path.join(args.val_path, "images"),
                labels_path=os.path.join(args.val_path, "labels.json"),
                img_transforms=T.Resize(
                    (args.input_size, 4 * args.input_size), preserve_aspect_ratio=True
                ),
            )
        elif args.val_datasets:
            val_hash = None
            val_datasets = args.val_datasets

            val_set = datasets.__dict__[val_datasets[0]](
                train=False,
                download=True,
                recognition_task=True,
                use_polygons=True,
                img_transforms=Compose(
                    [
                        T.Resize(
                            (args.input_size, 4 * args.input_size), preserve_aspect_ratio=True
                        ),
                        # Augmentations
                        T.RandomApply(T.ColorInversion(), 0.1),
                    ]
                ),
            )
            if len(val_datasets) > 1:
                for dataset_name in val_datasets[1:]:
                    _ds = datasets.__dict__[dataset_name](
                        train=False,
                        download=True,
                        recognition_task=True,
                        use_polygons=True,
                    )
                    val_set.data.extend((np_img, target) for np_img, target in _ds.data)
        else:
            val_hash = None
            # Load synthetic data generator
            val_set = WordGenerator(
                vocab=vocab,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                num_samples=args.val_samples * len(vocab),
                font_family=fonts,
                img_transforms=Compose(
                    [
                        T.Resize(
                            (args.input_size, 4 * args.input_size), preserve_aspect_ratio=True
                        ),
                        # Ensure we have a 90% split of white-background images
                        T.RandomApply(T.ColorInversion(), 0.9),
                    ]
                ),
            )

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

    batch_transforms = Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301))

    # Load doctr model
    model = recognition.__dict__[args.arch](pretrained=args.pretrained, vocab=vocab)

    # Resume weights
    if isinstance(args.resume, str):
        log_line(f"Resuming {args.resume}")
        model.from_pretrained(args.resume)

    # Backbone freezing
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        model = model.to(device)

    if distributed:
        # construct DDP model
        model = DDP(model, device_ids=[rank])

    if rank == 0:
        # Metrics
        val_metric = TextMatch()

    if rank == 0 and args.test_only:
        log_line("Running evaluation")
        val_loss, exact_match, partial_match = evaluate_with_progress(
            model,
            device,
            val_loader,
            batch_transforms,
            val_metric,
            amp=args.amp,
            progress_hook=progress_hook,
        )
        log_line(
            f"Validation loss: {val_loss:.6} (Exact: {exact_match:.2%} | Partial: {partial_match:.2%})"
        )
        return

    st = time.time()

    if isinstance(args.train_path, str):
        # Load train data generator
        base_path = Path(args.train_path)
        parts = (
            [base_path]
            if base_path.joinpath("labels.json").is_file()
            else [base_path.joinpath(sub) for sub in os.listdir(base_path)]
        )
        with open(parts[0].joinpath("labels.json"), "rb") as f:
            train_hash = hashlib.sha256(f.read()).hexdigest()

        train_set = RecognitionDataset(
            parts[0].joinpath("images"),
            parts[0].joinpath("labels.json"),
            img_transforms=Compose(
                [
                    T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                    # Augmentations
                    T.RandomApply(T.ColorInversion(), 0.1),
                    RandomGrayscale(p=0.1),
                    RandomPhotometricDistort(p=0.1),
                    T.RandomApply(T.RandomShadow(), p=0.4),
                    T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                    T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                    RandomPerspective(distortion_scale=0.2, p=0.3),
                ]
            ),
        )
        if len(parts) > 1:
            for subfolder in parts[1:]:
                train_set.merge_dataset(
                    RecognitionDataset(
                        subfolder.joinpath("images"), subfolder.joinpath("labels.json")
                    )
                )
    elif args.train_datasets:
        train_hash = None
        train_datasets = args.train_datasets

        train_set = datasets.__dict__[train_datasets[0]](
            train=True,
            download=True,
            recognition_task=True,
            use_polygons=True,
            img_transforms=Compose(
                [
                    T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                    # Augmentations
                    T.RandomApply(T.ColorInversion(), 0.1),
                ]
            ),
        )
        if len(train_datasets) > 1:
            for dataset_name in train_datasets[1:]:
                _ds = datasets.__dict__[dataset_name](
                    train=True,
                    download=True,
                    recognition_task=True,
                    use_polygons=True,
                )
                train_set.data.extend((np_img, target) for np_img, target in _ds.data)
    else:
        train_hash = None
        # Load synthetic data generator
        train_set = WordGenerator(
            vocab=vocab,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            num_samples=args.train_samples * len(vocab),
            font_family=fonts,
            img_transforms=Compose(
                [
                    T.Resize((args.input_size, 4 * args.input_size), preserve_aspect_ratio=True),
                    # Ensure we have a 90% split of white-background images
                    T.RandomApply(T.ColorInversion(), 0.9),
                    RandomGrayscale(p=0.1),
                    RandomPhotometricDistort(p=0.1),
                    T.RandomApply(T.RandomShadow(), p=0.4),
                    T.RandomApply(T.GaussianNoise(mean=0, std=0.1), 0.1),
                    T.RandomApply(T.GaussianBlur(sigma=(0.5, 1.5)), 0.3),
                    RandomPerspective(distortion_scale=0.2, p=0.3),
                ]
            ),
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

    # Optimizer
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

    # LR finder
    if rank == 0 and args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Scheduler
    if args.sched == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4
        )
    elif args.sched == "onecycle":
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))
    elif args.sched == "poly":
        scheduler = PolynomialLR(optimizer, args.epochs * len(train_loader))

    # Training monitoring
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
            "vocab": args.vocab,
            "train_hash": train_hash,
            "val_hash": val_hash,
            "pretrained": args.pretrained,
            "amp": args.amp,
        }

    global global_step
    global_step = 0  # Shared global step counter
    # W&B
    if rank == 0 and args.wb:
        import wandb

        run = wandb.init(
            name=exp_name,
            project="text-recognition",
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

    # ClearML
    if rank == 0 and args.clearml:
        from clearml import Logger, Task

        task = Task.init(
            project_name="docTR/text-recognition", task_name=exp_name, reuse_last_task_id=False
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
                    title="Step Learning Rate",
                    series="step_lr",
                    iteration=global_step,
                    value=lr,
                )

    def log_at_step(train_loss=None, val_loss=None, lr=None):
        global global_step
        if args.wb:
            wandb_log_at_step(train_loss, val_loss, lr)
        if args.clearml:
            clearml_log_at_step(train_loss, val_loss, lr)
        global_step += 1  # Increment the shared global step counter

    # Create loss queue
    min_loss = np.inf
    if args.early_stop:
        early_stopper = EarlyStopper(
            patience=args.early_stop_epochs, min_delta=args.early_stop_delta
        )
    # Training loop
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

            # Validation loop at the end of each epoch
            val_loss, exact_match, partial_match = evaluate_with_progress(
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
                # All processes should see same parameters as they all start from same
                # random parameters and gradients are synchronized in backward passes.
                # Therefore, saving it in one process is sufficient.
                log_line(
                    f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state..."
                )
                params = model.module if hasattr(model, "module") else model

                torch.save(params.state_dict(), Path(args.output_dir) / f"{exp_name}.pt")
                # Persist the resolved vocab as a sidecar file so downstream
                # consumers (e.g. the labeler) can build a matching predictor.
                try:
                    vocab_path = Path(args.output_dir) / f"{exp_name}.vocab"
                    vocab_path.write_text(vocab, encoding="utf-8")
                except Exception as exc:  # pragma: no cover - best effort
                    log_line(f"Warning: failed to write vocab sidecar: {exc}")
                # Persist the recognition architecture name so downstream
                # consumers can instantiate a matching model without guessing.
                try:
                    arch_path = Path(args.output_dir) / f"{exp_name}.arch"
                    arch_path.write_text(args.arch, encoding="utf-8")
                except Exception as exc:  # pragma: no cover - best effort
                    log_line(f"Warning: failed to write arch sidecar: {exc}")
                min_loss = val_loss
            log_line(
                f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                f"(Exact: {exact_match:.2%} | Partial: {partial_match:.2%})"
            )
            _emit_progress(
                progress_hook,
                event="epoch_end",
                epoch=epoch + 1,
                total_epochs=args.epochs,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                lr=float(actual_lr),
                exact=float(exact_match),
                partial=float(partial_match),
            )
            # W&B
            if args.wb:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "learning_rate": actual_lr,
                        "exact_match": exact_match,
                        "partial_match": partial_match,
                    }
                )

            # ClearML
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
                logger.report_scalar(
                    title="Exact Match", series="exact_match", value=exact_match, iteration=epoch
                )
                logger.report_scalar(
                    title="Partial Match",
                    series="partial_match",
                    value=partial_match,
                    iteration=epoch,
                )

            if args.early_stop and early_stopper.early_stop(val_loss):
                log_line("Training halted early due to reaching patience limit.")
                break

    if rank == 0:
        if args.wb:
            run.finish()

        if args.push_to_hub:
            push_to_hf_hub(model, exp_name, task="recognition", run_config=args)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="PGDP OCR training script for text recognition (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DDP related args
    parser.add_argument(
        "--backend", default="nccl", type=str, help="Backend to use for torch.distributed"
    )

    parser.add_argument("arch", type=str, help="text-recognition model to train")
    parser.add_argument(
        "--output_dir", type=str, default=".", help="path to save checkpoints and final model"
    )
    parser.add_argument("--train_path", type=str, default=None, help="path to train data folder(s)")
    parser.add_argument("--val_path", type=str, default=None, help="path to val data folder")
    parser.add_argument(
        "--train_datasets",
        type=str,
        nargs="+",
        choices=["CORD", "FUNSD", "IC03", "IIIT5K", "SVHN", "SVT", "SynthText"],
        default=None,
        help="Built-in datasets to use for training",
    )
    parser.add_argument(
        "--val_datasets",
        type=str,
        nargs="+",
        choices=["CORD", "FUNSD", "IC03", "IIIT5K", "SVHN", "SVT", "SynthText"],
        default=None,
        help="Built-in datasets to use for validation",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Multiplied by the vocab length gets you the number of synthetic training samples that will be used.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=20,
        help="Multiplied by the vocab length gets you the number of synthetic validation samples that will be used.",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf",
        help="Font family to be used",
    )
    parser.add_argument(
        "--min-chars", type=int, default=1, help="Minimum number of characters per synthetic sample"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=12,
        help="Maximum number of characters per synthetic sample",
    )
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train the model on"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument(
        "--input_size", type=int, default=32, help="input size H for the model, W = 4*H"
    )
    parser.add_argument(
        "--device",
        default=None,
        type=int,
        help="Specify gpu device for single-gpu training. In distributed setting, this parameter is ignored",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam or AdamW)"
    )
    parser.add_argument(
        "--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay"
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=None, help="number of workers used for dataloading"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument(
        "--vocab",
        type=str,
        default="french",
        help=(
            "Vocab to be used for training. Use a built-in name (e.g. 'french', 'english') "
            "or 'CUSTOM:<characters>' to define your own (e.g. 'CUSTOM:abc0123')."
        ),
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
        default="cosine",
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


def train_from_config(
    train_path: str | Path,
    val_path: str | Path,
    arch: str = "crnn_vgg16_bn",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    optimizer: str = "adam",
    scheduler: str = "cosine",
    input_size: int = 32,
    vocab: str = "french",
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
    """Run training with simplified configuration, suitable for UI calls.

    This is the primary entry point for programmatic training calls (e.g., from the UI).
    It converts simple parameters into an args object and calls main().

    Args:
        train_path: Path to training data folder
        val_path: Path to validation data folder
        arch: Model architecture name
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        weight_decay: Weight decay
        optimizer: Optimizer type ("adam" or "adamw")
        scheduler: LR scheduler type ("cosine", "onecycle", or "poly")
        input_size: Input image height
        vocab: Vocabulary name or "CUSTOM:<chars>"
        workers: Number of data loading workers
        amp: Whether to use automatic mixed precision
        early_stop: Whether to enable early stopping
        early_stop_epochs: Patience for early stopping
        early_stop_delta: Minimum delta for early stopping
        output_dir: Directory to save models
        device: GPU device index (-1 for CPU)
        pretrained: Whether to initialize from pretrained weights
        name: Experiment/model name used for output checkpoint naming
    """
    import argparse

    # Create a namespace object that mimics argparse args
    args = argparse.Namespace(
        arch=arch,
        train_path=str(train_path),
        val_path=str(val_path),
        train_datasets=None,
        val_datasets=None,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        optim=optimizer,
        sched=scheduler,
        input_size=input_size,
        vocab=vocab,
        workers=workers if isinstance(workers, int) else None,
        amp=amp,
        early_stop=early_stop,
        early_stop_epochs=early_stop_epochs,
        early_stop_delta=early_stop_delta,
        output_dir=output_dir,
        device=device,
        backend="nccl",
        train_samples=1000,
        val_samples=20,
        font="FreeMono.ttf,FreeSans.ttf,FreeSerif.ttf",
        min_chars=1,
        max_chars=12,
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
