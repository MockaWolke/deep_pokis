from models import ModelTemplate, LightningWrapper, TimmModel
from torch.utils.data import DataLoader
from data_loading import PokemonDataset
import os
from lightning import Trainer
import timm
from lightning.pytorch.loggers import WandbLogger
from torch import nn
import torch
import argparse
from datetime import datetime
from train_utils import gen_transforms, get_test_df, plot_test_preds
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

parser = argparse.ArgumentParser(
    description="Train a Pokemon classifier using Timm models."
)
parser.add_argument(
    "--cores", type=int, default=os.cpu_count(), help="Number of CPU cores to use"
)
parser.add_argument("--imgsz", type=int, default=224, help="Image size")
parser.add_argument("--crop_mode", type=str, default="crop", help="Cropping mode")
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument(
    "--backbone", type=str, default="mobilenetv3_small_050", help="Model backbone"
)
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
parser.add_argument(
    "--name",
    type=str,
    default=None,
)
parser.add_argument("--n_pred_imgs", type=int, default=10)
parser.add_argument("--stop_patience", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--cos_anneal", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--warmup_epoch", type=int, default=0)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument(
    "--transform_strength",
    type=str,
    choices=["mild", "medium", "strong"],
    default="mild",
)
parser.add_argument("--train_synth_frac", type=float, default=1.0)
parser.add_argument("--val_synth_frac", type=float, default=1.0)




args = parser.parse_args()


if args.name is None:
    args.name = f"{args.backbone}_{args.crop_mode}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

if args.test:
    args.name = args.name + "_test"

transform_pipeline = gen_transforms(args.transform_strength)


train_dataset = PokemonDataset(
    "train",
    args.crop_mode,
    imgsz=args.imgsz,
    transforms=transform_pipeline,
    synth_frac=args.train_synth_frac,
)
val_dataset = PokemonDataset(
    "val", args.crop_mode, imgsz=args.imgsz, synth_frac=args.val_synth_frac
)
train_loader = DataLoader(
    train_dataset, batch_size=args.bs, num_workers=args.cores, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=args.bs, num_workers=args.cores)



model = TimmModel(args.crop_mode == "both", 18, args.backbone, dropout=args.dropout)


wandb_logger = WandbLogger(
    args.name,
    dir="logs",
    log_model=True,
    config=vars(args),
    save_code=True,
)

callbacks = [LearningRateMonitor()]

if args.stop_patience:
    callbacks.append(
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=args.stop_patience,
            verbose=True,
        )
    )

checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",
    dirpath="checkpoints/",
    filename="best-checkpoint",
    save_top_k=1,
    mode="max",
)
callbacks.append(checkpoint_callback)


warmup_steps = args.warmup_epoch * len(train_loader)

wrapper = LightningWrapper(
    model,
    metric_average="micro",
    learning_rate=args.lr,
    cos_anneal=args.cos_anneal,
    warmup_steps=warmup_steps,
    min_lr=args.min_lr,
)
trainer = Trainer(
    devices="auto", max_epochs=args.epochs, logger=wandb_logger, callbacks=callbacks, fast_dev_run=args.test
)
trainer.fit(wrapper, train_loader, val_loader)

if checkpoint_callback.best_model_path:
    wrapper = LightningWrapper.load_from_checkpoint(checkpoint_callback.best_model_path, model = model)


try:

    eval_val = PokemonDataset("val", args.crop_mode, imgsz=args.imgsz, synth_frac=0)
    eval_loader = DataLoader(eval_val, batch_size=args.bs, num_workers=args.cores)

    trainer.test(wrapper, eval_loader)

    if hasattr(wrapper, "test_results"):
        wandb_logger.log_table("clean_val_by_class", dataframe=wrapper.test_results.reset_index(names="main_type"))

except Exception as e:
    print(f"Error while computing metrics by class: {e}")
    

try:

    pred_df = get_test_df(trainer, wrapper, batchsize=args.bs, num_workers=args.cores, crop_mode=args.crop_mode)

    stripped = pred_df[["Id", "main_type"]]

    wandb_logger.log_table("submission", dataframe=stripped)
except Exception as e:
    print(f"Test pred prediction: {e}")




try:

    imgs = [plot_test_preds(pred_df, wrapper.id2class) for i in range(args.n_pred_imgs)]

    wandb_logger.log_image("test_pred_imgs", imgs)

except Exception as e:
    print(f"There was an image loggin error: {e}")
