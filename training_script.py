from models import LightningWrapper, TimmModel, ArcFaceLightning
import os
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import torch
import argparse
from datetime import datetime
from train_utils import (
    gen_transforms,
    get_test_df,
    plot_test_preds,
    get_embeddings,
    plot_umap,
    get_top_10_best,
    get_val_transform,
    extract_mean_and_std,
)
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
import numpy as np
from datamodule import PokemonDataModule

parser = argparse.ArgumentParser(
    description="Train a Pokemon classifier using Timm models."
)


# model args
parser.add_argument(
    "--backbone", type=str, default="mobilenetv3_small_050", help="Model backbone"
)
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")


# logging args
parser.add_argument("--job_type", type=str, default=None)
parser.add_argument("--n_pred_imgs", type=int, default=10)
parser.add_argument("--name", type=str, default=None, help="name for logs")
parser.add_argument("--test_time_aug_its", type=int, default=0)

# training args
    

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--stop_patience", type=int, default=None)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--amsgrad", action="store_true")
parser.add_argument("--gradient_clip_val", type=float, default=None)


parser.add_argument("--cos_anneal", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--warmup_epoch", type=int, default=0)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--precision", type=int, default=32, choices=[32, 16])

# arcface
parser.add_argument("--arcface", action="store_true", help="Use ArcFace loss")
parser.add_argument("--arc_margin", type=float, default=28.6)
parser.add_argument("--arc_scale", type=float, default=4)
parser.add_argument(
    "--class_pow",
    type=float,
    default=0,
    help="power of inverse class counts to serve as weights",
)


# data loading args
parser.add_argument(
    "--transform_strength",
    type=str,
    choices=["mild", "medium", "strong", "medium_strong_crop", "extra_strong"],
    default="mild",
)

parser.add_argument(
    "--cores", type=int, default=os.cpu_count(), help="Number of CPU cores to use"
)
parser.add_argument("--imgsz", type=int, default=224, help="Image size")
parser.add_argument("--crop_mode", type=str, default="crop", help="Cropping mode")
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument("--val_og_percentage", type=float, default=1.0)
parser.add_argument("--fill_val_to", type=float, default=0.1)
parser.add_argument("--pin_memory", action="store_true")


args = parser.parse_args()


if args.name is None:
    args.name = f"{args.backbone}_{args.crop_mode}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"

if args.test:
    args.name = args.name + "_test"


model = TimmModel(
    args.crop_mode == "both",
    18,
    args.backbone,
    dropout=args.dropout,
    include_head=not args.arcface,
    imgsz=args.imgsz,
)


norm_mean, norm_std = extract_mean_and_std(model.backbone)

transform_pipeline = gen_transforms(
    args.transform_strength, mean=norm_mean, std=norm_std
)
val_transform = get_val_transform(mean=norm_mean, std=norm_std)


datamodule = PokemonDataModule(
    batch_size=args.bs,
    num_workers=args.cores,
    imgsz=args.imgsz,
    train_transforms=transform_pipeline,
    val_and_pred_transforms=val_transform,
    fill_val_to=args.fill_val_to,
    pin_memory=args.pin_memory,
    val_og_percentage=args.val_og_percentage,
)

train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()


wandb_logger = WandbLogger(
    args.name,
    dir="logs",
    log_model=not args.test,
    config=vars(args),
    save_code=True,
    job_type=args.job_type,
    offline=args.test,
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


wrapper_kwargs = {
    "learning_rate": args.lr,
    "cos_anneal": args.cos_anneal,
    "warmup_steps": warmup_steps,
    "min_lr": args.min_lr,
    "weight_decay": args.weight_decay,
    "amsgrad": args.amsgrad,
    "class_pow": args.class_pow,
}

if args.arcface == False:

    Wrapper_class = LightningWrapper

else:
    Wrapper_class = ArcFaceLightning

    wrapper_kwargs.update(
        {
            "margin": args.arc_margin,
            "scale": args.arc_scale,
        }
    )


wrapper = Wrapper_class(model, **wrapper_kwargs)


trainer = Trainer(
    devices="auto",
    max_epochs=args.epochs,
    logger=wandb_logger,
    callbacks=callbacks,
    fast_dev_run=args.test,
    precision=args.precision,
    gradient_clip_val=args.gradient_clip_val,
)
trainer.fit(wrapper, train_loader, val_loader)

if checkpoint_callback.best_model_path:

    wrapper = Wrapper_class.load_from_checkpoint(
        checkpoint_callback.best_model_path, model=model
    )

    if torch.cuda.is_available():
        wrapper = wrapper.cuda()


test_dataset = datamodule.get_test_dataset()
test_loader = datamodule.test_dataloader()


# special evaluation for metric learning arcface model
if args.arcface:

    try:

        # train_embeddings, train_labels = get_embeddings(wrapper, train_loader)
        val_embeddings, val_labels = get_embeddings(wrapper, val_loader)
        test_embeddings, test_labels = get_embeddings(wrapper, test_loader)

        combined_embeddings = val_embeddings # np.concatenate((train_embeddings, val_embeddings))
        combined_label = val_labels # np.concatenate((train_labels, val_labels))

        try:
            val_umap = plot_umap(
                val_embeddings,
                val_labels,
                id2class=wrapper.id2class,
                title="2d Umap Val Embeddings",
            )
            wandb_logger.log_image("val_umap", [val_umap])
        except Exception as e:
            print("val umap saving failed", e)

        try:
            top_10_best = get_top_10_best(
                combined_embeddings=combined_embeddings,
                test_embeddings=test_embeddings,
                combined_label=combined_label,
                test_dataset=test_dataset,
            )

            wandb_logger.log_table("emb_top_10", dataframe=top_10_best)

        except Exception as e:
            print("val umap saving failed", e)

    except Exception as e:

        print(f"Arc Face Eval error", e)
try:

    trainer.test(wrapper, datamodule.val_dataloader(without_synth=True))

    if hasattr(wrapper, "test_results"):
        wandb_logger.log_table(
            "clean_val_by_class",
            dataframe=wrapper.test_results.reset_index(names="main_type"),
        )

    else:
        raise AttributeError("wrapper has no test_results attr")

except Exception as e:
    print(f"Error while computing metrics by class", e)


try:

    pred_df = get_test_df(
        trainer,
        wrapper,
        dataset=test_dataset,
        loader=test_loader,
    )

    stripped = pred_df[["Id", "main_type"]]

    wandb_logger.log_table("submission", dataframe=stripped)
except Exception as e:
    print(f"Test pred prediction", e)

# new block for 
try:

    if args.test_time_aug_its > 0:
        
 
        

        aug_test_loader = datamodule.test_dataloader(overwrite_transforms=transform_pipeline)
        
    


        pred_df = get_test_df(
            trainer,
            wrapper,
            dataset=test_dataset,
            loader=aug_test_loader,
            n_iterations=args.test_time_aug_its
        )

        stripped = pred_df[["Id", "main_type"]]

        wandb_logger.log_table(f"test_time_aug_{args.test_time_aug_its}_submission", dataframe=stripped)
except Exception as e:
    print(f"test_time_augmentation failed", e)





try:

    imgs = [plot_test_preds(pred_df, wrapper.id2class) for i in range(args.n_pred_imgs)]

    wandb_logger.log_image("test_pred_imgs", imgs)

except Exception as e:
    print(f"There was an image loggin error", e)
    print(e)
