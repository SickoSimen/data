import sys
import torch
import monai
import logging
from monai.apps import get_logger
from monai.data import DataLoader
from dynamic_network_architectures.architectures import unet
from monai.transforms import (
    Compose,  
    ScaleIntensityRanged,
    LoadImaged,
    Activationsd,
    AsDiscreted,
    RandFlipd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    ToTensord,
)
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    LrScheduleHandler,
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    HausdorffDistance,
    from_engine,
    MeanDice
)
from monai.inferers import SimpleInferer, SlidingWindowInferer
from ignite.metrics import Accuracy, Recall, Loss
from Dataset import get_train_and_val_ds
from Loss import DiceFocalLoss




if __name__ == "__main__":
    print("started running main")
    monai.config.print_config()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")


    # Training and validation transforms

    train_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            ScaleIntensityRanged(keys=["image"], a_min=-147.79612407684326, a_max=631.4952392578125, b_min=0.0, b_max=1.0, clip=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[128, 128, 128],
                pos=1,
                neg=1,
                num_samples=3,
            ),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=2),
            ToTensord(keys=["image", "label"]),
        ])
    
    val_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            SpatialPadd(keys=["image"], spatial_size=[512, 512, 224], method="end", mode="constant", constant_values=-3435.0),
            SpatialPadd(keys=["label"], spatial_size=[512, 512, 224], method="end", mode="constant", constant_values=0),
            ScaleIntensityRanged(keys=["image"], a_min=-147.79612407684326, a_max=631.4952392578125, b_min=0.0, b_max=1.0),
            ToTensord(keys=["image", "label"]),
        ])



    # Getting the training and validation datasets
    train_ds, val_ds = get_train_and_val_ds(train_transforms, val_transforms)
 
    # Create a training data loader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
  
    # Create a validation data loader
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")


    # Create UNet, DiceLoss and Adam optimizer
    model = unet.PlainConvUNet(
        input_channels=1,
        num_classes=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.modules.conv.Conv3d,
        kernel_sizes=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-05, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    ).to(device)
    loss_function = DiceFocalLoss(weight_dice=0.5, weight_ce=0.5)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    # Validation post transforms
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    # Validation handlers
    val_handlers = [
        StatsHandler(name="train_log", output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir="./runs/", output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir="./runs/",
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
        CheckpointSaver(save_dir="./runs/", save_dict={"net": model}, save_key_metric=True, key_metric_n_saved=5),
    ]

    # Validation evaluator
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        inferer=SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=4, overlap=0.5),
        postprocessing=val_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"])),
        },
        additional_metrics={
            "val_hd": HausdorffDistance(include_background=True, output_transform=from_engine(["pred", "label"])),
            "val_acc": Accuracy(output_transform=from_engine(["pred", "label"])),
            "val_Recall": Recall(output_transform=from_engine(["pred", "label"])),
            "val_loss": Loss(loss_function, output_transform=from_engine(["pred", "label"])),
            },
        val_handlers=val_handlers,
        amp=True,
    )
    
    # Training post transforms
    train_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )
    
    # Training handlers
    train_handlers = [
        LrScheduleHandler(lr_scheduler=lr_scheduler, print_lr=True),
        ValidationHandler(validator=evaluator, interval=2, epoch_level=True),
        StatsHandler(name="train_log", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
        TensorBoardStatsHandler(
            log_dir="./runs/", tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
    ]
    
    # Trainer object
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=300,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        inferer=SimpleInferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        postprocessing=train_post_transforms,
        additional_metrics={
            "tain_acc": Accuracy(output_transform=from_engine(["pred", "label"])),
            "train_Recall": Recall(output_transform=from_engine(["pred", "label"])),
            "train_mean_dice": MeanDice(include_background=True, output_transform=from_engine(["pred", "label"])),
            },
        amp=True,
    )
  
    # Run the training
    trainer.run()
