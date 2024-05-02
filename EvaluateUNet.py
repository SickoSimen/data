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
    SpatialPadd,
    ToTensord,
)
from monai.engines import SupervisedEvaluator
from monai.handlers import (
    MeanDice,
    StatsHandler,
    TensorBoardImageHandler,
    TensorBoardStatsHandler,
    HausdorffDistance,
    from_engine,
    
)
from monai.inferers import SlidingWindowInferer
from ignite.metrics import Accuracy, Recall
from Dataset import get_train_and_val_ds
from ignite.handlers import Timer
from ignite.engine import Events



#Insert the checkpoint path here
checkpointPath = "/work/user/project/runs/net_key_metric=0.7400.pt"




if __name__ == "__main__":
    print("started running main")
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    get_logger("train_log")

    
    val_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            SpatialPadd(keys=["image"], spatial_size=[512, 512, 224], method="end", mode="constant", constant_values=-3435.0),
            SpatialPadd(keys=["label"], spatial_size=[512, 512, 224], method="end", mode="constant", constant_values=0),
            ScaleIntensityRanged(keys=["image"], a_min=-147.79612407684326, a_max=631.4952392578125, b_min=0.0, b_max=1.0),
            ToTensord(keys=["image", "label"]),
        ])



    # Create a validation dataset
    train_ds, val_ds = get_train_and_val_ds(val_transforms=val_transforms, validation_fraction=1)


    # Create a validation data loader
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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

    
    # Load the model
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint)

    
    # Postprocessing transform
    val_post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    # Handlers to log the validation metrics
    val_handlers = [
        StatsHandler(name="train_log", output_transform=lambda x: None),
        TensorBoardStatsHandler(log_dir="./eval/", output_transform=lambda x: None),
        TensorBoardImageHandler(
            log_dir="./eval/",
            batch_transform=from_engine(["image", "label"]),
            output_transform=from_engine(["pred"]),
        ),
    ]

    # Evaluator
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
            },
        val_handlers=val_handlers,
        amp=True,
    )

    # Timer to measure the time taken for each iteration
    timer = Timer()
    timer.attach(evaluator, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED, pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    evaluator.run()

    metrics = evaluator.state.metrics
    print(f"Model validation metrics: {metrics}")

    total_time = timer.value()
    num_iterations = evaluator.state.iteration
    average_time = total_time / num_iterations

    
    print(f"Average time: {average_time}")
    
    

    
    



    