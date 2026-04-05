"""
MLX inference backend for TotalSegmentator.

Drop-in replacement for nnUNetv2_predict when device="mlx".
Reads NIfTI from dir_in, writes segmentation to dir_out.

Uses nnunet-inference-mlx for model architectures and sliding window.
"""

from __future__ import annotations

import json
import os
import time
from glob import glob
from pathlib import Path

import mlx.core as mx
import nibabel as nib
import numpy as np

from nnunet_inference_mlx.inference import (
    predict_sliding_window,
    predict_sliding_window_segmentation,
    predict_sliding_window_streaming,
    choose_batch_size,
)
from nnunet_inference_mlx.plans import build_network_from_plans
from nnunet_inference_mlx.preprocessing import preprocess_volume
from nnunet_inference_mlx.weights import load_model_weights, fuzzy_load_weights


def find_model_folder(
    task_id: int,
    trainer: str = "nnUNetTrainer",
    plans: str = "nnUNetPlans",
    model: str = "3d_fullres",
    weights_dir: str | Path | None = None,
) -> Path:
    """Locate the model folder for a given task."""
    if weights_dir is None:
        if "TOTALSEG_WEIGHTS_PATH" in os.environ:
            weights_dir = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
        else:
            home = Path("/tmp") if str(Path.home()) == "/" else Path.home()
            weights_dir = home / ".totalsegmentator" / "nnunet" / "results"

    weights_dir = Path(weights_dir)
    matches = sorted(weights_dir.glob(f"Dataset{task_id}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No model found for task {task_id} in {weights_dir}. "
            f"Run TotalSegmentator once to download weights."
        )
    dataset_dir = matches[0]
    model_folder = dataset_dir / f"{trainer}__{plans}__{model}"
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    return model_folder


def nnUNetv2_predict_mlx(
    dir_in: str | Path,
    dir_out: str | Path,
    task_id: int,
    model: str = "3d_fullres",
    folds: list[int] | None = None,
    trainer: str = "nnUNetTrainer",
    tta: bool = False,
    plans: str = "nnUNetPlans",
    step_size: float = 0.5,
    quiet: bool = False,
    verbose: bool = False,
    use_compile: bool = True,
    use_logits: bool | None = None,
    batch_size: int | None = None,
    **kwargs,
):
    """Drop-in replacement for nnUNetv2_predict using MLX.

    Reads NIfTI files from dir_in, runs inference, saves results to dir_out.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    model_folder = find_model_folder(task_id, trainer, plans, model)

    plans_dict = json.loads((model_folder / "plans.json").read_text())
    dataset_json = json.loads((model_folder / "dataset.json").read_text())

    num_classes = len(dataset_json["labels"])
    channel_names = dataset_json.get("channel_names", dataset_json.get("modality", {}))
    num_input_channels = len(channel_names)

    configuration = model
    config = plans_dict["configurations"][configuration]
    patch_size = tuple(config["patch_size"])

    if batch_size is None:
        batch_size = choose_batch_size(patch_size, num_classes=num_classes, dtype_bytes=4)
        batch_size = max(1, batch_size)

    if use_logits is None:
        use_logits = True

    if not quiet:
        print(f"MLX inference: {num_classes} classes, patch {patch_size}, "
              f"batch={batch_size}, {'logits' if use_logits else 'segmentation'} mode")

    if folds is None:
        folds = [0]
    checkpoint_name = "checkpoint_final.pth"

    network = build_network_from_plans(
        plans_dict, configuration, num_input_channels, num_classes,
        deep_supervision=False,
    )

    all_fold_weights = [
        load_model_weights(model_folder, fold=f, checkpoint_name=checkpoint_name)
        for f in folds
    ]

    def _load_fold(weights):
        try:
            network.load_weights(list(weights.items()))
        except Exception:
            fuzzy_load_weights(network, weights, verbose=verbose)

    _load_fold(all_fold_weights[0])

    if use_compile:
        compiled_net = mx.compile(network)
    else:
        compiled_net = network

    # Warmup
    dummy = mx.random.normal((1, *patch_size, num_input_channels))
    mx.eval(compiled_net(dummy))
    del dummy

    # Process input files
    nifti_files = sorted(glob(str(dir_in / "*_0000.nii.gz")))
    if not nifti_files:
        nifti_files = sorted(glob(str(dir_in / "*.nii.gz")))

    for fpath in nifti_files:
        fname = Path(fpath).name
        out_name = fname.replace("_0000.nii.gz", ".nii.gz")
        out_path = dir_out / out_name

        if not quiet:
            print(f"  Processing {fname}")

        img = nib.load(fpath)
        data = np.asarray(img.dataobj, dtype=np.float32)

        preprocessed = preprocess_volume(data, plans_dict, configuration)
        preprocessed = preprocessed.transpose(0, 3, 2, 1).copy()

        st = time.perf_counter()

        def _run_one_fold(net):
            if use_logits:
                return predict_sliding_window_streaming(
                    network=net,
                    input_image=preprocessed,
                    patch_size=patch_size,
                    num_classes=num_classes,
                    tile_step_size=step_size,
                    use_gaussian=True,
                    use_mirroring=tta,
                    batch_size=batch_size,
                    use_fp16=False,
                    verbose=verbose,
                )
            else:
                top_labels, _ = predict_sliding_window_segmentation(
                    network=net,
                    input_image=preprocessed,
                    patch_size=patch_size,
                    num_classes=num_classes,
                    tile_step_size=step_size,
                    use_gaussian=True,
                    use_mirroring=tta,
                    batch_size=batch_size,
                    use_fp16=False,
                    verbose=verbose,
                )
                return top_labels

        if use_logits:
            logits_sum = _run_one_fold(compiled_net)
            for fold_weights in all_fold_weights[1:]:
                _load_fold(fold_weights)
                if use_compile:
                    compiled_net = mx.compile(network)
                logits_sum = logits_sum + _run_one_fold(compiled_net)
            if len(all_fold_weights) > 1:
                logits_sum /= len(all_fold_weights)
            seg = np.argmax(logits_sum.transpose(0, 3, 2, 1), axis=0)
        else:
            result = _run_one_fold(compiled_net)
            seg = result.transpose(2, 1, 0, 3)[..., 0]

        dt = time.perf_counter() - st

        if not quiet:
            print(f"  Predicted in {dt:.1f}s ({np.unique(seg).size} labels)")

        seg_img = nib.Nifti1Image(seg.astype(np.uint8), img.affine, img.header)
        nib.save(seg_img, str(out_path))

    if not quiet:
        print("Done.")
