"""
MLX inference backend for TotalSegmentator.

Drop-in replacement for nnUNetv2_predict when device="mlx".
Reads NIfTI from dir_in, writes segmentation to dir_out.

Thin wrapper around nnunet-inference-mlx's InferenceEngine.
"""

from __future__ import annotations

import gc
import os
import time
from glob import glob
from pathlib import Path

import mlx.core as mx
import nibabel as nib
import numpy as np

from nnunet_inference_mlx import InferenceEngine, ModelBundle


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
    batch_size: int | None = None,
    **kwargs,
):
    """Drop-in replacement for nnUNetv2_predict using MLX.

    Reads NIfTI files from dir_in, runs inference, saves results to dir_out.
    Uses InferenceEngine internally.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    model_folder = find_model_folder(task_id, trainer, plans, model)
    fold = (folds or [0])[0]

    bundle = ModelBundle.from_folder(model_folder, fold=fold)
    engine = InferenceEngine(
        bundle,
        configuration=model,
        step_size=step_size,
        compile=use_compile,
        batch_size=batch_size,
        verbose=verbose,
        progress=not quiet,
    )

    if not quiet:
        print(f"MLX inference: task {task_id}, {engine.num_classes} classes, "
              f"patch {engine.patch_size}, batch={engine._batch_size}")

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

        # Relabel nibabel (X, Y, Z) as nnU-Net (Z, Y, X).
        # View only — no data movement.
        vol_zyx = data.transpose(2, 1, 0)

        st = time.perf_counter()
        logits = engine.predict(vol_zyx)
        dt = time.perf_counter() - st

        # logits: (K, Z, Y, X) → segmentation → back to nibabel order
        seg = np.argmax(logits, axis=0).astype(np.uint8)
        seg = seg.transpose(2, 1, 0)

        if not quiet:
            print(f"  Predicted in {dt:.1f}s ({np.unique(seg).size} labels)")

        seg_img = nib.Nifti1Image(seg, img.affine, img.header)
        nib.save(seg_img, str(out_path))

    # Free MLX compiled graph and cache before returning.
    # TotalSegmentator calls this function 5 times in full mode —
    # without cleanup, the Metal cache accumulates and OOMs.
    del engine, bundle, logits
    gc.collect()
    mx.clear_cache()
