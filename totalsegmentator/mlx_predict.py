"""
MLX inference backend for TotalSegmentator.

Drop-in replacement for nnUNetv2_predict when device="mlx".
Reads NIfTI from dir_in, writes segmentation to dir_out.

Thin wrapper around nnunet-inference-mlx's InferenceEngine. Uses
SimpleITK for image I/O so this backend matches TotalSegmentator's
existing SimpleITK-everywhere convention — same library that
nnUNetPredictor uses internally for image reads. Side benefits over
the previous nibabel implementation:

  - Native (Z, Y, X) array order from ``GetArrayFromImage`` — no manual
    transpose at every read/write boundary.
  - Reads NIfTI / NRRD / MHA / MetaImage / MINC and DICOM series with
    the same call; only the input glob would need changing to extend.
  - Geometry round-trip via ``CopyInformation`` handles origin /
    spacing / direction in one line, including qform/sform edge cases.
"""

from __future__ import annotations

import gc
import os
import time
from glob import glob
from pathlib import Path

import numpy as np
import SimpleITK as sitk

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
    """Drop-in replacement for ``nnUNetv2_predict`` on the MLX backend.

    Reads input volumes from ``dir_in``, runs inference, writes
    segmentations to ``dir_out``. Parameters mirror the upstream
    predictor:

    * ``folds`` — list of fold IDs to use. Single fold → logits + argmax.
      Multiple folds → automatic softmax-averaged ensemble across folds
      (one network instance, weights swapped in place between folds).
      ``None`` defaults to fold 0.
    * ``tta`` — test-time augmentation via flip-averaging along the
      model's allowed mirroring axes (auto-read from the checkpoint).
      A no-op for ``NoMirroring`` trainers (TotalSegmentator's released
      models), so the default ``False`` is safe and ``tta=True`` is
      ignored harmlessly when axes are empty.
    * ``model`` — plans configuration name (passed through; the engine
      can also auto-detect from the checkpoint's ``init_args``).

    All inference state is released on function return via the engine's
    context manager — important because TotalSegmentator's full-body
    pipeline calls this five times in one process and the Metal
    allocator would otherwise accumulate buffers between calls.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    model_folder = find_model_folder(task_id, trainer, plans, model)

    # Honor the full fold list. Single fold → length-1 bundle (logits).
    # Multi-fold → ensemble (softmax-averaged for standard, sigmoid-
    # averaged for region-based; argmax/threshold-paint downstream picks
    # the right scheme automatically).
    bundle = ModelBundle.from_folder(model_folder, folds=folds or 0)

    with InferenceEngine(
        bundle,
        configuration=model,
        step_size=step_size,
        compile=use_compile,
        batch_size=batch_size,
        use_mirroring=tta,
        verbose=verbose,
        progress=not quiet,
    ) as engine:
        if not quiet:
            n_folds = len(bundle.fold_weights)
            suffix = f", folds={n_folds}" if n_folds > 1 else ""
            print(
                f"MLX inference: task {task_id}, "
                f"{engine.num_classes} classes, "
                f"patch {engine.patch_size}, "
                f"batch={engine.batch_size}{suffix}"
            )

        # nnU-Net input naming convention: <id>_0000.nii.gz for the first
        # (and for TS, only) channel. Fall back to *.nii.gz if no _0000
        # prefix is found, matching the previous wrapper's behavior.
        # SITK can also read .nrrd / .mha / etc. — extend the glob if a
        # caller ever needs that.
        nifti_files = sorted(glob(str(dir_in / "*_0000.nii.gz")))
        if not nifti_files:
            nifti_files = sorted(glob(str(dir_in / "*.nii.gz")))

        for fpath in nifti_files:
            fname = Path(fpath).name
            out_path = dir_out / fname.replace("_0000.nii.gz", ".nii.gz")
            if not quiet:
                print(f"  Processing {fname}")

            # SITK returns (Z, Y, X) natively in slowest-axis-first order;
            # no transpose needed. Cast to float32 for the engine — the
            # storage dtype for CT is int16/uint16, but inference works
            # in float32.
            img = sitk.ReadImage(fpath)
            vol_zyx = sitk.GetArrayFromImage(img).astype(np.float32, copy=False)

            st = time.perf_counter()
            # predict_segmentation handles standard + region-based label
            # schemes uniformly, and picks the smallest sufficient output
            # dtype from the dataset (uint8 for TS's 25-117 class models).
            seg_zyx = engine.predict_segmentation(vol_zyx)
            dt = time.perf_counter() - st

            if not quiet:
                print(
                    f"  Predicted in {dt:.1f}s "
                    f"({np.unique(seg_zyx).size} labels)"
                )

            # Round-trip geometry from input to output. CopyInformation
            # carries spacing + origin + direction in one call, handling
            # any qform/sform NIfTI subtleties that nibabel would have
            # forced us to pick a side on.
            seg_img = sitk.GetImageFromArray(seg_zyx)
            seg_img.CopyInformation(img)
            sitk.WriteImage(seg_img, str(out_path))

    # engine.close() and its Metal-cache clear ran on context-manager exit.
    # The explicit GC sweeps any Python-side references that the loop
    # held; harmless on the single-call path, mildly helpful when TS
    # invokes this five times back-to-back for the full-body workflow.
    del bundle
    gc.collect()
