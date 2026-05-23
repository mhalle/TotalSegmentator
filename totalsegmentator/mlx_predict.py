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


# ---------------------------------------------------------------------------
# Engine cache — skips compile/warmup between calls in batch / full-mode runs
# ---------------------------------------------------------------------------
#
# TotalSegmentator's full-body workflow calls this module's predict function
# 5 times per case (one per sub-model). Each fresh InferenceEngine pays ~2-3 s
# of mx.compile + warmup overhead before its first sliding-window forward.
# Across 5 models that's 10-15 s of pure setup time per case, comparable to
# the actual inference cost on fast hardware (M1 Max etc.).
#
# Caching the InferenceEngine across calls eliminates that overhead from the
# second invocation onward. The cache key is the tuple of arguments that
# determine engine identity (task / model / fold list / step / batch / compile).
# Two callers asking for the same configuration share the same engine.
#
# Memory: each cached engine holds ~600 MB of MLX state. Five cached engines
# is ~3 GB. On a 64 GB Mac that's trivial; on a 16 GB Mac it's a real fraction
# of working memory and can OOM the rest of TS's pipeline. So caching is
# auto-disabled on Macs with < 32 GB unified memory — same threshold the
# nnunet-inference-mlx Predictor uses for its cache_limit_fraction auto-tier.
#
# Override the auto-detect via the TOTALSEG_MLX_CACHE_ENGINES env var
# (set to "1" or "0") if the heuristic doesn't match your workload.

_ENGINE_CACHE: dict[tuple, InferenceEngine] = {}


def _cache_enabled() -> bool:
    """Should this process cache InferenceEngines across calls?

    Auto-detects from unified memory; override via TOTALSEG_MLX_CACHE_ENGINES.
    """
    env = os.environ.get("TOTALSEG_MLX_CACHE_ENGINES")
    if env is not None:
        return env.strip() not in ("", "0", "false", "False", "no", "No")
    try:
        import mlx.core as mx
        ram_gb = mx.device_info().get("memory_size", 0) / 1e9
        return ram_gb >= 32
    except Exception:
        return False


def clear_engine_cache() -> None:
    """Release all cached InferenceEngines and free their Metal memory.

    Call between unrelated TS workflows in a long-running process when you
    want to reclaim engine memory without exiting Python.
    """
    for engine in _ENGINE_CACHE.values():
        try:
            engine.close()
        except Exception:
            pass
    _ENGINE_CACHE.clear()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


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

    On Macs with >= 32 GB unified memory, the InferenceEngine is cached
    across calls (keyed on engine identity), so subsequent calls — and
    subsequent invocations from TS's full-mode pipeline — reuse the
    compiled network and skip ~2-3 s of warmup per model. On smaller Macs
    the cache is auto-disabled to avoid holding ~3 GB of inference state
    across multiple sub-models. Override with the TOTALSEG_MLX_CACHE_ENGINES
    env var (set to "1" or "0"). Call :func:`clear_engine_cache` to
    release cached engines manually.
    """
    dir_in = Path(dir_in)
    dir_out = Path(dir_out)
    dir_out.mkdir(parents=True, exist_ok=True)

    model_folder = find_model_folder(task_id, trainer, plans, model)
    folds_norm = tuple(folds) if folds else (0,)

    cache_enabled = _cache_enabled()
    # Engine identity: anything that would change the compiled graph or
    # weights. Step size, batch, compile, and tta all affect engine state.
    cache_key = (
        str(model_folder), model, folds_norm,
        step_size, use_compile, batch_size, bool(tta),
    )
    engine = _ENGINE_CACHE.get(cache_key) if cache_enabled else None

    if engine is None:
        bundle = ModelBundle.from_folder(model_folder, folds=folds or 0)
        engine = InferenceEngine(
            bundle,
            configuration=model,
            step_size=step_size,
            compile=use_compile,
            batch_size=batch_size,
            use_mirroring=tta,
            verbose=verbose,
            progress=not quiet,
        )
        if cache_enabled:
            _ENGINE_CACHE[cache_key] = engine
        owns_engine = not cache_enabled
    else:
        bundle = None  # bundle held inside the cached engine's predictor
        owns_engine = False
        if not quiet:
            print(f"MLX inference: task {task_id} (engine cached, reused)")

    try:
        if bundle is not None and not quiet:
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
    finally:
        if owns_engine:
            # Caching disabled — release engine + Metal cache on this call's
            # exit, matching the pre-cache behavior.
            engine.close()
        del bundle
        gc.collect()
