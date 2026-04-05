"""
TotalSegmentator inference from duckn volumes.

Reads input from duckn zarr or ZMP formats, resamples to model spacing,
runs InferenceEngine, returns logits.

For NIfTI input, use mlx_predict.py (the TotalSegmentator CLI path).
No file output — returns numpy arrays for downstream processing.
"""

from __future__ import annotations

from nnunet_inference_mlx import InferenceEngine, ModelBundle

from duckn.io import read
from duckn.resample import resample
from duckn.volume import Volume

from totalsegmentator.mlx_predict import find_model_folder


def predict_volume(
    source: str,
    task_id: int = 297,
    trainer: str = "nnUNetTrainer_4000epochs_NoMirroring",
    spacing: float = 3.0,
    step_size: float = 0.5,
    verbose: bool = False,
) -> tuple:
    """Run TotalSegmentator inference on a duckn volume.

    Parameters
    ----------
    source : str
        Path to input volume (.zmp, .zarr, .zarr.zip)
    task_id : int
        nnU-Net dataset/task ID (e.g. 297 for fast mode)
    trainer : str
        nnU-Net trainer name
    spacing : float
        Target isotropic spacing in mm for the model
    step_size : float
        Sliding window overlap (0.5 = 50%)
    verbose : bool
        Print progress

    Returns
    -------
    (logits, vol_resampled) : tuple
        logits: np.ndarray (K, Z, Y, X) float32
        vol_resampled: duckn Volume at model spacing (for spatial reference)
    """
    # Read from any supported format
    vol = read(source)

    if verbose:
        geom = vol.geometry
        print(f"Input: {vol.shape}, spacing={geom.voxel_size}, "
              f"size_mm={geom.volume_size}")

    # Resample to model spacing
    vol_rsp = resample(vol, spacing=spacing)

    if verbose:
        geom_rsp = vol_rsp.geometry
        print(f"Resampled: {vol_rsp.shape}, spacing={geom_rsp.voxel_size}")

    # Load model and run inference
    model_folder = find_model_folder(task_id, trainer=trainer)
    bundle = ModelBundle.from_folder(model_folder)
    engine = InferenceEngine(bundle, step_size=step_size, verbose=verbose)

    # vol_rsp.data is (Z, Y, X) C-contiguous — direct to engine
    logits = engine.predict(vol_rsp.data)

    return logits, vol_rsp
