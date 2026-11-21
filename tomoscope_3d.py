"""TOMOSCOPE 3-D volume reconstruction from full-height camera projections.

The camera image's vertical coordinate is retained instead of being averaged
away.  Each vertical detector band supplies one parallel-beam sinogram.  Each
sinogram is reconstructed with the existing 2-D filtered-backprojection (FBP)
engine, and the resulting cross-sections are stacked along z.

This is a valid slice-wise 3-D parallel-beam reconstruction for a sample that
rotates about its vertical axis, provided the camera's vertical detector axis is
aligned with that rotation axis and rays within a given detector row stay in
that horizontal plane.  It is NOT cone-beam CT and does not recover arbitrary
out-of-plane ray paths.  The LED/camera school apparatus is only an approximate
parallel-beam system; results are educational relative attenuation estimates.

Volume array convention: volume[z_index, y_index, x_index].  The z axis is
stored bottom-to-top (ascending physical z), x increases left-to-right, and y
increases from bottom-to-top as in the 2-D reconstruction module.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps

from tomoscope_core import SUPPORTED_SUFFIXES, validate_image_path, image_size
from tomoscope_reconstruction import SinogramData, reconstruct_arrays


@dataclass(frozen=True)
class VolumeReconstructionResult:
    output_dir: Path
    volume_npy_path: Path
    volume_npz_path: Path
    sinogram_stack_path: Path
    pointcloud_path: Path
    orthogonal_path: Path
    mip_path: Path
    montage_path: Path
    report_path: Path
    metadata_path: Path
    shape: tuple[int, int, int]  # (z, y, x)
    projection_count: int
    detector_samples: int
    filter_name: str
    z_extent_cm: float
    xy_extent_cm: float


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _read_linear_gray_crop(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    """Read only the selected image region and return linear-light grayscale."""
    try:
        with Image.open(path) as original:
            im = ImageOps.exif_transpose(original)
            crop = im.crop(box).convert("RGB")
            rgb = np.asarray(crop, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not read image {path.name}: {exc}") from exc
    linear = _srgb_to_linear(rgb)
    return (0.2126 * linear[:, :, 0]
            + 0.7152 * linear[:, :, 1]
            + 0.0722 * linear[:, :, 2]).astype(np.float32)


def _validate_angles(angles_deg: Sequence[float], projection_count: int) -> np.ndarray:
    angles = np.asarray(angles_deg, dtype=np.float64)
    if angles.ndim != 1 or angles.size != projection_count:
        raise ValueError("The number of assigned angles must match the number of projection photographs.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("Scan angles must all be finite numbers.")
    if np.any(angles < 0.0) or np.any(angles >= 180.0):
        raise ValueError("For this parallel-beam model, scan angles must be from 0° up to (but not including) 180°.")
    if np.unique(np.round(angles, 8)).size != angles.size:
        raise ValueError("The scan contains duplicate angles. Each projection must have a unique angle.")
    order = np.argsort(angles)
    sorted_angles = angles[order]
    gaps = np.diff(np.concatenate([sorted_angles, [sorted_angles[0] + 180.0]]))
    if gaps.max() > 3.0 * np.median(gaps):
        raise ValueError("The scan has a large missing angular sector. Capture evenly spaced projections over 0°–180°.")
    return order.astype(int)


def _extract_vertical_sinograms(
    flat_path: str | Path,
    dark_path: str | Path | None,
    projection_paths: Sequence[str | Path],
    angles_deg: Sequence[float],
    x_start_pct: float,
    x_end_pct: float,
    z_top_pct: float,
    z_bottom_pct: float,
    detector_width_cm: float,
    object_height_cm: float,
    z_slices: int,
    max_detector_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Build attenuation sinograms with shape (z, angle, detector) from images."""
    flat = validate_image_path(flat_path)
    dark = validate_image_path(dark_path) if dark_path else None
    projections = [validate_image_path(p) for p in projection_paths]
    if len(projections) < 3:
        raise ValueError("At least three projection photographs are required; 18 or more are recommended.")
    if not (0.0 <= x_start_pct < x_end_pct <= 100.0):
        raise ValueError("Horizontal ROI percentages must satisfy 0 ≤ left < right ≤ 100.")
    if not (0.0 <= z_top_pct < z_bottom_pct <= 100.0):
        raise ValueError("Vertical ROI must satisfy 0 ≤ top < bottom ≤ 100.")
    if detector_width_cm <= 0 or object_height_cm <= 0:
        raise ValueError("The calibrated horizontal width and vertical height must both be positive.")
    if not (8 <= int(z_slices) <= 256):
        raise ValueError("Choose between 8 and 256 reconstructed height slices.")
    if max_detector_samples < 64:
        raise ValueError("At least 64 detector samples are required for useful reconstruction.")

    all_paths = [flat] + ([dark] if dark else []) + projections
    dimensions = [image_size(p) for p in all_paths]
    if any(size != dimensions[0] for size in dimensions[1:]):
        details = ", ".join(f"{p.name}:{s[0]}x{s[1]}" for p, s in zip(all_paths, dimensions))
        raise ValueError("All reference and projection photographs must have identical pixel dimensions. " + details)
    width_px, height_px = dimensions[0]

    x0 = int(round(width_px * x_start_pct / 100.0))
    x1 = int(round(width_px * x_end_pct / 100.0))
    y0 = int(round(height_px * z_top_pct / 100.0))
    y1 = int(round(height_px * z_bottom_pct / 100.0))
    if x1 - x0 < 64:
        raise ValueError("The selected horizontal detector region is too narrow. Include at least 64 image pixels.")
    if y1 - y0 < z_slices:
        raise ValueError("The selected vertical region has fewer pixels than the requested number of z slices. Widen the height ROI or reduce z slices.")

    order = _validate_angles(angles_deg, len(projections))
    angle_array = np.asarray(angles_deg, dtype=np.float64)[order]
    ordered_projections = [projections[i] for i in order]

    n_detector = min(int(max_detector_samples), x1 - x0)
    x_sample = np.linspace(0, x1 - x0 - 1, n_detector, dtype=np.float64)
    detector_cm = np.linspace(-detector_width_cm / 2.0, detector_width_cm / 2.0,
                              n_detector, dtype=np.float64)

    # Split source rows into non-overlapping bins, bottom-to-top, so the volume's
    # physical z coordinate increases with its first index.
    row_bins = np.array_split(np.arange(y1 - 1, y0 - 1, -1, dtype=int), int(z_slices))
    if any(len(rows) == 0 for rows in row_bins):
        raise ValueError("Some z slices contain no camera rows. Reduce the number of z slices.")

    roi_box = (x0, y0, x1, y1)
    flat_img = _read_linear_gray_crop(flat, roi_box)
    dark_img = _read_linear_gray_crop(dark, roi_box) if dark else np.zeros_like(flat_img)

    def profiles_by_z(image: np.ndarray) -> np.ndarray:
        rows_out = np.empty((z_slices, n_detector), dtype=np.float32)
        for zi, source_rows in enumerate(row_bins):
            row_profile = np.mean(image[source_rows - y0, :], axis=0, dtype=np.float64)
            rows_out[zi] = np.interp(x_sample, np.arange(row_profile.size), row_profile).astype(np.float32)
        return rows_out

    flat_profiles = profiles_by_z(flat_img)
    dark_profiles = profiles_by_z(dark_img)
    flat_signal = flat_profiles - dark_profiles
    if np.any(~np.isfinite(flat_signal)) or np.any(flat_signal <= 1e-6):
        bad = int(np.count_nonzero(flat_signal <= 1e-6))
        raise ValueError(f"Flat reference is too dark or invalid in {bad} z/detector samples. Check LED illumination and exposure.")

    sinogram_by_angle: list[np.ndarray] = []
    above_one = 0
    nonpositive = 0
    total = 0
    for path in ordered_projections:
        image = _read_linear_gray_crop(path, roi_box)
        projection_profiles = profiles_by_z(image)
        corrected = projection_profiles - dark_profiles
        transmission = corrected / flat_signal
        above_one += int(np.count_nonzero(transmission > 1.0))
        nonpositive += int(np.count_nonzero(transmission <= 0.0))
        total += int(transmission.size)
        attenuation = -np.log(np.clip(transmission, 1e-4, 1.0))
        sinogram_by_angle.append(attenuation.astype(np.float32))

    # (angle, z, detector) -> (z, angle, detector)
    sinogram_cube = np.stack(sinogram_by_angle, axis=0).transpose(1, 0, 2)
    z_edges = np.linspace(-object_height_cm / 2.0, object_height_cm / 2.0,
                          int(z_slices) + 1, dtype=np.float64)
    z_cm = 0.5 * (z_edges[:-1] + z_edges[1:])
    info = {
        "image_width_px": width_px,
        "image_height_px": height_px,
        "roi_pixels_xy": {"left": x0, "right": x1, "top": y0, "bottom": y1},
        "roi_percent_xy": {"left": x_start_pct, "right": x_end_pct,
                           "top": z_top_pct, "bottom": z_bottom_pct},
        "detector_width_cm": float(detector_width_cm),
        "object_height_cm": float(object_height_cm),
        "detector_samples": int(n_detector),
        "z_slices": int(z_slices),
        "z_spacing_cm": float(object_height_cm / z_slices),
        "projection_count": int(len(ordered_projections)),
        "angles_deg": angle_array.tolist(),
        "flat_reference": str(flat),
        "dark_reference": str(dark) if dark else None,
        "projection_files_ordered": [str(p) for p in ordered_projections],
        "transmission_above_one_percent": 100.0 * above_one / max(1, total),
        "nonpositive_transmission_percent": 100.0 * nonpositive / max(1, total),
        "dark_reference_used": bool(dark),
    }
    return sinogram_cube, detector_cm, angle_array, z_cm, info


def reconstruct_volume_from_sinogram_stack(
    sinogram_cube: np.ndarray,
    detector_positions_cm: np.ndarray,
    angles_deg: np.ndarray,
    z_positions_cm: np.ndarray,
    resolution: int = 128,
    filter_name: str = "Hann",
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Reconstruct (z,y,x) volume from one 2-D sinogram per physical height."""
    cube = np.asarray(sinogram_cube, dtype=np.float64)
    detector = np.asarray(detector_positions_cm, dtype=np.float64)
    angles = np.asarray(angles_deg, dtype=np.float64)
    z = np.asarray(z_positions_cm, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("Sinogram stack must have shape (z_slices, projection_angles, detector_samples).")
    if cube.shape != (z.size, angles.size, detector.size):
        raise ValueError("Sinogram-stack dimensions do not match z, angle, and detector coordinate arrays.")
    if z.size < 8 or np.any(np.diff(z) <= 0):
        raise ValueError("At least eight strictly increasing z positions are required.")
    if not np.all(np.isfinite(cube)) or np.any(cube < -1e-8):
        raise ValueError("Sinogram stack contains invalid or negative attenuation values. Recreate the scan.")
    if resolution < 64 or resolution > 512:
        raise ValueError("XY reconstruction resolution must be between 64 and 512 pixels.")

    volume = None
    x_coords = y_coords = None
    slice_info: list[dict] = []
    for zi in range(z.size):
        data = SinogramData(
            detector_positions_cm=detector,
            angles_deg=angles,
            attenuation=cube[zi],
            source_path=Path(f"vertical_slice_{zi:03d}.csv"),
        )
        x, y, _bp, fbp, info = reconstruct_arrays(data, resolution=resolution, filter_name=filter_name)
        if volume is None:
            x_coords, y_coords = x, y
            volume = np.empty((z.size, resolution, resolution), dtype=np.float32)
        volume[zi] = fbp.astype(np.float32)
        slice_info.append(info)
        if progress_callback is not None:
            progress_callback(zi + 1, int(z.size))
    assert volume is not None and x_coords is not None and y_coords is not None
    return volume, x_coords, y_coords, z, slice_info


def _finite_limits(volume: np.ndarray) -> tuple[float, float]:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        raise ValueError("The reconstructed volume contains no finite voxels.")
    lo, hi = np.percentile(finite, [1.0, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo, hi = float(np.min(finite)), float(np.max(finite)) + 1e-9
    return float(lo), float(hi)


def _save_orthogonal_views(path: Path, volume: np.ndarray,
                           x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    midz, midy, midx = len(z) // 2, len(y) // 2, len(x) // 2
    lo, hi = _finite_limits(volume)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    items = [
        (volume[midz], [x[0], x[-1], y[0], y[-1]], "XY plane (middle height)", "x (cm)", "y (cm)"),
        (volume[:, midy, :], [x[0], x[-1], z[0], z[-1]], "XZ plane (middle y)", "x (cm)", "z (cm)"),
        (volume[:, :, midx], [y[0], y[-1], z[0], z[-1]], "YZ plane (middle x)", "y (cm)", "z (cm)"),
    ]
    for ax, (image, extent, title, xlabel, ylabel) in zip(axes, items):
        im = ax.imshow(image, origin="lower", extent=extent, cmap="magma",
                       interpolation="nearest", aspect="auto", vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, shrink=0.78, label="Relative reconstructed attenuation")
    fig.suptitle("TOMOSCOPE — orthogonal 3-D volume slices")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def _save_mips(path: Path, volume: np.ndarray, x: np.ndarray,
               y: np.ndarray, z: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = np.where(np.isfinite(volume), np.maximum(volume, 0.0), 0.0)
    projections = [
        (np.max(positive, axis=0), [x[0], x[-1], y[0], y[-1]], "Maximum intensity — XY", "x (cm)", "y (cm)"),
        (np.max(positive, axis=1), [x[0], x[-1], z[0], z[-1]], "Maximum intensity — XZ", "x (cm)", "z (cm)"),
        (np.max(positive, axis=2), [y[0], y[-1], z[0], z[-1]], "Maximum intensity — YZ", "y (cm)", "z (cm)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, (image, extent, title, xlabel, ylabel) in zip(axes, projections):
        im = ax.imshow(image, origin="lower", extent=extent, cmap="inferno", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=axes, shrink=0.78, label="Maximum relative attenuation")
    fig.suptitle("TOMOSCOPE — three orthogonal maximum-intensity projections")
    fig.savefig(path, dpi=175)
    plt.close(fig)


def _save_pointcloud(path: Path, volume: np.ndarray, x: np.ndarray,
                     y: np.ndarray, z: np.ndarray, max_points: int = 60000) -> dict:
    """Save a thresholded 3-D voxel point-cloud view (no extra packages needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = np.where(np.isfinite(volume), np.maximum(volume, 0.0), 0.0)
    vals = positive[positive > 0]
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    if vals.size < 10:
        ax.text2D(0.15, 0.5, "Insufficient positive voxels for a point-cloud view.\nInspect the orthogonal slices and acquisition report.", transform=ax.transAxes)
        threshold = None
        displayed = 0
    else:
        threshold = float(np.percentile(vals, 98.0))
        mask = positive >= threshold
        indices = np.argwhere(mask)
        if indices.shape[0] > max_points:
            stride = int(math.ceil(indices.shape[0] / max_points))
            indices = indices[::stride]
        vz, vy, vx = indices[:, 0], indices[:, 1], indices[:, 2]
        colors = positive[vz, vy, vx]
        scatter = ax.scatter(x[vx], y[vy], z[vz], c=colors, cmap="inferno",
                             s=1.5, alpha=0.65, linewidths=0)
        fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1, label="Relative attenuation")
        displayed = int(indices.shape[0])
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_title("TOMOSCOPE — reconstructed 3-D voxel point cloud\n(display threshold: 98th percentile of positive values)")
    ax.set_box_aspect((float(x[-1]-x[0]), float(y[-1]-y[0]), float(z[-1]-z[0])))
    fig.savefig(path, dpi=185)
    plt.close(fig)
    return {"display_threshold": threshold, "displayed_points": displayed,
            "threshold_percentile": 98.0, "max_points": int(max_points)}


def _save_slice_montage(path: Path, volume: np.ndarray, x: np.ndarray,
                        y: np.ndarray, z: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = min(12, len(z))
    indices = np.unique(np.linspace(0, len(z)-1, count).round().astype(int))
    cols = 4
    rows = int(math.ceil(len(indices) / cols))
    lo, hi = _finite_limits(volume)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.1*rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, indices):
        ax.imshow(volume[idx], origin="lower", extent=[x[0], x[-1], y[0], y[-1]],
                  cmap="magma", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"z = {z[idx]:.3f} cm")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
    for ax in axes[len(indices):]:
        ax.axis("off")
    fig.suptitle("TOMOSCOPE — reconstructed axial slices from bottom to top")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reconstruct_3d_from_images(
    flat_path: str | Path,
    dark_path: str | Path | None,
    projection_paths: Sequence[str | Path],
    angles_deg: Sequence[float],
    output_dir: str | Path,
    x_start_pct: float = 15.0,
    x_end_pct: float = 85.0,
    z_top_pct: float = 15.0,
    z_bottom_pct: float = 85.0,
    detector_width_cm: float = 10.0,
    object_height_cm: float = 3.0,
    z_slices: int = 48,
    resolution: int = 128,
    filter_name: str = "Hann",
    max_detector_samples: int = 512,
    progress_callback=None,
) -> VolumeReconstructionResult:
    """Create calibrated vertical sinograms from camera images and reconstruct a 3-D volume."""
    sinograms, detector_cm, angles, z_cm, acquisition_info = _extract_vertical_sinograms(
        flat_path, dark_path, projection_paths, angles_deg,
        x_start_pct, x_end_pct, z_top_pct, z_bottom_pct,
        detector_width_cm, object_height_cm, z_slices, max_detector_samples,
    )
    volume, x, y, z_cm, per_slice_info = reconstruct_volume_from_sinogram_stack(
        sinograms, detector_cm, angles, z_cm, resolution=resolution,
        filter_name=filter_name, progress_callback=progress_callback,
    )

    target = Path(output_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    volume_npy = target / "volume_filtered.npy"
    volume_npz = target / "volume_filtered.npz"
    sino_npz = target / "sinogram_stack_3d.npz"
    pointcloud_png = target / "volume_3d_pointcloud.png"
    orthogonal_png = target / "volume_orthogonal_views.png"
    mip_png = target / "volume_max_intensity_projections.png"
    montage_png = target / "volume_axial_slices_montage.png"
    report_path = target / "volume_reconstruction_report.txt"
    metadata_path = target / "volume_reconstruction_metadata.json"

    np.save(volume_npy, volume, allow_pickle=False)
    np.savez_compressed(volume_npz, volume=volume, x_cm=x, y_cm=y, z_cm=z_cm)
    np.savez_compressed(sino_npz, sinograms=sinograms, detector_positions_cm=detector_cm,
                        angles_deg=angles, z_positions_cm=z_cm)
    _save_orthogonal_views(orthogonal_png, volume, x, y, z_cm)
    _save_mips(mip_png, volume, x, y, z_cm)
    _save_slice_montage(montage_png, volume, x, y, z_cm)
    pointcloud_info = _save_pointcloud(pointcloud_png, volume, x, y, z_cm)

    finite = volume[np.isfinite(volume)]
    metadata = {
        "project": "TOMOSCOPE 3-D educational optical tomography volume",
        "created": datetime.now().isoformat(timespec="seconds"),
        "algorithm": "Slice-wise parallel-beam filtered backprojection (2-D FBP per height, stacked into a 3-D volume)",
        "volume_axis_order": "volume[z_index, y_index, x_index]",
        "shape_zyx": list(volume.shape),
        "x_min_cm": float(x[0]), "x_max_cm": float(x[-1]),
        "y_min_cm": float(y[0]), "y_max_cm": float(y[-1]),
        "z_min_cm": float(z_cm[0]), "z_max_cm": float(z_cm[-1]),
        "z_spacing_cm": float(np.median(np.diff(z_cm))) if len(z_cm) > 1 else None,
        "xy_resolution_pixels": int(resolution),
        "z_slices": int(len(z_cm)),
        "filter": filter_name,
        "sinogram_stack_shape_z_angle_detector": list(sinograms.shape),
        "attenuation_value_range_finite": [float(np.min(finite)), float(np.max(finite))],
        "positive_voxel_count": int(np.count_nonzero(finite > 0)),
        "point_cloud_display": pointcloud_info,
        "acquisition": acquisition_info,
        "geometry": {
            "assumption": "The phantom rotates about a vertical axis; camera vertical rows correspond to separate horizontal z planes; illumination is approximately parallel within each plane.",
            "radon_coordinate": "s = x*cos(theta) + y*sin(theta)",
            "z_axis": "The selected image rows are mapped linearly from the measured physical object height.",
        },
        "limitations": [
            "This is slice-wise 3-D parallel-beam reconstruction, not cone-beam CT or a general 3-D cone-beam algorithm.",
            "The camera vertical axis must be aligned with the phantom's rotation axis.",
            "The object must remain at the same height and centre during all projections; only rotation changes.",
            "All selected image rows must be illuminated and visible in the flat reference and every projection.",
            "The z scale is only as accurate as the entered physical height calibration.",
            "The LED and camera do not create perfectly parallel rays; perspective, refraction, scattering and JPEG processing cause artefacts.",
            "Voxel values are relative camera-derived attenuation estimates, not calibrated physical coefficients or medical CT numbers.",
        ],
        "outputs": {
            "volume_npy": volume_npy.name,
            "volume_npz": volume_npz.name,
            "sinogram_stack_npz": sino_npz.name,
            "pointcloud_png": pointcloud_png.name,
            "orthogonal_views_png": orthogonal_png.name,
            "mip_png": mip_png.name,
            "axial_montage_png": montage_png.name,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = [
        "TOMOSCOPE — 3-D VOLUME RECONSTRUCTION REPORT",
        "=" * 62,
        f"Created: {metadata['created']}",
        f"Input projections: {len(angles)}",
        f"Projection angle range: {angles[0]:g}° to {angles[-1]:g}°",
        f"Sinogram stack (z, angle, detector): {sinograms.shape}",
        f"Reconstructed volume (z, y, x): {volume.shape}",
        f"Physical x/y field of view: {detector_width_cm:g} cm across the detector width",
        f"Physical z extent: {object_height_cm:g} cm",
        f"Voxel spacing z: {object_height_cm/z_slices:.6f} cm",
        f"FBP filter: {filter_name}",
        f"XY resolution: {resolution} × {resolution}",
        f"Flat-reference transmission above 1 (clipped to zero attenuation): {acquisition_info['transmission_above_one_percent']:.2f}%",
        f"Nonpositive transmission (clipped to small positive value): {acquisition_info['nonpositive_transmission_percent']:.2f}%",
        "",
        "OUTPUTS",
        f"- {volume_npy.name}: raw 3-D NumPy volume array, axis order [z, y, x]",
        f"- {volume_npz.name}: volume and physical coordinate axes in one compressed archive",
        f"- {sino_npz.name}: full per-height sinogram stack used for reconstruction",
        f"- {pointcloud_png.name}: thresholded 3-D voxel point cloud",
        f"- {orthogonal_png.name}: representative XY, XZ and YZ cross-sections",
        f"- {mip_png.name}: three orthogonal maximum-intensity projections",
        f"- {montage_png.name}: multiple axial slices from bottom to top",
        "",
        "INTERPRETATION AND LIMITATIONS",
        "This educational 3-D volume is formed by reconstructing each vertical detector band",
        "with 2-D parallel-beam FBP and stacking the slices along z. It is not cone-beam CT.",
        "Correct z calibration, a stable rotation axis, parallel-ish illumination and consistent",
        "camera settings are essential. Voxel values are relative estimates, not medical CT values.",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")

    return VolumeReconstructionResult(
        output_dir=target,
        volume_npy_path=volume_npy,
        volume_npz_path=volume_npz,
        sinogram_stack_path=sino_npz,
        pointcloud_path=pointcloud_png,
        orthogonal_path=orthogonal_png,
        mip_path=mip_png,
        montage_path=montage_png,
        report_path=report_path,
        metadata_path=metadata_path,
        shape=tuple(int(v) for v in volume.shape),
        projection_count=int(len(angles)),
        detector_samples=int(sinograms.shape[2]),
        filter_name=filter_name,
        z_extent_cm=float(object_height_cm),
        xy_extent_cm=float(detector_width_cm),
    )
