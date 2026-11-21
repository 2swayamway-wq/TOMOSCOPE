"""Core image-import and projection-processing functions for TOMOSCOPE.

This capture-stage module converts camera photographs into calibrated
one-dimensional projection profiles and a sinogram CSV. The integrated GUI
then passes that CSV directly to tomoscope_reconstruction.py for 2-D imaging.
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ROISettings:
    """Region of interest, expressed as percentages of the full image."""
    x_start_pct: float = 15.0
    x_end_pct: float = 85.0
    center_y_pct: float = 50.0
    band_height_pct: float = 3.0


@dataclass
class ProcessSettings:
    angle_start_deg: float = 0.0
    angle_step_deg: float = 10.0
    detector_width_cm: float = 15.0
    roi: ROISettings = field(default_factory=ROISettings)


def natural_sort_key(path: str | Path):
    """Sort DSC_2.JPG before DSC_10.JPG."""
    name = Path(path).name
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def validate_image_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported image format: {p.suffix}. Use JPEG Fine (.JPG) from the Nikon D5000, "
            "or PNG/TIFF/BMP images. NEF/RAW files are not supported by this first version."
        )
    return p


def image_size(path: str | Path) -> tuple[int, int]:
    p = validate_image_path(path)
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            return im.size
    except Exception as exc:
        raise ValueError(f"Could not read image {p.name}: {exc}") from exc


def calculate_roi(width: int, height: int, roi: ROISettings) -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom), after EXIF orientation correction."""
    if not (0 <= roi.x_start_pct < roi.x_end_pct <= 100):
        raise ValueError("Detector left/right percentages must satisfy 0 <= left < right <= 100.")
    if not (0 <= roi.center_y_pct <= 100):
        raise ValueError("Band centre must be between 0 and 100 percent.")
    if not (0 < roi.band_height_pct <= 100):
        raise ValueError("Band height must be greater than 0 and at most 100 percent.")

    x0 = int(round(width * roi.x_start_pct / 100.0))
    x1 = int(round(width * roi.x_end_pct / 100.0))
    band_pixels = max(1, int(round(height * roi.band_height_pct / 100.0)))
    yc = int(round(height * roi.center_y_pct / 100.0))
    y0 = max(0, yc - band_pixels // 2)
    y1 = min(height, y0 + band_pixels)
    y0 = max(0, y1 - band_pixels)

    x0 = max(0, min(width - 1, x0))
    x1 = max(x0 + 1, min(width, x1))
    if x1 - x0 < 16:
        raise ValueError("The selected detector width is too narrow. Choose a wider horizontal region.")
    return x0, y0, x1, y1


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Approximate conversion of 8-bit sRGB JPEG values to linear-light values."""
    c = values.astype(np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_detector_profile(path: str | Path, roi: ROISettings) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Read a narrow horizontal band and average its rows into one detector profile.

    Returns:
      profile: 1-D linear-light grayscale profile from left to right
      image_size: EXIF-corrected (width, height)
      box: selected ROI (left, top, right, bottom)
    """
    p = validate_image_path(path)
    try:
        with Image.open(p) as original:
            im = ImageOps.exif_transpose(original)
            width, height = im.size
            box = calculate_roi(width, height, roi)
            crop = im.crop(box).convert("RGB")
            rgb8 = np.asarray(crop, dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not process {p.name}: {exc}") from exc

    linear = _srgb_to_linear(rgb8)
    # Standard linear-light luminance weights. The input is an ordinary camera JPEG,
    # so this is an approximation rather than laboratory-grade radiometry.
    gray = (0.2126 * linear[:, :, 0]
            + 0.7152 * linear[:, :, 1]
            + 0.0722 * linear[:, :, 2])
    profile = np.mean(gray, axis=0, dtype=np.float64).astype(np.float32)
    return profile, (width, height), box


def _write_matrix_csv(path: Path, detector_cm: np.ndarray,
                       matrix: np.ndarray, angles: Sequence[float],
                       value_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["detector_position_cm"] + [f"{value_name}_{a:g}deg" for a in angles])
        for i, s in enumerate(detector_cm):
            writer.writerow([f"{s:.6f}"] + [f"{float(v):.8f}" for v in matrix[:, i]])


def _save_sinogram_plot(path: Path, sino: np.ndarray, detector_cm: np.ndarray,
                        angles: Sequence[float], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    extent = [float(detector_cm[0]), float(detector_cm[-1]),
              float(angles[0]), float(angles[-1])]
    im = ax.imshow(sino, aspect="auto", origin="lower", extent=extent,
                   cmap="magma", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Detector position s (cm)")
    ax.set_ylabel("Scan angle θ (degrees)")
    fig.colorbar(im, ax=ax, label="Estimated attenuation  -ln(transmission)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_profile_plot(path: Path, detector_cm: np.ndarray, sino: np.ndarray,
                       angles: Sequence[float]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    # Plot up to four representative angles, selected by nearest available value.
    desired = [0.0, 45.0, 90.0, 135.0]
    chosen: list[int] = []
    for target in desired:
        idx = int(np.argmin(np.abs(np.asarray(angles, dtype=float) - target)))
        if idx not in chosen:
            chosen.append(idx)
    for idx in chosen:
        ax.plot(detector_cm, sino[idx], label=f"{angles[idx]:g}°")
    ax.set_title("Attenuation profiles at selected scan angles")
    ax.set_xlabel("Detector position s (cm)")
    ax.set_ylabel("Estimated attenuation")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Scan angle")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_roi_preview(path: Path, flat_path: Path, roi: ROISettings) -> None:
    with Image.open(flat_path) as original:
        im = ImageOps.exif_transpose(original).convert("RGB")
        w, h = im.size
        x0, y0, x1, y1 = calculate_roi(w, h, roi)
        # A thumbnail keeps this file convenient to open.
        im.thumbnail((1200, 900))
        sx, sy = im.width / w, im.height / h
        draw = ImageDraw.Draw(im)
        box = (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy))
        draw.rectangle(box, outline=(255, 40, 40), width=max(2, im.width // 450))
        im.save(path, quality=92)


def _copy_inputs(session_dir: Path, flat_path: Path, dark_path: Path | None,
                 projection_paths: Sequence[Path], angles: Sequence[float]) -> dict:
    refs_dir = session_dir / "input" / "references"
    proj_dir = session_dir / "input" / "projections"
    refs_dir.mkdir(parents=True, exist_ok=True)
    proj_dir.mkdir(parents=True, exist_ok=True)

    flat_dest = refs_dir / f"flat_reference{flat_path.suffix.lower()}"
    shutil.copy2(flat_path, flat_dest)
    dark_dest = None
    if dark_path:
        dark_dest = refs_dir / f"dark_reference{dark_path.suffix.lower()}"
        shutil.copy2(dark_path, dark_dest)

    manifest = []
    for idx, (src, angle) in enumerate(zip(projection_paths, angles)):
        safe_angle = f"{angle:06.1f}".replace(".", "p")
        dest = proj_dir / f"projection_{idx:03d}_{safe_angle}deg{src.suffix.lower()}"
        shutil.copy2(src, dest)
        manifest.append({
            "index": idx,
            "angle_deg": float(angle),
            "source_filename": src.name,
            "copied_filename": dest.name,
        })
    return {
        "flat_reference": str(flat_dest.relative_to(session_dir)),
        "dark_reference": str(dark_dest.relative_to(session_dir)) if dark_dest else None,
        "projections": manifest,
    }


def process_acquisition(
    flat_path: str | Path,
    dark_path: str | Path | None,
    projection_paths: Sequence[str | Path],
    session_dir: str | Path,
    settings: ProcessSettings,
) -> dict:
    """Calibrate imported photographs, create projections, and save sinogram PNG/CSV paths."""
    flat_path = validate_image_path(flat_path)
    dark_path = validate_image_path(dark_path) if dark_path else None
    projection_paths = [validate_image_path(p) for p in projection_paths]
    session_dir = Path(session_dir).expanduser().resolve()

    if len(projection_paths) < 3:
        raise ValueError("Please import at least 3 projection photographs. 18 or 36 are recommended.")
    if settings.angle_step_deg <= 0:
        raise ValueError("Angle step must be greater than zero.")
    if settings.detector_width_cm <= 0:
        raise ValueError("Detector width must be greater than zero.")

    angles = [settings.angle_start_deg + i * settings.angle_step_deg
              for i in range(len(projection_paths))]
    if any(a < 0 or a >= 180 for a in angles):
        raise ValueError(
            "For this parallel-beam setup, enter unique angles from 0° up to (but not including) 180°. "
            "For example, 18 images use start 0° and step 10°."
        )

    all_paths = [flat_path] + ([dark_path] if dark_path else []) + projection_paths
    dimensions = [(p, image_size(p)) for p in all_paths]
    reference_size = dimensions[0][1]
    mismatched = [(p.name, size) for p, size in dimensions if size != reference_size]
    if mismatched:
        details = "; ".join(f"{name}: {size[0]}×{size[1]}" for name, size in mismatched[:5])
        raise ValueError(
            f"All photographs must have the same dimensions after orientation correction. "
            f"Reference is {reference_size[0]}×{reference_size[1]}. Mismatches: {details}"
        )

    width, height = reference_size
    box = calculate_roi(width, height, settings.roi)
    flat_profile, _, _ = load_detector_profile(flat_path, settings.roi)
    if dark_path:
        dark_profile, _, _ = load_detector_profile(dark_path, settings.roi)
    else:
        dark_profile = np.zeros_like(flat_profile)

    flat_signal = flat_profile - dark_profile
    if np.any(flat_signal <= 1e-6):
        bad = int(np.count_nonzero(flat_signal <= 1e-6))
        raise ValueError(
            f"The flat reference is too dark in {bad} detector pixels. Increase LED brightness or exposure, "
            "and make sure the blank reference is photographed with the LED on."
        )

    raw_profiles = []
    transmission_profiles = []
    attenuation_profiles = []
    above_one_count = 0
    below_zero_count = 0
    total_values = 0
    epsilon = 1e-4

    for projection_path in projection_paths:
        profile, _, _ = load_detector_profile(projection_path, settings.roi)
        corrected = profile - dark_profile
        transmission = corrected / flat_signal
        above_one_count += int(np.count_nonzero(transmission > 1.0))
        below_zero_count += int(np.count_nonzero(transmission <= 0.0))
        total_values += transmission.size
        # This educational model assumes an absorbing object, so T is limited to
        # (epsilon, 1]. Values above 1 are treated as zero attenuation and counted.
        transmission_safe = np.clip(transmission, epsilon, 1.0)
        attenuation = -np.log(transmission_safe)
        raw_profiles.append(profile)
        transmission_profiles.append(transmission)
        attenuation_profiles.append(attenuation.astype(np.float32))

    raw_matrix = np.stack(raw_profiles, axis=0)
    transmission_matrix = np.stack(transmission_profiles, axis=0)
    sinogram = np.stack(attenuation_profiles, axis=0)
    detector_cm = np.linspace(-settings.detector_width_cm / 2.0,
                              settings.detector_width_cm / 2.0,
                              flat_profile.size, dtype=np.float64)

    # Prepare a clean session folder. Refuse to overwrite existing results.
    results_dir = session_dir / "results"
    if results_dir.exists() and any(results_dir.iterdir()):
        raise FileExistsError(
            f"The results folder already contains files: {results_dir}. "
            "Choose a new/empty session folder or move the old results first."
        )
    results_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_inputs(session_dir, flat_path, dark_path, projection_paths, angles)

    _write_matrix_csv(results_dir / "raw_intensity_profiles.csv", detector_cm,
                      raw_matrix, angles, "intensity")
    _write_matrix_csv(results_dir / "transmission_profiles.csv", detector_cm,
                      transmission_matrix, angles, "transmission")
    _write_matrix_csv(results_dir / "sinogram_attenuation.csv", detector_cm,
                      sinogram, angles, "attenuation")
    _save_sinogram_plot(results_dir / "sinogram.png", sinogram, detector_cm, angles,
                        "TOMOSCOPE — measured attenuation sinogram")
    _save_profile_plot(results_dir / "selected_profiles.png", detector_cm, sinogram, angles)
    _save_roi_preview(results_dir / "roi_preview.jpg", flat_path, settings.roi)

    clipped_high_pct = 100.0 * above_one_count / max(1, total_values)
    clipped_low_pct = 100.0 * below_zero_count / max(1, total_values)
    metadata = {
        "project": "TOMOSCOPE image capture and projection processing",
        "created": datetime.now().isoformat(timespec="seconds"),
        "camera_workflow": "Photographs imported from Nikon D5000 SD card/DCIM or USB transfer folder",
        "reference_image_size_pixels": {"width": width, "height": height},
        "projection_count": len(projection_paths),
        "angles_deg": angles,
        "angle_start_deg": settings.angle_start_deg,
        "angle_step_deg": settings.angle_step_deg,
        "detector_width_cm_assumed": settings.detector_width_cm,
        "roi_percent": asdict(settings.roi),
        "roi_pixels": {"left": box[0], "top": box[1], "right": box[2], "bottom": box[3]},
        "detector_sample_count": int(flat_profile.size),
        "dark_reference_used": bool(dark_path),
        "flat_reference_filename": flat_path.name,
        "dark_reference_filename": dark_path.name if dark_path else None,
        "projection_files": [p.name for p in projection_paths],
        "transmission_above_1_0_percent_clipped_to_zero_attenuation": clipped_high_pct,
        "nonpositive_transmission_percent_clipped_to_epsilon": clipped_low_pct,
        "srgb_to_linear_conversion": "Applied as an approximation to JPEG images",
        "limitations": [
            "JPEG camera processing is not laboratory-grade radiometry.",
            "The model assumes approximately parallel rays and predominantly absorbing material.",
            "Refraction, scattering, reflections, camera auto-processing and sample movement can affect results.",
            "Detector positions are scaled to the entered phantom width; this assumes the ROI spans that width."
        ],
        "copied_inputs": copied,
    }
    (results_dir / "session_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report = [
        "TOMOSCOPE — IMAGE CAPTURE / PROJECTION PROCESSING REPORT",
        "=" * 62,
        f"Date: {metadata['created']}",
        f"Image dimensions: {width} × {height} pixels",
        f"Projection photographs: {len(projection_paths)}",
        f"Angles: {angles[0]:g}° to {angles[-1]:g}° in {settings.angle_step_deg:g}° steps",
        f"Detector samples per projection: {flat_profile.size}",
        f"Assumed detector width: {settings.detector_width_cm:g} cm",
        f"ROI pixels: left={box[0]}, top={box[1]}, right={box[2]}, bottom={box[3]}",
        f"Dark reference used: {'Yes' if dark_path else 'No (zero baseline assumed)'}",
        f"Values with transmission > 1.00 (clipped to zero attenuation): {clipped_high_pct:.2f}%",
        f"Values with transmission <= 0 (clipped to epsilon): {clipped_low_pct:.2f}%",
        "",
        "OUTPUT FILES",
        "- sinogram.png: visual plot of attenuation versus detector position and angle",
        "- sinogram_attenuation.csv: calibrated attenuation measurements",
        "- transmission_profiles.csv: transmission ratio at every detector position and angle",
        "- raw_intensity_profiles.csv: measured camera brightness profiles",
        "- selected_profiles.png: example attenuation profiles",
        "- roi_preview.jpg: shows the selected image band and detector width",
        "- session_metadata.json: acquisition settings and file manifest",
        "",
        "IMPORTANT INTERPRETATION",
        "These are educational, camera-derived transmission estimates. They are not medical",
        "measurements. JPEG tone processing, scattering, refraction, reflections, imperfect",
        "parallel illumination and movement can cause deviations from the ideal model.",
    ]
    (results_dir / "quality_report.txt").write_text("\n".join(report), encoding="utf-8")

    return {
        "session_dir": str(session_dir),
        "results_dir": str(results_dir),
        "sinogram_path": str(results_dir / "sinogram.png"),
        "sinogram_csv_path": str(results_dir / "sinogram_attenuation.csv"),
        "report_path": str(results_dir / "quality_report.txt"),
        "projection_count": len(projection_paths),
        "detector_samples": int(flat_profile.size),
        "image_size": reference_size,
        "high_transmission_pct": clipped_high_pct,
        "low_transmission_pct": clipped_low_pct,
    }
