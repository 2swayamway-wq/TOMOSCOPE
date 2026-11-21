import io
import math
import tempfile
from pathlib import Path
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageOps

# Import your partner's real TOMOSCOPE math & camera engines
from tomoscope_core import (
    ROISettings,
    ProcessSettings,
    natural_sort_key,
    calculate_roi,
    process_acquisition,
)
from tomoscope_reconstruction import (
    SinogramData,
    load_sinogram_csv,
    reconstruct_arrays,
)
from tomoscope_3d import (
    reconstruct_3d_from_images,
    reconstruct_volume_from_sinogram_stack,
)

# --- PAGE CONFIG ---
st.set_page_config(page_title="TOMOSCOPE | 2D & 3D Optical CT", page_icon="🔬", layout="wide")

# --- CLEAN LIGHT-MODE SCIENTIFIC STYLING ---
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }
    label, p, span, li, div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"] p {
        color: #1e293b !important;
    }
    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f1f5f9 !important;
        border: 1px dashed #94a3b8 !important;
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #0f172a !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background-color: #ffffff !important;
        color: #0284c7 !important;
        border: 1px solid #cbd5e1 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="popover"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }
    li[role="option"]:hover {
        background-color: #e0f2fe !important;
    }
    .tomo-badge {
        display: inline-block;
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        color: #0369a1 !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 8px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] div {
        color: #0284c7 !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        white-space: normal !important;
        line-height: 1.2 !important;
    }
    .stage-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .stage-num {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1 !important;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .stage-card h4 {
        color: #0f172a !important;
        font-size: 16px !important;
        margin: 2px 0 8px 0 !important;
    }
    .stage-card p {
        color: #475569 !important;
        font-size: 13.5px !important;
        line-height: 1.5 !important;
        margin: 0 !important;
    }
    .formula-box {
        background-color: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 7px 10px;
        font-family: monospace;
        color: #0369a1 !important;
        font-size: 12.5px;
        font-weight: 600;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

plt.style.use("default")

# --- HEADER ---
st.markdown('<div class="tomo-badge">2D & 3D OPTICAL COMPUTED TOMOGRAPHY • COMPETITION MODEL</div>', unsafe_allow_html=True)
st.title("🔬 TOMOSCOPE: Visible-Light Tomography Simulator & Scanner")
st.caption("By Priyam Patro and Shanmukh Teja • Calibrated Beer–Lambert Sinograms & Filtered Backprojection")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Source")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Upload Real Scan Photos", "Demo Simulation", "Upload Sinogram CSV"]
)

st.sidebar.header("2. Reconstruction Settings")
filter_name = st.sidebar.selectbox(
    "FBP Frequency Filter",
    ["Hann", "Ram-Lak", "Shepp-Logan"],
    index=0
)
resolution = st.sidebar.select_slider(
    "2D Grid Resolution (pixels)",
    options=[128, 192, 256, 320, 384],
    value=256
)


# --- HELPER PLOTTING FUNCTIONS ---
def plot_tomo_grid(image, title, xlabel, ylabel, extent, cmap="magma", aspect="equal"):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f1f5f9")

    finite = image[np.isfinite(image)]
    if finite.size > 0:
        vmin, vmax = np.percentile(finite, [1.0, 99.5])
        if abs(vmax - vmin) < 1e-12:
            vmin, vmax = float(np.min(finite)), float(np.max(finite)) + 1e-9
    else:
        vmin, vmax = 0.0, 1.0

    im = ax.imshow(
        image,
        origin="lower",
        extent=extent,
        cmap=cmap,
        aspect=aspect,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#334155")
    ax.set_ylabel(ylabel, fontsize=9, color="#334155")
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#475569", labelsize=8)
    return fig


def build_vpython_style_plotly(volume, x_cm, y_cm, z_cm, percentile, slice_z_val,
                               show_voxels=True, show_plane=True, show_box=True):
    """WebGL 3D scene matching tomoscope_vpython_viewer.py (5-band palette + Z slice plane)."""
    palette_hex = [
        "#145cd1",  # blue
        "#00b8e0",  # cyan
        "#2ed19e",  # teal/green
        "#fab833",  # gold
        "#fa4d1f",  # orange/red
    ]
    band_labels = ["Band 1 (Low)", "Band 2", "Band 3 (Mid)", "Band 4", "Band 5 (High)"]

    xmin, xmax = float(x_cm[0]), float(x_cm[-1])
    ymin, ymax = float(y_cm[0]), float(y_cm[-1])
    zmin, zmax = float(z_cm[0]), float(z_cm[-1])

    traces = []
    displayed_count = 0

    if show_voxels:
        positive_mask = np.isfinite(volume) & (volume > 0)
        positive_vals = volume[positive_mask]
        if positive_vals.size >= 10:
            threshold = float(np.percentile(positive_vals, percentile))
            selected_flat = np.flatnonzero(positive_mask & (volume >= threshold))
            max_points = 30000
            if selected_flat.size > max_points:
                rng = np.random.default_rng(1729)
                chosen = np.sort(rng.choice(selected_flat.size, size=max_points, replace=False))
                selected_flat = selected_flat[chosen]

            iz, iy, ix = np.unravel_index(selected_flat, volume.shape)
            intensity = volume[iz, iy, ix]
            displayed_count = int(intensity.size)

            if displayed_count > 0:
                cuts = np.quantile(intensity, [0.2, 0.4, 0.6, 0.8])
                band_ids = np.digitize(intensity, cuts, right=True)
                for b_idx in range(5):
                    mask = band_ids == b_idx
                    if not np.any(mask):
                        continue
                    traces.append(
                        go.Scatter3d(
                            x=x_cm[ix[mask]],
                            y=y_cm[iy[mask]],
                            z=z_cm[iz[mask]],
                            mode="markers",
                            name=band_labels[b_idx],
                            marker=dict(
                                size=3.2,
                                color=palette_hex[b_idx],
                                opacity=0.92,
                            ),
                            hovertemplate="x: %{x:.2f} cm<br>y: %{y:.2f} cm<br>z: %{z:.2f} cm<extra></extra>",
                        )
                    )

    if show_plane:
        traces.append(
            go.Surface(
                x=[[xmin, xmax], [xmin, xmax]],
                y=[[ymin, ymin], [ymax, ymax]],
                z=[[slice_z_val, slice_z_val], [slice_z_val, slice_z_val]],
                colorscale=[[0, "rgb(13, 199, 255)"], [1, "rgb(13, 199, 255)"]],
                opacity=0.26,
                showscale=False,
                name=f"Slice z = {slice_z_val:.2f} cm",
                hoverinfo="skip",
            )
        )

    if show_box:
        bx = [xmin, xmax, xmax, xmin, xmin, None,
              xmin, xmax, xmax, xmin, xmin, None,
              xmin, xmin, None, xmax, xmax, None,
              xmax, xmax, None, xmin, xmin]
        by = [ymin, ymin, ymax, ymax, ymin, None,
              ymin, ymin, ymax, ymax, ymin, None,
              ymin, ymin, None, ymin, ymin, None,
              ymax, ymax, None, ymax, ymax]
        bz = [zmin, zmin, zmin, zmin, zmin, None,
              zmax, zmax, zmax, zmax, zmax, None,
              zmin, zmax, None, zmin, zmax, None,
              zmin, zmax, None, zmin, zmax]
        traces.append(
            go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="lines",
                name="Scan Boundary",
                line=dict(color="rgb(85, 115, 165)", width=3),
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="rgb(6, 9, 17)",
        plot_bgcolor="rgb(6, 9, 17)",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=0, r=0, t=36, b=0),
        height=520,
        title=dict(
            text=f"Interactive 3D Voxel Volume ({displayed_count:,} voxels ≥ {percentile:.1f}th pct)",
            x=0.03,
            font=dict(size=13, color="#f8fafc"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.75)",
            font=dict(size=10, color="#f1f5f9"),
        ),
        scene=dict(
            bgcolor="rgb(6, 9, 17)",
            aspectmode="data",
            xaxis=dict(title="X (cm)", gridcolor="#1e293b", zerolinecolor="#334155", showbackground=False),
            yaxis=dict(title="Y (cm)", gridcolor="#1e293b", zerolinecolor="#334155", showbackground=False),
            zaxis=dict(title="Z Height (cm)", gridcolor="#1e293b", zerolinecolor="#334155", showbackground=False),
        ),
    )
    return fig


def render_3d_volume_explorer(volume, x_cm, y_cm, z_cm, key_prefix="vol"):
    """Interactive VPython-style 3D canvas + Z-slice inspector + NPZ export."""
    st.markdown("---")
    st.subheader(f"🧊 Interactive 3D Voxel Reconstruction ({volume.shape[0]} Z × {volume.shape[1]} Y × {volume.shape[2]} X)")
    st.caption("Mouse / Touch: Drag to orbit 360° • Scroll / Pinch to zoom • Right-drag / Shift-drag to pan")

    c_top1, c_top2, c_top3, c_top4 = st.columns([2, 2, 1, 1])
    pct = c_top1.slider(
        "Voxel Display Threshold (%)",
        min_value=75.0,
        max_value=99.5,
        value=90.0,
        step=0.5,
        key=f"{key_prefix}_pct",
    )
    z_idx = c_top2.slider(
        "Horizontal Height Slice Z",
        min_value=0,
        max_value=len(z_cm) - 1,
        value=len(z_cm) // 2,
        key=f"{key_prefix}_zslice",
    )
    show_plane = c_top3.checkbox("Slice Plane", value=True, key=f"{key_prefix}_plane")
    show_box = c_top4.checkbox("Scan Box", value=True, key=f"{key_prefix}_box")

    vc1, vc2 = st.columns([1.35, 1.0])

    # Left: Full interactive 3D WebGL scene (VPython style)
    with vc1:
        fig_3d = build_vpython_style_plotly(
            volume,
            x_cm,
            y_cm,
            z_cm,
            percentile=pct,
            slice_z_val=float(z_cm[z_idx]),
            show_voxels=True,
            show_plane=show_plane,
            show_box=show_box,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    # Right: Corresponding 2D XY Cross-Section at the cyan slice plane
    with vc2:
        finite = volume[np.isfinite(volume)]
        lo, hi = np.percentile(finite, [1.0, 99.5]) if finite.size > 0 else (0.0, 1.0)
        if abs(hi - lo) < 1e-12:
            hi = lo + 1e-9

        fig_z, ax_z = plt.subplots(figsize=(4.8, 4.6))
        fig_z.patch.set_facecolor("#ffffff")
        im_z = ax_z.imshow(
            volume[z_idx],
            origin="lower",
            extent=[float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])],
            cmap="magma",
            vmin=lo,
            vmax=hi,
            interpolation="nearest",
        )
        ax_z.set_title(
            f"XY Cross-Section at z = {z_cm[z_idx]:.3f} cm ({z_idx + 1}/{len(z_cm)})",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )
        ax_z.set_xlabel("x (cm)", fontsize=9)
        ax_z.set_ylabel("y (cm)", fontsize=9)
        ax_z.tick_params(labelsize=8)
        fig_z.colorbar(im_z, ax=ax_z, fraction=0.046, pad=0.04, label="Relative attenuation")
        st.pyplot(fig_z)
        plt.close(fig_z)

    # Orthogonal XZ/YZ views + NPZ archive download for local VPython
    mid_y, mid_x = len(y_cm) // 2, len(x_cm) // 2
    ox1, ox2 = st.columns(2)
    with ox1:
        fig_xz = plot_tomo_grid(
            volume[:, mid_y, :],
            f"Front Vertical Cross-Section (XZ at y = {y_cm[mid_y]:.2f} cm)",
            "x (cm)",
            "z height (cm)",
            [float(x_cm[0]), float(x_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            aspect="auto",
        )
        st.pyplot(fig_xz)
        plt.close(fig_xz)
    with ox2:
        fig_yz = plot_tomo_grid(
            volume[:, :, mid_x],
            f"Side Vertical Cross-Section (YZ at x = {x_cm[mid_x]:.2f} cm)",
            "y (cm)",
            "z height (cm)",
            [float(y_cm[0]), float(y_cm[-1]), float(z_cm[0]), float(z_cm[-1])],
            aspect="auto",
        )
        st.pyplot(fig_yz)
        plt.close(fig_yz)

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, volume=volume.astype(np.float32), x_cm=x_cm, y_cm=y_cm, z_cm=z_cm)
    st.download_button(
        "⬇️ Download 3D Voxel Archive (volume_filtered.npz for local VPython viewer)",
        data=npz_buf.getvalue(),
        file_name="volume_filtered.npz",
        mime="application/octet-stream",
        key=f"{key_prefix}_dl_npz",
    )


def build_synthetic_scan(num_views, phantom_choice):
    """Generates exact analytical line integrals matching generate_demo_data.py."""
    width_cm = 15.0
    n_det = 256
    s = np.linspace(-width_cm / 2, width_cm / 2, n_det)
    angles = np.linspace(0.0, 180.0, num_views, endpoint=False)

    if phantom_choice == "4 Circular Inclusions (Demo)":
        objects = [
            (-2.8, 1.8, 1.05, 0.35),
            (2.4, 2.0, 0.90, 0.48),
            (3.0, -1.8, 0.72, 0.58),
            (-1.2, -3.0, 0.82, 0.42),
        ]
    elif phantom_choice == "Single Off-Center Point":
        objects = [(2.5, 3.0, 0.55, 0.80)]
    else:
        objects = [(0.0, 0.0, 4.2, 0.25), (0.0, 0.0, 2.2, -0.20)]

    coords = np.linspace(-width_cm / 2, width_cm / 2, resolution)
    xx, yy = np.meshgrid(coords, coords)
    phantom = np.zeros_like(xx)
    for cx, cy, r, mu in objects:
        phantom[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] += mu
    phantom[xx ** 2 + yy ** 2 > (width_cm / 2) ** 2] = np.nan

    sino = np.zeros((num_views, n_det), dtype=np.float64)
    for i, ang in enumerate(angles):
        th = np.deg2rad(ang)
        for cx, cy, r, mu in objects:
            offset = s - (cx * np.cos(th) + cy * np.sin(th))
            inside = np.abs(offset) < r
            chord = np.zeros_like(s)
            chord[inside] = 2.0 * np.sqrt(np.maximum(0.0, r * r - offset[inside] ** 2))
            sino[i] += mu * chord
    sino = np.maximum(sino, 0.0)

    sino_data = SinogramData(
        detector_positions_cm=s,
        angles_deg=angles,
        attenuation=sino,
        source_path=Path("synthetic_demo.csv"),
    )
    return phantom, sino_data


@st.cache_data(show_spinner=False)
def run_3d_cylinder_demo(num_views, xy_res, filt_name):
    """Simulates & reconstructs the 3-cylinder 3D dataset from generate_3d_demo_data.py."""
    x_width_cm = 8.0
    z_height_cm = 4.0
    n_det = 160
    z_slices = 24
    s = np.linspace(-x_width_cm / 2, x_width_cm / 2, n_det)
    z = np.linspace(-z_height_cm / 2, z_height_cm / 2, z_slices)
    angles = np.linspace(0.0, 180.0, num_views, endpoint=False)

    cylinders = [
        (-1.45, 0.70, 0.48, 0.62, -1.95, 1.95),
        (1.45, 0.85, 0.43, 0.78, -0.85, 1.95),
        (0.15, -1.50, 0.52, 0.70, -1.95, 0.55),
    ]

    sino_cube = np.zeros((z_slices, num_views, n_det), dtype=np.float64)
    for ai, angle_deg in enumerate(angles):
        theta = np.deg2rad(angle_deg)
        for cx, cy, radius, mu, z_min, z_max in cylinders:
            offset = s - (cx * np.cos(theta) + cy * np.sin(theta))
            inside = np.abs(offset) < radius
            chord = np.zeros_like(s)
            chord[inside] = 2.0 * np.sqrt(np.maximum(0.0, radius ** 2 - offset[inside] ** 2))
            active = (z >= z_min) & (z <= z_max)
            sino_cube[active, ai, :] += mu * chord[None, :]

    volume, x_cm, y_cm, z_cm, _ = reconstruct_volume_from_sinogram_stack(
        sino_cube, s, angles, z, resolution=min(xy_res, 128), filter_name=filt_name
    )
    return volume, x_cm, y_cm, z_cm


# =====================================================================
# MODE 1: UPLOAD REAL CAMERA PHOTOS (USES PARTNER'S EXACT ENGINE)
# =====================================================================
if mode == "Upload Real Scan Photos":
    st.sidebar.header("3. Camera & ROI Calibration")
    flat_file = st.sidebar.file_uploader("Flat Reference (LED ON, no object)", type=["jpg", "jpeg", "png", "tif"])
    dark_file = st.sidebar.file_uploader("Dark Reference (LED OFF, optional)", type=["jpg", "jpeg", "png", "tif"])
    proj_files = st.sidebar.file_uploader(
        "Projection Scan Photos (select all)",
        type=["jpg", "jpeg", "png", "tif"],
        accept_multiple_files=True,
    )

    col_a, col_b = st.sidebar.columns(2)
    start_angle = col_a.number_input("Start Angle (°)", value=0.0, step=1.0)
    auto_step = round(180.0 / len(proj_files), 4) if len(proj_files) >= 3 else 10.0
    step_angle = col_b.number_input("Angle Step (°)", value=float(auto_step), step=0.5)
    detector_width_cm = st.sidebar.number_input("ROI Physical Width (cm)", value=15.0, min_value=0.1, step=0.5)

    st.sidebar.subheader("2D Detector Strip ROI (%)")
    r1, r2 = st.sidebar.columns(2)
    x_start_pct = r1.number_input("Left Edge %", value=15.0, min_value=0.0, max_value=99.0)
    x_end_pct = r2.number_input("Right Edge %", value=85.0, min_value=1.0, max_value=100.0)
    center_y_pct = r1.number_input("Strip Centre Y %", value=50.0, min_value=0.0, max_value=100.0)
    band_height_pct = r2.number_input("Strip Height %", value=3.0, min_value=0.5, max_value=100.0)

    st.sidebar.subheader("3D Volume Reconstruction")
    enable_3d = st.sidebar.checkbox("Enable Full 3D Volume Reconstruction", value=True)
    if enable_3d:
        z1, z2 = st.sidebar.columns(2)
        z_top_pct = z1.number_input("3D Top ROI %", value=15.0, min_value=0.0, max_value=99.0)
        z_bottom_pct = z2.number_input("3D Bottom ROI %", value=85.0, min_value=1.0, max_value=100.0)
        object_height_cm = z1.number_input("Object Height (cm)", value=3.0, min_value=0.1, step=0.5)
        z_slices = int(z2.number_input("Z Height Slices", value=24, min_value=8, max_value=64, step=4))

    if flat_file is not None:
        with st.expander("👁️ Preview Selected ROI Boxes on Flat Reference Photo", expanded=(len(proj_files) < 3)):
            roi_cfg = ROISettings(x_start_pct, x_end_pct, center_y_pct, band_height_pct)
            pil_flat = ImageOps.exif_transpose(Image.open(flat_file)).convert("RGB")
            w, h = pil_flat.size
            try:
                x0, y0, x1, y1 = calculate_roi(w, h, roi_cfg)
                preview_img = pil_flat.copy()
                draw = ImageDraw.Draw(preview_img)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 35, 35), width=max(4, w // 250))
                if enable_3d:
                    zy0 = int(round(h * z_top_pct / 100.0))
                    zy1 = int(round(h * z_bottom_pct / 100.0))
                    draw.rectangle([x0, zy0, x1, zy1], outline=(2, 132, 199), width=max(4, w // 250))
                st.image(
                    preview_img,
                    caption="Red box = 2D central strip | Blue box = Full 3D object height region",
                    use_container_width=True,
                )
            except Exception as roi_err:
                st.error(str(roi_err))

    if flat_file is not None and len(proj_files) >= 3:
        btn_label = (
            "🚀 Process Camera Photos → 2D & 3D Reconstruction"
            if enable_3d
            else "🚀 Process Camera Photos → 2D Reconstruction"
        )
        if st.button(btn_label, type="primary", use_container_width=True):
            try:
                with st.spinner("Calibrating camera images and building sinogram..."):
                    tmp_dir = Path(tempfile.mkdtemp(prefix="tomoscope_web_"))
                    flat_path = tmp_dir / flat_file.name
                    flat_path.write_bytes(flat_file.getvalue())

                    dark_path = None
                    if dark_file is not None:
                        dark_path = tmp_dir / dark_file.name
                        dark_path.write_bytes(dark_file.getvalue())

                    sorted_uploads = sorted(proj_files, key=lambda f: natural_sort_key(f.name))
                    proj_paths = []
                    for uf in sorted_uploads:
                        p = tmp_dir / uf.name
                        p.write_bytes(uf.getvalue())
                        proj_paths.append(p)

                    roi_settings = ROISettings(x_start_pct, x_end_pct, center_y_pct, band_height_pct)
                    proc_settings = ProcessSettings(start_angle, step_angle, detector_width_cm, roi_settings)
                    session_dir = tmp_dir / "session"

                    cap_res = process_acquisition(flat_path, dark_path, proj_paths, session_dir, proc_settings)
                    sino_data = load_sinogram_csv(cap_res["sinogram_csv_path"])
                    x_cm, y_cm, bp, fbp, info = reconstruct_arrays(
                        sino_data, resolution=resolution, filter_name=filter_name
                    )

                    st.session_state["real_scan_done"] = True
                    st.session_state["cap_res"] = cap_res
                    st.session_state["sino_data"] = sino_data
                    st.session_state["recon_2d"] = (x_cm, y_cm, bp, fbp, info)
                    st.session_state["csv_bytes"] = Path(cap_res["sinogram_csv_path"]).read_bytes()
                    st.session_state["report_text"] = Path(cap_res["report_path"]).read_text(encoding="utf-8")

                if enable_3d:
                    prog_bar = st.progress(0.0, text="Reconstructing 3D height slices...")
                    angles_list = [start_angle + i * step_angle for i in range(len(proj_paths))]
                    vol_res = reconstruct_3d_from_images(
                        flat_path=flat_path,
                        dark_path=dark_path,
                        projection_paths=proj_paths,
                        angles_deg=angles_list,
                        output_dir=session_dir / "results" / "3d_reconstruction",
                        x_start_pct=x_start_pct,
                        x_end_pct=x_end_pct,
                        z_top_pct=z_top_pct,
                        z_bottom_pct=z_bottom_pct,
                        detector_width_cm=detector_width_cm,
                        object_height_cm=object_height_cm,
                        z_slices=z_slices,
                        resolution=min(resolution, 128),
                        filter_name=filter_name,
                        progress_callback=lambda done, total: prog_bar.progress(
                            done / total, text=f"Reconstructing 3D slice {done}/{total}..."
                        ),
                    )
                    prog_bar.empty()
                    with np.load(vol_res.volume_npz_path) as npz:
                        st.session_state["vol_3d_data"] = (
                            npz["volume"].copy(),
                            npz["x_cm"].copy(),
                            npz["y_cm"].copy(),
                            npz["z_cm"].copy(),
                        )
                else:
                    st.session_state["vol_3d_data"] = None
            except Exception as exc:
                st.error(f"Processing error: {exc}")

        if st.session_state.get("real_scan_done"):
            sino_data = st.session_state["sino_data"]
            x_cm, y_cm, bp, fbp, info = st.session_state["recon_2d"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scan Views", f"{info['projection_count']} photos")
            m2.metric("Detector Samples", f"{info['detector_samples']} px")
            m3.metric("Field of View", f"{info['detector_width_cm']:.1f} cm")
            m4.metric("Filter", info["filter_name"])

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                sino_extent = [
                    float(sino_data.detector_positions_cm[0]),
                    float(sino_data.detector_positions_cm[-1]),
                    float(sino_data.angles_deg[0]),
                    float(sino_data.angles_deg[-1]),
                ]
                fig1 = plot_tomo_grid(
                    sino_data.attenuation,
                    "1. Calibrated Sinogram p(θ, s)",
                    "Detector position s (cm)",
                    "Scan angle θ (degrees)",
                    sino_extent,
                    aspect="auto",
                )
                st.pyplot(fig1)
                plt.close(fig1)

            xy_extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
            with c2:
                fig2 = plot_tomo_grid(bp, "2. Simple Backprojection (Blurred)", "x (cm)", "y (cm)", xy_extent)
                st.pyplot(fig2)
                plt.close(fig2)
            with c3:
                fig3 = plot_tomo_grid(fbp, f"3. Filtered Backprojection ({filter_name})", "x (cm)", "y (cm)", xy_extent)
                st.pyplot(fig3)
                plt.close(fig3)

            d1, d2 = st.columns(2)
            d1.download_button(
                "⬇️ Download Calibrated Sinogram CSV",
                data=st.session_state["csv_bytes"],
                file_name="sinogram_attenuation.csv",
                mime="text/csv",
                use_container_width=True,
            )
            with d2.expander("📄 View Camera Calibration & Quality Report"):
                st.text(st.session_state["report_text"])

            vol_3d_data = st.session_state.get("vol_3d_data")
            if vol_3d_data is not None:
                render_3d_volume_explorer(*vol_3d_data, key_prefix="real3d")
    else:
        st.info(
            "👈 Upload your **Flat Reference Photo (LED ON)** and at least **3 Projection Scan Photos** "
            "in the left sidebar, then click **Process Camera Photos** (or switch to **Demo Simulation**)."
        )


# =====================================================================
# MODE 2: DEMO SIMULATION (2D & 3D SYNTHETIC MODELS)
# =====================================================================
elif mode == "Demo Simulation":
    st.sidebar.header("3. Simulation Controls")
    sim_dim = st.sidebar.radio(
        "Simulation Dimension",
        ["2D Cross-Section", "3D Voxel Volume (3 Cylinders)"]
    )
    num_views = st.sidebar.select_slider(
        "Projection Views (over 180°)",
        options=[6, 9, 18, 36, 72],
        value=36
    )

    if sim_dim == "2D Cross-Section":
        phantom_choice = st.sidebar.selectbox(
            "Synthetic Phantom",
            ["4 Circular Inclusions (Demo)", "Single Off-Center Point", "Hollow Ring"]
        )
        inspect_angle = st.sidebar.slider("Inspect Light Ray Angle θ (°)", 0, 179, 45)

        phantom, sino_data = build_synthetic_scan(num_views, phantom_choice)
        x_cm, y_cm, bp, fbp, info = reconstruct_arrays(sino_data, resolution=resolution, filter_name=filter_name)

        valid = np.isfinite(phantom) & np.isfinite(fbp)
        p_v = phantom[valid]
        r_v = fbp[valid]
        p_norm = (p_v - p_v.min()) / (p_v.max() - p_v.min() + 1e-8)
        r_norm = (r_v - r_v.min()) / (r_v.max() - r_v.min() + 1e-8)
        rmse = float(np.sqrt(np.mean((p_norm - r_norm) ** 2)))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scan Views", f"{num_views} angles")
        m2.metric("Angular Step", f"{180 / num_views:.1f}°")
        m3.metric("Image RMSE", f"{rmse:.4f}")
        m4.metric("Filter", filter_name)

        st.markdown("<br>", unsafe_allow_html=True)
        xy_extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
        sino_extent = [
            float(sino_data.detector_positions_cm[0]),
            float(sino_data.detector_positions_cm[-1]),
            float(sino_data.angles_deg[0]),
            float(sino_data.angles_deg[-1]),
        ]

        c1, c2, c3 = st.columns(3)
        with c1:
            fig1 = plot_tomo_grid(phantom, "1. Known Phantom μ(x, y)", "x (cm)", "y (cm)", xy_extent)
            ax1 = fig1.axes[0]
            rad = np.deg2rad(inspect_angle)
            nx, ny = np.cos(rad), np.sin(rad)
            dx, dy = -np.sin(rad), np.cos(rad)
            for s_val in [-3.5, 0.0, 3.5]:
                t = np.linspace(-8.0, 8.0, 40)
                ax1.plot(s_val * nx + t * dx, s_val * ny + t * dy, color="#0284c7",
                         lw=2.0 if s_val == 0 else 1.0, alpha=0.9 if s_val == 0 else 0.5)
            ax1.annotate("", xy=(3.0 * nx, 3.0 * ny), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="#d97706", lw=2.2))
            ax1.set_xlim(xy_extent[0], xy_extent[1])
            ax1.set_ylim(xy_extent[2], xy_extent[3])
            st.pyplot(fig1)
            plt.close(fig1)

        with c2:
            fig2 = plot_tomo_grid(
                sino_data.attenuation,
                "2. Sinogram p(θ, s)",
                "Detector position s (cm)",
                "Scan angle θ (degrees)",
                sino_extent,
                aspect="auto",
            )
            fig2.axes[0].axhline(inspect_angle, color="#0284c7", linestyle="--", lw=1.8)
            st.pyplot(fig2)
            plt.close(fig2)

        with c3:
            fig3 = plot_tomo_grid(fbp, f"3. Filtered Backprojection ({filter_name})", "x (cm)", "y (cm)", xy_extent)
            st.pyplot(fig3)
            plt.close(fig3)

    else:
        with st.spinner("Reconstructing 3D multi-height cylinder volume..."):
            vol_demo, x_cm, y_cm, z_cm = run_3d_cylinder_demo(num_views, resolution, filter_name)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Scan Views", f"{num_views} angles")
        m2.metric("3D Volume Grid", f"{vol_demo.shape[0]}×{vol_demo.shape[1]}×{vol_demo.shape[2]}")
        m3.metric("Physical Height", "4.0 cm")
        m4.metric("Filter", filter_name)

        render_3d_volume_explorer(vol_demo, x_cm, y_cm, z_cm, key_prefix="demo3d")


# =====================================================================
# MODE 3: UPLOAD EXISTING SINOGRAM CSV
# =====================================================================
else:
    csv_file = st.sidebar.file_uploader("Upload sinogram_attenuation.csv", type=["csv"])
    if csv_file is not None:
        try:
            tmp_csv = Path(tempfile.mkdtemp()) / "sinogram_attenuation.csv"
            tmp_csv.write_bytes(csv_file.getvalue())
            sino_data = load_sinogram_csv(tmp_csv)
            x_cm, y_cm, bp, fbp, info = reconstruct_arrays(sino_data, resolution=resolution, filter_name=filter_name)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Scan Views", f"{info['projection_count']} angles")
            m2.metric("Detector Samples", f"{info['detector_samples']}")
            m3.metric("Field of View", f"{info['detector_width_cm']:.1f} cm")
            m4.metric("Filter", info["filter_name"])

            xy_extent = [float(x_cm[0]), float(x_cm[-1]), float(y_cm[0]), float(y_cm[-1])]
            c1, c2 = st.columns(2)
            with c1:
                fig1 = plot_tomo_grid(bp, "1. Simple Backprojection (Blurred)", "x (cm)", "y (cm)", xy_extent)
                st.pyplot(fig1)
                plt.close(fig1)
            with c2:
                fig2 = plot_tomo_grid(fbp, f"2. Filtered Backprojection ({filter_name})", "x (cm)", "y (cm)", xy_extent)
                st.pyplot(fig2)
                plt.close(fig2)
        except Exception as exc:
            st.error(f"Could not reconstruct CSV: {exc}")
    else:
        st.info("👈 Upload a `sinogram_attenuation.csv` file generated by TOMOSCOPE in the left sidebar.")


# --- 4-STAGE PROCESS CARDS (2x2 GRID) ---
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("How TOMOSCOPE Works: The 4-Stage Process")

row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.markdown("""
    <div class="stage-card">
        <div class="stage-num">Stage 1</div>
        <h4>Geometry</h4>
        <p>Describe every parallel light-ray path through the phantom using a normal scan angle θ and signed perpendicular distance s from the origin.</p>
        <div class="formula-box">x cos θ + y sin θ = s</div>
    </div>
    """, unsafe_allow_html=True)

with row1_col2:
    st.markdown("""
    <div class="stage-card">
        <div class="stage-num">Stage 2</div>
        <h4>Measurement</h4>
        <p>Record the transmitted light intensity I leaving the sample at each angle and compare it against the unobstructed reference brightness I₀ (with optional dark-frame subtraction).</p>
        <div class="formula-box">I = I₀ e^(−μL)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns(2)
with row2_col1:
    st.markdown("""
    <div class="stage-card">
        <div class="stage-num">Stage 3</div>
        <h4>Projection</h4>
        <p>Convert the exponential light loss into a linear line measurement p(θ, s) via the Radon transform, forming wavy sinusoidal tracks in a sinogram.</p>
        <div class="formula-box">p(θ, s) = ln(I₀ / I)</div>
    </div>
    """, unsafe_allow_html=True)

with row2_col2:
    st.markdown("""
    <div class="stage-card">
        <div class="stage-num">Stage 4</div>
        <h4>Reconstruction</h4>
        <p>Solve the inverse problem by frequency-filtering (Hann, Ram-Lak, or Shepp-Logan) and backprojecting measurements from all scan angles to estimate μ̂(x, y).</p>
        <div class="formula-box">μ̂(x, y) ≈ ∫₀^π q(θ, s) dθ</div>
    </div>
    """, unsafe_allow_html=True)
