import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import radon, iradon

# --- PAGE CONFIG ---
st.set_page_config(page_title="TOMOSCOPE | 2D Optical CT", page_icon="🔬", layout="wide")

# --- CLEAN LIGHT-MODE SCIENTIFIC STYLING ---
st.markdown("""
<style>
    /* Force bright, crisp Light Mode across entire app */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    /* Ensure all text, labels, captions, and upload boxes are dark & legible */
    label, p, span, li, div[data-testid="stWidgetLabel"] p,
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stCaptionContainer"] p {
        color: #1e293b !important;
    }

    h1, h2, h3, h4 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* File uploader box in Light Mode */
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

    /* Dropdown selectbox & popup menu */
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

    /* Top badge */
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

    /* Metric cards */
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

    /* 4-Stage Process Cards */
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
st.markdown('<div class="tomo-badge">2D OPTICAL COMPUTED TOMOGRAPHY • COMPETITION MODEL</div>', unsafe_allow_html=True)
st.title("🔬 TOMOSCOPE: Visible-Light Tomography Simulator")
st.caption("By Shanmukh Teja & Sai Priyam Patro • Parallel-Beam Radon Transform & Filtered Backprojection")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Source")
mode = st.sidebar.radio("Choose Mode", ["Demo Simulation", "Upload Real Scan Photos"])

ref_photo = None
scan_photos = []
phantom_type = "Multi-Region Phantom"
num_views = 36
inspect_angle = 45

if mode == "Demo Simulation":
    phantom_type = st.sidebar.selectbox(
        "Test Phantom",
        ["Multi-Region Phantom", "Single Off-Center Point", "Hollow Ring"]
    )
    num_views = st.sidebar.select_slider(
        "Projection Views (over 180°)",
        options=[6, 9, 18, 36, 72],
        value=36
    )
    inspect_angle = st.sidebar.slider(
        "Inspect Light Ray Angle θ (degrees)",
        min_value=0,
        max_value=179,
        value=45
    )
else:
    ref_photo = st.sidebar.file_uploader(
        "Blank Reference Photo (I₀)", type=["jpg", "jpeg", "png"]
    )
    scan_photos = st.sidebar.file_uploader(
        "Angle Scan Photos (I) in rotation order",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

st.sidebar.header("2. Reconstruction Math")
use_filter = st.sidebar.toggle("Ramp Filter (Filtered Backprojection)", value=True)


# =====================================================================
#  DROP-IN ZONE: PASTE YOUR PARTNER'S PYCHARM CAMERA CODE HERE LATER
# =====================================================================
def process_camera_photos(ref_file, scan_files, use_filter):
    """
    Inputs:
      - ref_file: uploaded blank-reference photo (I_0)
      - scan_files: list of uploaded scan photos (I)
      - use_filter: True for Filtered Backprojection, False for Unfiltered
    Outputs:
      - sinogram: 2D numpy array p(theta, s)
      - reconstruction: 2D numpy array of reconstructed attenuation map
    """
    # --- TEMPORARY PLACEHOLDER UNTIL PARTNER SENDS PYCHARM CODE ---
    n_angles = len(scan_files)
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)

    sample = np.array(Image.open(scan_files[0]).convert("L"))
    size = min(sample.shape[0], sample.shape[1], 128)
    temp_grid = np.zeros((size, size))
    temp_grid[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0

    sinogram = radon(temp_grid, theta=theta, circle=True)
    filter_name = "ramp" if use_filter else None
    reconstruction = iradon(sinogram, theta=theta, filter_name=filter_name, circle=True)
    # --- END OF PLACEHOLDER ---

    return sinogram, reconstruction


def run_demo_simulation(phantom_name, num_angles, use_filter):
    N = 128
    y, x = np.ogrid[-1:1:N*1j, -1:1:N*1j]
    phantom = np.zeros((N, N))

    if phantom_name == "Multi-Region Phantom":
        phantom[x**2 + y**2 <= 0.85**2] = 0.15
        phantom[((x + 0.35)/0.18)**2 + ((y - 0.25)/0.28)**2 <= 1] = 0.75
        phantom[((x - 0.30)/0.20)**2 + ((y - 0.30)/0.15)**2 <= 1] = 0.55
        phantom[((x + 0.25)/0.13)**2 + ((y + 0.30)/0.13)**2 <= 1] = 0.50
        phantom[((x - 0.30)/0.15)**2 + ((y + 0.30)/0.22)**2 <= 1] = 0.95
    elif phantom_name == "Single Off-Center Point":
        phantom[(x - 0.35)**2 + (y - 0.45)**2 <= 0.10**2] = 1.0
    else:
        phantom[(x**2 + y**2 <= 0.75**2) & (x**2 + y**2 >= 0.45**2)] = 0.85

    theta = np.linspace(0.0, 180.0, num_angles, endpoint=False)
    sinogram = radon(phantom, theta=theta, circle=True)
    filter_name = "ramp" if use_filter else None
    reconstruction = iradon(sinogram, theta=theta, filter_name=filter_name, circle=True)
    return phantom, sinogram, reconstruction
# =====================================================================


def make_plot(data, title, xlabel, ylabel, extent, aspect="equal", ray_angle=None, sino_angle=None):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(data, cmap="gray", origin="lower", extent=extent, aspect=aspect)

    # Overlay parallel light rays on phantom
    if ray_angle is not None:
        rad = np.deg2rad(ray_angle)
        nx, ny = np.cos(rad), np.sin(rad)
        dx, dy = -np.sin(rad), np.cos(rad)
        for s_val in [-0.5, 0.0, 0.5]:
            t = np.linspace(-1.2, 1.2, 50)
            rx = s_val * nx + t * dx
            ry = s_val * ny + t * dy
            lw = 2.0 if s_val == 0.0 else 1.0
            alpha = 0.95 if s_val == 0.0 else 0.55
            ax.plot(rx, ry, color="#0284c7", linewidth=lw, alpha=alpha)
        ax.annotate("", xy=(0.45 * nx, 0.45 * ny), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#d97706", lw=2.2))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    # Highlight inspected angle column on sinogram
    if sino_angle is not None:
        ax.axvline(sino_angle, color="#0284c7", linestyle="--", linewidth=1.8, alpha=0.95)

    ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#334155")
    ax.set_ylabel(ylabel, fontsize=9, color="#334155")
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#cbd5e1")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#475569", labelsize=8)
    return fig


# --- MAIN DASHBOARD ---
if mode == "Demo Simulation":
    phantom, sinogram, recon = run_demo_simulation(phantom_type, num_views, use_filter)

    p_norm = (phantom - phantom.min()) / (phantom.max() - phantom.min() + 1e-8)
    r_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    rmse = np.sqrt(np.mean((p_norm - r_norm) ** 2))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scan Views", f"{num_views} angles")
    m2.metric("Angular Step", f"{180 / num_views:.1f}°")
    m3.metric("Image RMSE", f"{rmse:.4f}")
    m4.metric("Filter", "Ramp (FBP)" if use_filter else "Unfiltered")

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fig1 = make_plot(phantom, "1. Known Phantom μ(x, y)", "x", "y", [-1, 1, -1, 1], ray_angle=inspect_angle)
        st.pyplot(fig1)
        plt.close(fig1)
    with c2:
        fig2 = make_plot(sinogram, "2. Sinogram p(θ, s)", "Angle θ (degrees)", "Detector position s",
                         [0, 180, -1, 1], aspect="auto", sino_angle=inspect_angle)
        st.pyplot(fig2)
        plt.close(fig2)
    with c3:
        label = "3. Filtered Backprojection" if use_filter else "3. Unfiltered Backprojection"
        fig3 = make_plot(recon, label, "x", "y", [-1, 1, -1, 1])
        st.pyplot(fig3)
        plt.close(fig3)

else:
    if ref_photo is not None and len(scan_photos) >= 2:
        sinogram, recon = process_camera_photos(ref_photo, scan_photos, use_filter)
        st.success(f"Loaded reference photo (I₀) and {len(scan_photos)} scan views.")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = make_plot(sinogram, "1. Measured Sinogram p(θ, s)", "Angle θ (degrees)", "Detector s", [0, 180, -1, 1], aspect="auto")
            st.pyplot(fig1)
            plt.close(fig1)
        with c2:
            label = "2. Filtered Backprojection" if use_filter else "2. Unfiltered Backprojection"
            fig2 = make_plot(recon, label, "x", "y", [-1, 1, -1, 1])
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.info("Upload your **Blank Reference Photo (I₀)** and at least **2 Angle Scan Photos (I)** in the left sidebar.")

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
        <p>Record the transmitted light intensity I leaving the sample at each angle and compare it against the unobstructed reference brightness I₀.</p>
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
        <p>Solve the inverse problem by ramp-filtering and backprojecting measurements from all scan angles to estimate the 2D attenuation map μ̂(x, y).</p>
        <div class="formula-box">μ̂(x, y) ≈ ∫₀^π q(θ, s) dθ</div>
    </div>
    """, unsafe_allow_html=True)
