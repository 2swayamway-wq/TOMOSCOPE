import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import radon, iradon

# --- PAGE CONFIG & DARK SCIENTIFIC STYLING ---
st.set_page_config(page_title="TOMOSCOPE", page_icon="🔬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1e293b; }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricValue"] { color: #38bdf8; }
    h1, h2, h3 { color: #f8fafc !important; }
</style>
""", unsafe_allow_html=True)

plt.style.use("dark_background")

# --- HEADER ---
st.title("🔬 TOMOSCOPE: 2D Optical Tomography")
st.caption("By Shanmukh Teja & Sai Priyam Patro • Visible-Light Parallel-Beam Radon Transform & Reconstruction")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Source")
mode = st.sidebar.radio("Choose Mode", ["Demo Simulation", "Upload Real Scan Photos"])

ref_photo = None
scan_photos = []
phantom_type = "Multi-Region Phantom"
num_views = 36

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


def make_plot(data, title, xlabel, ylabel, extent, aspect="equal"):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#0b0f19")
    im = ax.imshow(data, cmap="gray", origin="lower", extent=extent, aspect=aspect)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#f8fafc", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color="#94a3b8")
    ax.set_ylabel(ylabel, fontsize=9, color="#94a3b8")
    ax.tick_params(colors="#64748b", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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

    c1, c2, c3 = st.columns(3)
    with c1:
        fig1 = make_plot(phantom, "1. Known Phantom μ(x, y)", "x", "y", [-1, 1, -1, 1])
        st.pyplot(fig1)
        plt.close(fig1)
    with c2:
        fig2 = make_plot(sinogram, "2. Sinogram p(θ, s)", "Angle θ (degrees)", "Detector s", [0, 180, -1, 1], aspect="auto")
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

# --- MATH REFERENCE EXPANDER ---
with st.expander("📐 Mathematical Pipeline Behind TOMOSCOPE"):
    st.markdown("""
    1. **Beer–Lambert Attenuation:** Light intensity drops exponentially through absorbing regions, so the line integral projection is calculated from transmitted intensity $I$ and blank reference intensity $I_0$ as $p(\\theta, s) = \\ln(I_0 / I)$.
    2. **Radon Transform & Sinogram:** A single absorbing feature at $(x_0, y_0)$ traces a sinusoidal curve $s(\\theta) = x_0\\cos\\theta + y_0\\sin\\theta$ across scan angles from $0^\\circ$ to $180^\\circ$.
    3. **Filtered Backprojection:** Applying a ramp filter in the frequency domain removes the $1/r$ blurring caused by raw backprojection before smearing measurements back across the 2D grid.
    """)
