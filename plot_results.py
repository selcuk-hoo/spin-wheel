import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter

_BASE = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(_BASE, *parts)

def _load_cod_from_file():
    """Read cod_data.txt written by C++ at each element entry.

    Returns (s_arr, cod_x_mm, cod_y_mm) sorted by s, or (None, None, None).
    Each of the nFODO*8 lattice positions appears once per turn; values are
    averaged over all turns to give the closed-orbit deviation.
    """
    cod_path = _p("cod_data.txt")
    if not os.path.exists(cod_path):
        return None, None, None
    try:
        cd = np.loadtxt(cod_path, skiprows=1)
        if cd.ndim == 1:
            cd = cd.reshape(1, -1)
        if len(cd) == 0:
            return None, None, None
    except (ValueError, OSError):
        return None, None, None

    s_raw  = cd[:, 0]
    x_vals = cd[:, 1]  # mm
    y_vals = cd[:, 2]  # mm

    # s values repeat identically each turn (computed from lattice geometry).
    # Round to suppress floating-point noise, then group.
    s_rounded = np.round(s_raw, 4)
    unique_s  = np.unique(s_rounded)
    cod_x = np.array([x_vals[s_rounded == s].mean() for s in unique_s])
    cod_y = np.array([y_vals[s_rounded == s].mean() for s in unique_s])
    print(f"[COD: {len(unique_s)} örgü konumu, {len(s_raw)//len(unique_s)} tur ortalaması]")
    return unique_s, cod_x, cod_y

def _save_rf_plot(params):
    """Save RF phase-space diagram to rf.png (only if rf.txt exists)."""
    rf_path = _p("rf.txt")
    if not os.path.exists(rf_path):
        return
    try:
        rf_data = np.loadtxt(rf_path, skiprows=1)
        if rf_data.ndim == 1:
            rf_data = rf_data.reshape(1, -1)
        if rf_data.shape[0] == 0:
            return
    except (ValueError, OSError):
        return

    fig_rf, ax_rf = plt.subplots(figsize=(6, 5))
    nc = rf_data.shape[1]
    if nc >= 7:
        psi_wrap  = rf_data[:, 4]
        dp_over_p = rf_data[:, 6]
        psi_deg   = (psi_wrap * 180.0 / np.pi + 180) % 360 - 180
        ax_rf.plot(psi_deg, dp_over_p * 1e3, "ko", markersize=4)
        ax_rf.set_xlabel("Ψ (sarılı, derece)")
        ax_rf.set_ylabel("dp/p ($10^{-3}$)")
    elif nc >= 3:
        phi_rf    = rf_data[:, 1]
        dp_over_p = rf_data[:, 2]
        phi_deg   = (phi_rf * 180.0 / np.pi + 180) % 360 - 180
        ax_rf.plot(phi_deg, dp_over_p * 1e3, "ko", markersize=4)
        ax_rf.set_xlabel("Φ_RF (derece)")
        ax_rf.set_ylabel("dp/p ($10^{-3}$)")
    ax_rf.set_title("RF Faz Diyagramı (Ψ vs dp/p)")
    ax_rf.grid(True, linestyle='--', alpha=0.6)
    fig_rf.tight_layout()
    fig_rf.savefig(_p("rf.png"), dpi=150)
    plt.close(fig_rf)
    print("RF faz diyagramı 'rf.png' olarak kaydedildi.")

def main():
    sim_path = _p("simulation_data.txt")
    if not os.path.exists(sim_path):
        print("HATA: 'simulation_data.txt' bulunamadı.")
        return

    data = np.loadtxt(sim_path, skiprows=1)
    t_sec  = data[:, 0]
    t      = t_sec * 1e6      # μs
    x      = data[:, 1] * 1000  # mm
    y      = data[:, 2] * 1000  # mm
    z_long = data[:, 3]        # cumulative arc length (m)
    sx     = data[:, 7]
    sy     = data[:, 8]
    sz     = data[:, 9]

    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0       = params.get("R0", 95.49)
    nFODO    = params.get("nFODO", 24)
    quadLen  = params.get("quadLen", 0.4)
    driftLen = params.get("driftLen", 2.0833)

    arc_len       = np.pi * R0 / nFODO
    circumference = nFODO * (2 * arc_len + 4 * driftLen + 2 * quadLen)

    # COD from exact lattice-entry positions written by C++ integrator
    cod_s, cod_x, cod_y = _load_cod_from_file()

    # Poincaré data
    x_pc = xp_pc = y_pc = yp_pc = np.array([])
    if os.path.exists(_p("poincare_data.txt")):
        try:
            pc_data = np.loadtxt(_p("poincare_data.txt"), skiprows=1)
            if pc_data.ndim == 1:
                pc_data = pc_data.reshape(1, -1)
            if len(pc_data) > 0:
                print(f"[{len(pc_data)} adet Poincaré noktası çiziliyor]")
                pz_pc = pc_data[:, 5]
                x_pc  = pc_data[:, 0] * 1000
                y_pc  = pc_data[:, 1] * 1000
                xp_pc = (pc_data[:, 3] / pz_pc) * 1000
                yp_pc = (pc_data[:, 4] / pz_pc) * 1000
        except (ValueError, OSError):
            pass

    # RF plot → separate file
    _save_rf_plot(params)

    # ---- Main 3×3 figure ----
    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('6D Spin-Wheel Simülasyon Sonuçları', fontsize=16, fontweight='bold')

    # ---- Row 1: radial x ----
    axs[0, 0].plot(t, x, 'k-', lw=0.8)
    axs[0, 0].set_title("Radyal Konum (x-t)")
    axs[0, 0].set_xlabel("Zaman (μs)")
    axs[0, 0].set_ylabel("x (mm)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    if cod_s is not None:
        axs[0, 1].plot(cod_s, cod_x, 'b-', lw=1.5)
    axs[0, 1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axs[0, 1].set_title("Kapalı Yörünge Bozulması — COD x")
    axs[0, 1].set_xlabel("s (m)")
    axs[0, 1].set_ylabel("⟨x⟩ (mm)")
    axs[0, 1].set_xlim(0, circumference)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    if len(x_pc) > 1:
        axs[0, 2].plot(x_pc, xp_pc, 'ko', markersize=3)
        vx  = np.var(x_pc); vxp = np.var(xp_pc)
        eps = 2 * np.sqrt(max(0, vx * vxp - np.cov(x_pc, xp_pc)[0, 1]**2))
        axs[0, 2].text(0.05, 0.95, f"$\\epsilon_x = {eps:.1e}$ $\\pi$·mm·mrad",
                       transform=axs[0, 2].transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[0, 2].plot(x, (data[:, 4] / data[:, 6]) * 1000, 'k.', markersize=1)
    axs[0, 2].set_title("Yatay Faz Uzayı (x–x')")
    axs[0, 2].set_xlabel("x (mm)")
    axs[0, 2].set_ylabel("x' (mrad)")
    axs[0, 2].grid(True, linestyle='--', alpha=0.5)

    # ---- Row 2: vertical y ----
    axs[1, 0].plot(t, y, 'k-', lw=0.8)
    axs[1, 0].set_title("Dikey Konum (y-t)")
    axs[1, 0].set_xlabel("Zaman (μs)")
    axs[1, 0].set_ylabel("y (mm)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    if cod_s is not None:
        axs[1, 1].plot(cod_s, cod_y, 'b-', lw=1.5)
    axs[1, 1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axs[1, 1].set_title("Kapalı Yörünge Bozulması — COD y")
    axs[1, 1].set_xlabel("s (m)")
    axs[1, 1].set_ylabel("⟨y⟩ (mm)")
    axs[1, 1].set_xlim(0, circumference)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    if len(y_pc) > 1:
        axs[1, 2].plot(y_pc, yp_pc, 'ko', markersize=3)
        vy  = np.var(y_pc); vyp = np.var(yp_pc)
        eps = 2 * np.sqrt(max(0, vy * vyp - np.cov(y_pc, yp_pc)[0, 1]**2))
        axs[1, 2].text(0.05, 0.95, f"$\\epsilon_y = {eps:.1e}$ $\\pi$·mm·mrad",
                       transform=axs[1, 2].transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[1, 2].plot(y, (data[:, 5] / data[:, 6]) * 1000, 'k.', markersize=1)
    axs[1, 2].set_title("Dikey Faz Uzayı (y–y')")
    axs[1, 2].set_xlabel("y (mm)")
    axs[1, 2].set_ylabel("y' (mrad)")
    axs[1, 2].grid(True, linestyle='--', alpha=0.5)

    # ---- Row 3: spin ----
    sg_win = (len(sx) // 4) * 2 + 1
    if sg_win < 5:
        sg_win = 5

    def _spin_panel(ax, signal, ylabel):
        ax.plot(t, signal, 'k-', lw=0.8, alpha=0.4, label='Ham')
        if sg_win >= 5:
            filt = savgol_filter(signal, window_length=sg_win, polyorder=1)
            ax.plot(t, filt, 'r-', lw=1.5, label='Filtrelenmiş')
            trim = int(len(filt) * 0.1)
            if trim > 0 and len(filt) - 2 * trim > 10:
                ft = data[trim:-trim, 0]; fs = filt[trim:-trim]
            else:
                ft = data[:, 0]; fs = filt
            slope, _ = np.polyfit(ft, fs, 1)
            ax.text(0.05, 0.05, f"Eğim: {slope:.2e} rad/s",
                    transform=ax.transAxes, fontsize=9, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel("Zaman (μs)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)

    _spin_panel(axs[2, 0], sx, "$S_x$")
    axs[2, 0].set_title("Radyal Spin ($S_x$-t)")

    _spin_panel(axs[2, 1], sy, "$S_y$")
    axs[2, 1].set_title("Dikey Spin ($S_y$-t)")

    axs[2, 2].plot(t, sz, 'k-', lw=0.8)
    axs[2, 2].set_title("Longitudinal Spin ($S_z$-t)")
    axs[2, 2].set_xlabel("Zaman (μs)")
    axs[2, 2].set_ylabel("$S_z$")
    axs[2, 2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(_p("simulasyon_sonuclari.png"), dpi=150)
    print("Grafik 'simulasyon_sonuclari.png' olarak kaydedildi!")

if __name__ == "__main__":
    main()
