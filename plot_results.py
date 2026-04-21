import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

_BASE = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(_BASE, *parts)

def main():
    sim_path = _p("simulation_data.txt")
    if not os.path.exists(sim_path):
        print("HATA: 'simulation_data.txt' bulunamadı.")
        return

    data = np.loadtxt(sim_path, skiprows=1)
    t_sec = data[:, 0]
    t     = t_sec * 1e6   # μs
    x     = data[:, 1] * 1000  # mm
    y     = data[:, 2] * 1000  # mm
    sx    = data[:, 7]
    sy    = data[:, 8]
    sz    = data[:, 9]

    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0        = params.get("R0", 95.49)
    nFODO     = params.get("nFODO", 24)
    quadLen   = params.get("quadLen", 0.4)
    driftLen  = params.get("driftLen", 2.0833)
    dt_sim    = params.get("dt", 1e-11)
    t_end     = params.get("t2", 0.001)
    n_ret     = params.get("return_steps", 10000)

    # Ring circumference and revolution period
    arc_len      = np.pi * R0 / nFODO
    cell_len     = 2 * arc_len + 4 * driftLen + 2 * quadLen
    circumference = nFODO * cell_len
    M2  = 0.938272046
    AMU = 1.792847356
    p_m = M2 / np.sqrt(AMU)
    E_t = np.sqrt(p_m**2 + M2**2)
    beta0 = p_m / E_t
    T_rev = circumference / (beta0 * 299792458.0)

    # Effective sample period and COD window (≈ 1 revolution)
    total_steps  = int(t_end / dt_sim)
    save_interval = max(1, int(total_steps / n_ret))
    dt_eff = save_interval * dt_sim
    cod_window = max(3, int(round(T_rev / dt_eff)))
    print(f"[COD penceresi: {cod_window} örnek ≈ 1 devir ({T_rev*1e6:.2f} μs)]")

    cod_x = uniform_filter1d(x, size=cod_window)
    cod_y = uniform_filter1d(y, size=cod_window)

    # Poincaré data
    x_pc = xp_pc = y_pc = yp_pc = np.array([])
    if os.path.exists(_p("poincare_data.txt")):
        try:
            pc_data = np.loadtxt(_p("poincare_data.txt"), skiprows=1)
            if pc_data.ndim == 1:
                pc_data = pc_data.reshape(1, -1)
            if len(pc_data) > 0:
                print(f"[{len(pc_data)} adet Poincaré noktası çiziliyor]")
                x_pc  = pc_data[:, 0] * 1000
                y_pc  = pc_data[:, 1] * 1000
                pz_pc = pc_data[:, 5]
                xp_pc = (pc_data[:, 3] / pz_pc) * 1000
                yp_pc = (pc_data[:, 4] / pz_pc) * 1000
        except (ValueError, OSError):
            pass

    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle('6D Spin-Wheel Simülasyon Sonuçları', fontsize=16, fontweight='bold')

    # ---- Row 1: radial x ----
    axs[0, 0].plot(t, x, 'k-', lw=0.8)
    axs[0, 0].set_title("Radyal Konum (x-t)")
    axs[0, 0].set_xlabel("Zaman (μs)")
    axs[0, 0].set_ylabel("x (mm)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    axs[0, 1].plot(t, cod_x, 'b-', lw=1.5)
    axs[0, 1].set_title(f"Kapalı Yörünge Boz. — COD x  (w≈{cod_window})")
    axs[0, 1].set_xlabel("Zaman (μs)")
    axs[0, 1].set_ylabel("⟨x⟩ (mm)")
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

    axs[1, 1].plot(t, cod_y, 'b-', lw=1.5)
    axs[1, 1].set_title(f"Kapalı Yörünge Boz. — COD y  (w≈{cod_window})")
    axs[1, 1].set_xlabel("Zaman (μs)")
    axs[1, 1].set_ylabel("⟨y⟩ (mm)")
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

    def _spin_panel(ax, signal, label, color='k'):
        ax.plot(t, signal, color=color, lw=0.8, alpha=0.4, label='Ham')
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

    _spin_panel(axs[2, 0], sx, 'S_x')
    axs[2, 0].set_title("Radyal Spin ($S_x$-t)")
    axs[2, 0].set_xlabel("Zaman (μs)")
    axs[2, 0].set_ylabel("$S_x$")
    axs[2, 0].grid(True, linestyle='--', alpha=0.5)

    _spin_panel(axs[2, 1], sy, 'S_y')
    axs[2, 1].set_title("Dikey Spin ($S_y$-t)")
    axs[2, 1].set_xlabel("Zaman (μs)")
    axs[2, 1].set_ylabel("$S_y$")
    axs[2, 1].grid(True, linestyle='--', alpha=0.5)

    axs[2, 2].plot(t, sz, 'k-', lw=0.8)
    axs[2, 2].set_title("Longitudinal Spin ($S_z$-t)")
    axs[2, 2].set_xlabel("Zaman (μs)")
    axs[2, 2].set_ylabel("$S_z$")
    axs[2, 2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = _p("simulasyon_sonuclari.png")
    plt.savefig(out_path, dpi=150)
    print(f"Grafik '{out_path}' olarak kaydedildi!")

if __name__ == "__main__":
    main()
