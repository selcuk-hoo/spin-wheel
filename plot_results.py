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


def _estimate_tune(u, up, nFODO, poincare_quad_index):
    uc  = u  - u.mean()
    upc = up - up.mean()
    if np.std(uc) < 1e-12 or np.std(upc) < 1e-12:
        return None
    dphi     = np.diff(np.unwrap(np.arctan2(upc, uc)))
    avg_dphi = abs(np.mean(dphi))
    if poincare_quad_index < 0:
        return nFODO * avg_dphi / (2.0 * np.pi)
    return avg_dphi / (2.0 * np.pi)


def _load_cod(n_per_turn):
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
    print(f"[COD: {len(cd)} örgü elemanı okundu]")
    return cd[:, 0], cd[:, 1], cd[:, 2]


def _save_rf_plot(params):
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

    data  = np.loadtxt(sim_path, skiprows=1)
    t_sec = data[:, 0]
    t     = t_sec * 1e6        # μs
    x     = data[:, 1] * 1000  # mm
    y     = data[:, 2] * 1000  # mm
    sx    = data[:, 7]
    sy    = data[:, 8]
    sz    = data[:, 9]

    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0             = params.get("R0", 95.49)
    nFODO          = params.get("nFODO", 24)
    quadLen        = params.get("quadLen", 0.4)
    driftLen       = params.get("driftLen", 2.0833)
    pq_idx         = params.get("poincare_quad_index", -1)
    base_spin_freq = params.get("base_spin_freq", 0.0)

    arc_len       = np.pi * R0 / nFODO
    circumference = nFODO * (2 * arc_len + 4 * driftLen + 2 * quadLen)
    n_per_turn    = nFODO * 8

    # ---- Poincaré data & tune estimation ----
    x_pc = xp_pc = y_pc = yp_pc = np.array([])
    Qx = Qy = None
    if os.path.exists(_p("poincare_data.txt")):
        try:
            pc_data = np.loadtxt(_p("poincare_data.txt"), skiprows=1)
            if pc_data.ndim == 1:
                pc_data = pc_data.reshape(1, -1)
            if len(pc_data) > 4:
                print(f"[{len(pc_data)} adet Poincaré noktası çiziliyor]")
                pz_pc = pc_data[:, 5]
                x_pc  = pc_data[:, 0] * 1000
                y_pc  = pc_data[:, 1] * 1000
                xp_pc = (pc_data[:, 3] / pz_pc) * 1000
                yp_pc = (pc_data[:, 4] / pz_pc) * 1000
                Qx = _estimate_tune(x_pc, xp_pc, nFODO, pq_idx)
                Qy = _estimate_tune(y_pc, yp_pc, nFODO, pq_idx)
                if Qx is not None and Qy is not None:
                    print(f"[Tune: Qx={Qx:.4f}  Qy={Qy:.4f}]")
        except (ValueError, OSError):
            pass

    # ---- COD extraction ----
    cod_s, cod_x, cod_y = _load_cod(n_per_turn)

    # ---- RF plot → separate file ----
    _save_rf_plot(params)

    # ======== Main 3×4 figure ========
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle('6D Spin-Wheel Simülasyon Sonuçları', fontsize=16, fontweight='bold')

    def _plot_fft(ax, t_seconds, signal_mm, title):
        dt = np.mean(np.diff(t_seconds))
        freq = np.fft.rfftfreq(len(signal_mm), d=dt)
        amp = np.abs(np.fft.rfft(signal_mm - np.mean(signal_mm))) / len(signal_mm)
        mask = (freq > 0.0) & (amp > 0.0)
        if mask.any():
            ax.plot(freq[mask], amp[mask], 'k-', lw=1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.text(0.5, 0.5, "Sinyal yok", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Frekans (Hz)")
        ax.set_ylabel("Genlik (mm)")
        ax.grid(True, linestyle='--', alpha=0.5)

    # ---- Row 1: radial x ----
    axs[0, 0].plot(t, x, 'k-', lw=0.8)
    axs[0, 0].set_title("Radyal Konum (x-t)")
    axs[0, 0].set_xlabel("Zaman (μs)")
    axs[0, 0].set_ylabel("x (mm)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    if cod_s is not None:
        lbl = f"Qx={Qx:.3f}" if Qx is not None else "tur ort."
        axs[0, 1].plot(cod_s, cod_x, 'b-', lw=1.5, label=lbl)
        rms_x = np.sqrt(np.mean(cod_x**2))
        axs[0, 1].text(0.97, 0.97,
                       f"RMS = {rms_x*1e3:.2f} μm\nTop = {np.sum(cod_x)*1e3:.2f} μm",
                       transform=axs[0, 1].transAxes, fontsize=8, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        axs[0, 1].legend(fontsize=8)
    axs[0, 1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axs[0, 1].set_title("Kapalı Yörünge — COD x")
    axs[0, 1].set_xlabel("s (m)")
    axs[0, 1].set_ylabel("$x_{CO}$ (mm)")
    axs[0, 1].set_xlim(0, circumference)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    if len(x_pc) > 1:
        if pq_idx < 0:
            plot_x_pc = x_pc[::int(nFODO)]
            plot_xp_pc = xp_pc[::int(nFODO)]
        else:
            plot_x_pc = x_pc
            plot_xp_pc = xp_pc
        axs[0, 2].plot(plot_x_pc, plot_xp_pc, 'ko', markersize=3)
        vx  = np.var(plot_x_pc); vxp = np.var(plot_xp_pc)
        eps = 2 * np.sqrt(max(0, vx * vxp - np.cov(plot_x_pc, plot_xp_pc)[0, 1]**2))
        axs[0, 2].text(0.05, 0.95, f"$\\epsilon_x = {eps:.1e}$ $\\pi$·mm·mrad",
                       transform=axs[0, 2].transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[0, 2].plot(x, (data[:, 4] / data[:, 6]) * 1000, 'k.', markersize=1)
    axs[0, 2].set_title("Yatay Faz Uzayı (x–x')")
    axs[0, 2].set_xlabel("x (mm)")
    axs[0, 2].set_ylabel("x' (mrad)")
    axs[0, 2].grid(True, linestyle='--', alpha=0.5)
    _plot_fft(axs[0, 3], t_sec, x, "x(t) FFT")

    # ---- Row 2: vertical y ----
    axs[1, 0].plot(t, y, 'k-', lw=0.8)
    axs[1, 0].set_title("Dikey Konum (y-t)")
    axs[1, 0].set_xlabel("Zaman (μs)")
    axs[1, 0].set_ylabel("y (mm)")
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    if cod_s is not None:
        lbl = f"Qy={Qy:.3f}" if Qy is not None else "tur ort."
        axs[1, 1].plot(cod_s, cod_y, 'b-', lw=1.5, label=lbl)
        rms_y = np.sqrt(np.mean(cod_y**2))
        axs[1, 1].text(0.97, 0.97,
                       f"RMS = {rms_y*1e3:.2f} μm\nTop = {np.sum(cod_y)*1e3:.2f} μm",
                       transform=axs[1, 1].transAxes, fontsize=8, va='top', ha='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        axs[1, 1].legend(fontsize=8)
    axs[1, 1].axhline(0, color='gray', lw=0.8, linestyle='--')
    axs[1, 1].set_title("Kapalı Yörünge — COD y")
    axs[1, 1].set_xlabel("s (m)")
    axs[1, 1].set_ylabel("$y_{CO}$ (mm)")
    axs[1, 1].set_xlim(0, circumference)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    if len(y_pc) > 1:
        if pq_idx < 0:
            plot_y_pc = y_pc[::int(nFODO)]
            plot_yp_pc = yp_pc[::int(nFODO)]
        else:
            plot_y_pc = y_pc
            plot_yp_pc = yp_pc
        axs[1, 2].plot(plot_y_pc, plot_yp_pc, 'ko', markersize=3)
        vy  = np.var(plot_y_pc); vyp = np.var(plot_yp_pc)
        eps = 2 * np.sqrt(max(0, vy * vyp - np.cov(plot_y_pc, plot_yp_pc)[0, 1]**2))
        axs[1, 2].text(0.05, 0.95, f"$\\epsilon_y = {eps:.1e}$ $\\pi$·mm·mrad",
                       transform=axs[1, 2].transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        axs[1, 2].plot(y, (data[:, 5] / data[:, 6]) * 1000, 'k.', markersize=1)
    axs[1, 2].set_title("Dikey Faz Uzayı (y–y')")
    axs[1, 2].set_xlabel("y (mm)")
    axs[1, 2].set_ylabel("y' (mrad)")
    axs[1, 2].grid(True, linestyle='--', alpha=0.5)
    _plot_fft(axs[1, 3], t_sec, y, "y(t) FFT")

    # ---- Row 3: spin ----
    sg_win = (len(sx) // 4) * 2 + 1
    if sg_win < 5:
        sg_win = 5

    def _spin_panel(ax, signal, ylabel):
        ax.plot(t, signal, 'k-', lw=0.8, alpha=0.4, label='Ham')
        if sg_win >= 5:
            filt  = savgol_filter(signal, window_length=sg_win, polyorder=1)
            ax.plot(t, filt, 'r-', lw=1.5, label='Filtrelenmiş')
            trim  = int(len(filt) * 0.1)
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

    def _sy_freq_analysis(ax, signal, ylabel):
        """
        S_y frekans analizi — IQ demodülasyon ile Δf ölçümü.

        base_spin_freq > 0: S_y ~f_base'de salınır (C++ lab çerçevesinde).
        IQ demodülasyonu ile Δf = f_gerçek − f_base ölçülür:
          I(t) = S_y × cos(2π f_base t)  →  LPF
          Q(t) = S_y × sin(2π f_base t)  →  LPF
          phase = atan2(Q, I) = 2π Δf t + φ₀
          Δf = d(phase)/dt / (2π)
          f_ölçülen = f_base + Δf

        LPF penceresi: null @ 2×f_base → IQ karışım artefaktını bastırır.
        Float64 hassasiyeti: ~nanohertz (C++ dönen çerçeve gerekmez).
        """
        ax.plot(t, signal, 'k-', lw=0.8, alpha=0.4, label='Ham')

        if base_spin_freq <= 0:
            _spin_panel(ax, signal, ylabel)
            return

        try:
            from scipy.ndimage import uniform_filter1d

            dt = t_sec[1] - t_sec[0] if len(t_sec) > 1 else 1e-6

            # IQ demodülasyonu: f_base referans sinyaliyle çarp
            ref_cos = np.cos(2 * np.pi * base_spin_freq * t_sec)
            ref_sin = np.sin(2 * np.pi * base_spin_freq * t_sec)
            I_raw = signal * ref_cos
            Q_raw = signal * ref_sin

            # LPF: kutu filtre, null @ 1/(win_lp * dt) ≈ f_base
            # Bu, 2×f_base'deki IQ karışım terimini bastırır.
            win_lp = max(5, int(round(1.0 / (base_spin_freq * dt))))
            I_filt = uniform_filter1d(I_raw, size=win_lp)
            Q_filt = uniform_filter1d(Q_raw, size=win_lp)

            trim = win_lp
            if len(I_filt) - 2 * trim < 10:
                trim = max(1, len(I_filt) // 10)
            t_trim  = t_sec[trim:-trim]
            I_trim  = I_filt[trim:-trim]
            Q_trim  = Q_filt[trim:-trim]

            amplitude = 2 * np.mean(np.sqrt(I_trim**2 + Q_trim**2))

            # Anlık faz: atan2(Q, I) → 2π Δf t + φ₀
            phase = np.unwrap(np.arctan2(Q_trim, I_trim))
            slope, _ = np.polyfit(t_trim, phase, 1)
            delta_f   = slope / (2 * np.pi)
            f_measured = base_spin_freq + delta_f

            print("-" * 50)
            print("S_y FREKANS ANALiZi (Python IQ demodülasyon):")
            print(f"-> Taban Frekans (Base)  : {base_spin_freq:.16f} Hz")
            print(f"-> Δf (IQ faz eğimi)     : {delta_f:+.16f} Hz")
            print(f"-> Ölçülen Frekans       : {f_measured:.16f} Hz")
            print(f"-> Genlik                : {amplitude:.4e}")
            print("-" * 50)

            scale  = np.max(np.abs(signal)) * 0.9 if np.max(np.abs(signal)) > 0 else 1.0
            amp_iq = np.max(np.abs(I_trim)) if np.max(np.abs(I_trim)) > 0 else 1.0
            ax.plot(t[trim:-trim], I_trim / amp_iq * scale,
                    'g-', lw=1.5, label='I(t) demod', alpha=0.8)
            ax.text(
                0.05, 0.05,
                f"Base    : {base_spin_freq:.10f} Hz\n"
                f"Δf      : {delta_f:+.10f} Hz\n"
                f"Ölçülen : {f_measured:.10f} Hz\n"
                f"Genlik  : {amplitude:.3e}",
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

        except Exception as e:
            print(f"S_y analiz hatası: {e}")
            _spin_panel(ax, signal, ylabel)
            return

        ax.legend(fontsize=8, loc='upper right')
        ax.set_xlabel("Zaman (μs)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)

    _sy_freq_analysis(axs[2, 1], sy, "$S_y$")
    axs[2, 1].set_title("Dikey Spin ($S_y$-t)")

    axs[2, 2].plot(t, sz, 'k-', lw=0.8)
    axs[2, 2].set_title("Longitudinal Spin ($S_z$-t)")
    axs[2, 2].set_xlabel("Zaman (μs)")
    axs[2, 2].set_ylabel("$S_z$")
    axs[2, 2].grid(True, linestyle='--', alpha=0.5)
    _plot_fft(axs[2, 3], t_sec, sy, "S_y(t) FFT")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(_p("simulasyon_sonuclari.png"), dpi=150)
    print("Grafik 'simulasyon_sonuclari.png' olarak kaydedildi!")


if __name__ == "__main__":
    main()
