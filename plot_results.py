import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter, find_peaks

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
    t     = t_sec * 1e3        # ms
    x     = data[:, 1] * 1000  # mm
    y     = data[:, 2] * 1000  # mm
    sx    = data[:, 7]   # S_Rady (radyal spin, ~sabit)
    sy    = data[:, 8]   # S_Dikey (dikey spin, -sin(Omega*t))
    sz    = data[:, 9]   # S_Long (boylamsal spin, -cos(Omega*t))

    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0             = params.get("R0", 95.49)
    nFODO          = params.get("nFODO", 24)
    quadLen        = params.get("quadLen", 0.4)
    driftLen       = params.get("driftLen", 2.0833)
    pq_idx         = params.get("poincare_quad_index", -1)
    simulate_ideal = int(params.get("simulate_ideal", 0))

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
        ax.set_ylabel("Genlik")
        ax.grid(True, linestyle='--', alpha=0.5)

    def _sy_fft_peaks(t_seconds, signal, f_center=1000.0, f_window=500.0):
        """S_y FFT'sinden ana tepe (~1 kHz) ve side band'leri bulup yazdırır.

        Tepe frekansları parabolik interpolasyonla sub-bin hassasiyete taşınır:
          δk = (a[k-1] - a[k+1]) / (2*(a[k-1] - 2*a[k] + a[k+1]))
          f_hassas = freq[k] + δk * df
        """
        dt = np.mean(np.diff(t_seconds))
        N  = len(signal)
        freq = np.fft.rfftfreq(N, d=dt)
        # Hanning penceresi: sinc yan loblarını bastırır, parabolik interpolasyon
        # hatası dikdörtgen pencerenin ~1/18'ine düşer (~0.004*df)
        window = np.hanning(N)
        amp  = np.abs(np.fft.rfft((signal - np.mean(signal)) * window)) / (N * 0.5)
        df   = freq[1] - freq[0]

        def _parabolic(amp_full, k):
            """Bin k etrafında parabolik interpolasyon; hassas frekans ofsetini döndürür."""
            if k <= 0 or k >= len(amp_full) - 1:
                return 0.0
            a0, a1, a2 = amp_full[k - 1], amp_full[k], amp_full[k + 1]
            denom = a0 - 2 * a1 + a2
            if abs(denom) < 1e-30:
                return 0.0
            return 0.5 * (a0 - a2) / denom  # δk cinsinden ofset

        # Ana arama penceresi: f_center ± f_window Hz
        win = (freq >= max(1.0, f_center - f_window)) & (freq <= f_center + f_window)
        if not win.any():
            return
        win_indices = np.where(win)[0]   # penceredeki global bin indeksleri
        amp_w  = amp[win_indices]
        freq_w = freq[win_indices]

        # Ana tepe (penceredeki maksimum), parabolik interpolasyonla hassaslaştır
        local_main = np.argmax(amp_w)
        global_main = win_indices[local_main]
        dk_main  = _parabolic(amp, global_main)
        f_main   = freq[global_main] + dk_main * df
        a_main   = amp_w[local_main]

        # Side band tespiti: prominans eşiği ana tepenin %1'i, min ayırım 5 Hz
        min_dist = max(1, int(5.0 / df))
        min_prom = a_main * 0.01
        peaks_local, _ = find_peaks(amp_w, prominence=min_prom, distance=min_dist)

        print("-" * 62)
        print(f"S_y FFT TEPE ANALİZİ  (pencere: {f_center-f_window:.0f}–{f_center+f_window:.0f} Hz)")
        acc = 0.004 * df  # Hanning + parabolik interpolasyon hassasiyeti
        print(f"  FFT bin genişliği   : {df:.1f} Hz  (T={1/df:.4f} s, Hanning penceresi)")
        print(f"  Hassasiyet          : ~±{acc:.1f} Hz")
        print(f"  Ana tepe            : {f_main:.1f} Hz   genlik={a_main:.4e}")

        if len(peaks_local) > 1:
            print(f"  Side band'ler ({len(peaks_local)-1} adet):")
            for li in peaks_local:
                gi = win_indices[li]
                dk = _parabolic(amp, gi)
                f_i = freq[gi] + dk * df
                a_i = amp_w[li]
                if abs(f_i - f_main) < df * 0.5:
                    continue
                print(f"    {f_i:.1f} Hz   genlik={a_i:.4e}   Δf={f_i-f_main:+.1f} Hz")
        else:
            print("  Side band tespit edilmedi.")
        print("-" * 62)

    # ---- Row 1: radial x ----
    axs[0, 0].plot(t, x, 'k-', lw=0.8)
    axs[0, 0].set_title("Radyal Konum (x-t)")
    axs[0, 0].set_xlabel("Zaman (ms)")
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
    axs[1, 0].set_xlabel("Zaman (ms)")
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
        ax.set_xlabel("Zaman (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.5)

    _spin_panel(axs[2, 0], sx, "$S_x$ (radyal)")
    axs[2, 0].set_title("Radyal Spin ($S_x$-t)")

    # axs[2,1]: simulate_ideal=1 ise ΔS_y, değilse ham S_y
    sy_ideal = None
    if simulate_ideal and os.path.exists(_p("simulation_data_ideal.txt")):
        try:
            ideal_data = np.loadtxt(_p("simulation_data_ideal.txt"), skiprows=1)
            if ideal_data.shape[0] == len(sy):
                sy_ideal = ideal_data[:, 8]
        except (ValueError, OSError):
            print("Uyarı: simulation_data_ideal.txt okunamadı.")

    if sy_ideal is not None:
        delta_sy = sy - sy_ideal
        _spin_panel(axs[2, 1], delta_sy, "$\\Delta S_y$")
        axs[2, 1].set_title("$\\Delta S_y$ = Ana $-$ İdeal")
    else:
        _spin_panel(axs[2, 1], sy, "$S_y$")
        axs[2, 1].set_title("Dikey Spin ($S_y$-t)")

    _spin_panel(axs[2, 2], sz, "$S_z$ (boylamsal)")
    axs[2, 2].set_title("Boylamsal Spin ($S_z$-t)")
    axs[2, 2].set_xlabel("Zaman (ms)")
    axs[2, 2].set_ylabel("$S_z$")
    axs[2, 2].grid(True, linestyle='--', alpha=0.5)

    _plot_fft(axs[2, 3], t_sec, sy, "$S_y$(t) FFT")
    _sy_fft_peaks(t_sec, sy)

    # Spin korunumu kontrolü: |S|² = Sx²+Sy²+Sz² her an 1 olmalı
    s_norm_sq = sx**2 + sy**2 + sz**2
    print(f"-> Spin normu |S|²: min={s_norm_sq.min():.6f}  max={s_norm_sq.max():.6f}  (ideal=1.000000)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(_p("simulasyon_sonuclari.png"), dpi=150)
    print("Grafik 'simulasyon_sonuclari.png' olarak kaydedildi!")


if __name__ == "__main__":
    main()
