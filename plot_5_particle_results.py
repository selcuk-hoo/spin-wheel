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


def _fft_peak(t_sec, signal, f_center=1000.0, f_window=500.0):
    """Ana FFT tepesini Hanning + parabolik interpolasyonla döndürür."""
    dt   = np.mean(np.diff(t_sec))
    N    = len(signal)
    freq = np.fft.rfftfreq(N, d=dt)
    amp  = np.abs(np.fft.rfft((signal - signal.mean()) * np.hanning(N))) / (N * 0.5)
    df   = freq[1] - freq[0]
    win  = (freq >= max(1.0, f_center - f_window)) & (freq <= f_center + f_window)
    if not win.any():
        return None, None
    win_idx  = np.where(win)[0]
    amp_w    = amp[win_idx]
    k        = win_idx[np.argmax(amp_w)]
    if 0 < k < len(amp) - 1:
        denom = amp[k-1] - 2*amp[k] + amp[k+1]
        dk    = 0.5 * (amp[k-1] - amp[k+1]) / denom if abs(denom) > 1e-30 else 0.0
    else:
        dk = 0.0
    return freq[k] + dk * df, amp[k]


def main():
    # --- Veri yükleme ---
    particles = []
    for i in range(5):
        fname = _p(f"particle_{i}.txt")
        if not os.path.exists(fname):
            print(f"HATA: {fname} bulunamadı. Önce run_5_particles.py çalıştırın.")
            return
        particles.append(np.loadtxt(fname, skiprows=1))

    t_sec = particles[0][:, 0]
    t_ms  = t_sec * 1e3
    sy    = [p[:, 8] for p in particles]   # S_Dikey sütunu

    with open(_p("params.json")) as f:
        params = json.load(f)
    theta0 = params.get("theta0", params.get("theta0_hor", 1e-5))

    # --- 5 spin kombinasyonu ---
    combos = [
        (r"$s_{1y} - s_{0y}$",
         sy[1] - sy[0]),
        (r"$\frac{1}{2}(s_{1y}+s_{2y}) - s_{0y}$",
         0.5*(sy[1]+sy[2]) - sy[0]),
        (r"$\frac{1}{2}(s_{1y}+s_{3y}) - s_{0y}$",
         0.5*(sy[1]+sy[3]) - sy[0]),
        (r"$\frac{1}{2}(s_{1y}+s_{4y}) - s_{0y}$",
         0.5*(sy[1]+sy[4]) - sy[0]),
        (r"$\frac{1}{4}(s_{1y}+s_{2y}+s_{3y}+s_{4y}) - s_{0y}$",
         0.25*(sy[1]+sy[2]+sy[3]+sy[4]) - sy[0]),
    ]

    sg_win = (len(t_sec) // 4) * 2 + 1
    if sg_win < 5:
        sg_win = 5

    fig, axs = plt.subplots(5, 1, figsize=(13, 18), sharex=True)
    fig.suptitle(
        f"Betatron Spin Kombinasyonları  (θ₀ = {theta0:.2e} rad)",
        fontsize=14, fontweight='bold'
    )

    print("=" * 60)
    print(f"BETATRON SPİN KOMBİNASYON ANALİZİ  (θ₀={theta0:.2e} rad)")
    print("=" * 60)

    for ax, (label, signal) in zip(axs, combos):
        ax.plot(t_ms, signal, 'k-', lw=0.6, alpha=0.35, label='Ham')

        if sg_win >= 5:
            filt  = savgol_filter(signal, window_length=sg_win, polyorder=1)
            ax.plot(t_ms, filt, 'r-', lw=1.5, label='Filtrelenmiş')
            trim  = int(len(filt) * 0.1)
            ft    = t_sec[trim:-trim] if trim > 0 else t_sec
            fs    = filt[trim:-trim]  if trim > 0 else filt
            slope, _ = np.polyfit(ft, fs, 1)
            ax.text(0.98, 0.05, f"Eğim: {slope:.2e} /s",
                    transform=ax.transAxes, fontsize=9,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        f_peak, a_peak = _fft_peak(t_sec, signal)
        info = f"  {label}"
        if f_peak is not None:
            info += f"  →  tepe: {f_peak:.1f} Hz,  genlik: {a_peak:.3e}"
        print(info)

        ax.set_ylabel(label, fontsize=10)
        ax.axhline(0, color='gray', lw=0.8, linestyle=':')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')

    print("=" * 60)
    axs[-1].set_xlabel("Zaman (ms)")
    plt.tight_layout()
    out = _p("betatron_spin.png")
    plt.savefig(out, dpi=150)
    print(f"Grafik '{out}' olarak kaydedildi.")


if __name__ == "__main__":
    main()
