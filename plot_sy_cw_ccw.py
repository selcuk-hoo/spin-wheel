"""
Tek parçacık simülasyonu: CW (dir=-1) ve CCW (dir=+1) için Sy(t) karşılaştırması.

Çizimler:
  1. Sy_CW(t)  ve  Sy_CCW(t)
  2. Fark: ΔSy = Sy_CCW − Sy_CW
  3. FFT: her iki yönde spin frekansı

Kullanım:
  python plot_sy_cw_ccw.py               # varsayılan parametreler
  python plot_sy_cw_ccw.py --edm         # EDM açık (EDM_ETA = 1.88e-15)
  python plot_sy_cw_ccw.py --sc          # uzay yükü açık (N=1e8)
  python plot_sy_cw_ccw.py --t 10        # t_end = 10 ms
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from integrator import integrate_particle, FieldParams

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
M2  = 0.938272046
AMU = 1.792847356
C   = 299792458.0
M1  = 1.672621777e-27

p_magic = M2 / np.sqrt(AMU)
E_tot   = np.sqrt(p_magic**2 + M2**2)
beta0   = p_magic / E_tot
gamma0  = 1.0 / np.sqrt(1.0 - beta0**2)
R0      = 95.49
p_mag   = gamma0 * M1 * C * beta0
E0_V_m  = -(p_magic * (p_magic / np.sqrt(p_magic**2 + M2**2)) / R0) * 1e9

T_rev = 2 * np.pi * R0 / (beta0 * C)

# ---------------------------------------------------------------------------
# Argümanlar
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--edm",   action="store_true", help="EDM açık")
parser.add_argument("--sc",    action="store_true", help="Uzay yükü açık (N=1e8)")
parser.add_argument("--fodo",  action="store_true", help="FODO quad'ları açık")
parser.add_argument("--E0ver", type=float, default=0.0,
                    help="Dikey elektrik alan (V/m), varsayılan=0 (frozen spin için)")
parser.add_argument("--t",   type=float, default=5.0, help="t_end (ms), varsayılan=5")
parser.add_argument("--steps", type=int, default=5000, help="return_steps")
parser.add_argument("--out", type=str, default="sy_cw_ccw.png", help="Çıktı dosyası")
args = parser.parse_args()

t_end        = args.t * 1e-3
return_steps = args.steps
h            = 1e-11

# ---------------------------------------------------------------------------
# Alan parametreleri
# ---------------------------------------------------------------------------
def make_fields(direction):
    f = FieldParams()
    f.R0         = R0
    f.E0         = E0_V_m
    f.E0_power   = 1.0
    f.quadK1     = 0.6
    f.quadK0     = 0.6
    f.quadSwitch = 1.0 if args.fodo else 0.0
    f.sextSwitch = 0.0
    f.direction  = float(direction)
    f.nFODO      = 24.0
    f.quadLen    = 0.4
    f.driftLen   = 2.0833
    f.E0ver      = args.E0ver          # varsayılan 0: frozen spin bozulmaz
    f.EDMSwitch  = 1.0 if args.edm else 0.0
    f.EDM_ETA    = 1.88e-15
    f.N_particles    = 1e8 if args.sc else 0.0
    f.beam_radius_a  = 0.01
    return f

# ---------------------------------------------------------------------------
# Simülasyon
# ---------------------------------------------------------------------------
def run(direction):
    fields = make_fields(direction)
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * float(direction),
          0.0, 0.0, float(direction)]
    hist, _, _ = integrate_particle(y0, 0.0, t_end, h,
                                    fields=fields, return_steps=return_steps)
    return hist

print(f"Parametreler: t_end={args.t} ms,  EDM={'açık' if args.edm else 'kapalı'},"
      f"  SC={'açık' if args.sc else 'kapalı'}")
print(f"T_rev = {T_rev*1e6:.3f} µs,  {t_end/T_rev:.1f} devir")

print("CW  (dir=-1) simüle ediliyor…")
hist_cw  = run(-1)
print("CCW (dir=+1) simüle ediliyor…")
hist_ccw = run(+1)

# Zaman ekseni
t_ms = np.linspace(0, t_end * 1e3, len(hist_cw))

# Koordinat tablosu (integrator.py convert_global_to_local_matrix):
#   hist[:,6] = Sx_global  → radyal spin
#   hist[:,7] = Sz_global  → DİKEY spin  (fizik notasyonunda Sy)
#   hist[:,8] = Sy_global  → BOYLAMSAL spin (Ss; başlangıçta ±1)
sy_vert_cw  = hist_cw[:,  7]   # dikey — EDM sinyali burada
sy_vert_ccw = hist_ccw[:, 7]
ss_long_cw  = hist_cw[:,  8]   # boylamsal — başlangıçta ±1 (doğru)
ss_long_ccw = hist_ccw[:, 8]
diff        = sy_vert_ccw - sy_vert_cw

# Spin normu kontrolü
norm_cw  = np.sqrt(hist_cw[:,6]**2  + hist_cw[:,7]**2  + hist_cw[:,8]**2)
norm_ccw = np.sqrt(hist_ccw[:,6]**2 + hist_ccw[:,7]**2 + hist_ccw[:,8]**2)
print(f"Spin normu  CW : ort={norm_cw.mean():.7f}  std={norm_cw.std():.2e}")
print(f"Spin normu CCW : ort={norm_ccw.mean():.7f}  std={norm_ccw.std():.2e}")
print(f"Boylamsal spin başlangıcı  CW={ss_long_cw[0]:+.4f}  CCW={ss_long_ccw[0]:+.4f}  (±1 → boylamsal ok)")
print(f"Dikey spin başlangıcı      CW={sy_vert_cw[0]:+.4f}  CCW={sy_vert_ccw[0]:+.4f}  (0 → beklenen)")

# ---------------------------------------------------------------------------
# FFT — spin frekansı (boylamsal bileşenden — orada osilasyon en belirgin)
# ---------------------------------------------------------------------------
def fft_peak(sig, dt, f_lo=200.0, f_hi=3000.0):
    N    = len(sig)
    freq = np.fft.rfftfreq(N, d=dt)
    win  = np.hanning(N)
    amp  = np.abs(np.fft.rfft((sig - sig.mean()) * win)) / (N * 0.5)
    mask = (freq >= f_lo) & (freq <= f_hi)
    idx  = np.argmax(amp[mask])
    return freq[mask][idx], amp, freq

dt_s = t_end / len(ss_long_cw)
f_cw,  amp_cw,  freq_cw  = fft_peak(ss_long_cw,  dt_s)
f_ccw, amp_ccw, freq_ccw = fft_peak(ss_long_ccw, dt_s)
print(f"FFT spin frekansı  CW : {f_cw:.2f} Hz")
print(f"FFT spin frekansı CCW : {f_ccw:.2f} Hz")
print(f"Fark: Δf = {abs(f_cw - f_ccw):.2f} Hz")

# ---------------------------------------------------------------------------
# Çizim
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(13, 11))

tag_edm  = " | EDM açık"  if args.edm  else ""
tag_sc   = " | SC açık"  if args.sc   else ""
tag_fodo = " | FODO açık" if args.fodo else " | saf E"
tag_ever = f" | E0ver={args.E0ver:.0e}" if args.E0ver != 0 else ""
fig.suptitle(f"Tek parçacık spin — CW vs CCW{tag_fodo}{tag_edm}{tag_sc}{tag_ever}  "
             f"(t_end={args.t} ms, boylamsal başlangıç)", fontsize=13)

# — Panel 1: Sy dikey(t) — EDM sinyali burada büyür
axes[0].plot(t_ms, sy_vert_cw,  'b',   lw=0.8, label='CW (dir=-1)')
axes[0].plot(t_ms, sy_vert_ccw, 'r--', lw=0.8, label='CCW (dir=+1)')
axes[0].set_ylabel('$S_y$ (dikey)')
axes[0].set_xlabel('t (ms)')
axes[0].set_title('Dikey spin bileşeni — EDM sinyali')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.4)

# — Panel 2: Fark ΔSy dikey —
axes[1].plot(t_ms, diff, 'g', lw=0.8)
axes[1].set_ylabel(r'$\Delta S_y^{\rm vert} = S_y^{\rm CCW} - S_y^{\rm CW}$')
axes[1].set_xlabel('t (ms)')
axes[1].set_title('CW−CCW dikey spin farkı')
axes[1].grid(True, alpha=0.4)
if len(diff) > 10:
    coeffs = np.polyfit(t_ms, diff, 1)
    slope_per_s = coeffs[0] * 1e3
    fit_line = np.polyval(coeffs, t_ms)
    axes[1].plot(t_ms, fit_line, 'k--', lw=1.0,
                 label=f'Lineer fit: slope={slope_per_s:+.3e} /s')
    axes[1].legend(fontsize=9)

# — Panel 3: Boylamsal Ss(t) + FFT — başlangıçta ±1, osilasyon frekansı spin frekansı
ax3a = axes[2]
ax3b = ax3a.twinx()
ax3a.plot(t_ms, ss_long_cw,  'b',   lw=0.6, alpha=0.7, label=f'CW Ss,  f={f_cw:.1f} Hz')
ax3a.plot(t_ms, ss_long_ccw, 'r--', lw=0.6, alpha=0.7, label=f'CCW Ss, f={f_ccw:.1f} Hz')
ax3a.set_ylabel('$S_s$ (boylamsal)')
ax3a.set_xlabel('t (ms)')
ax3a.set_title(f'Boylamsal spin (başlangıç ±1) | spin frekansı CW={f_cw:.2f} Hz  CCW={f_ccw:.2f} Hz  Δf={abs(f_cw-f_ccw):.2f} Hz')
ax3a.legend(fontsize=9, loc='upper left')
ax3a.grid(True, alpha=0.4)

f_mask = freq_cw < 4000
ax3b.plot(freq_cw[f_mask],  amp_cw[f_mask],  'b:',  lw=1.0, alpha=0.5)
ax3b.plot(freq_ccw[f_mask], amp_ccw[f_mask], 'r:',  lw=1.0, alpha=0.5)
ax3b.axvline(f_cw,  color='b', alpha=0.4, ls=':')
ax3b.axvline(f_ccw, color='r', alpha=0.4, ls=':')
ax3b.set_ylabel('FFT genlik', color='gray')
ax3b.tick_params(axis='y', labelcolor='gray')

plt.tight_layout()
plt.savefig(args.out, dpi=130)
print(f"\nGrafik kaydedildi: {args.out}")

np.savez("sy_cw_ccw_data.npz",
         t_ms=t_ms,
         sy_vert_cw=sy_vert_cw, sy_vert_ccw=sy_vert_ccw,
         ss_long_cw=ss_long_cw, ss_long_ccw=ss_long_ccw,
         diff=diff, f_cw=f_cw, f_ccw=f_ccw)
print("Veri kaydedildi:   sy_cw_ccw_data.npz")
