"""
direction=-1 vs +1 simetri testleri.

Test 1 — SC kapalı: iki yönde spin normu ve frekans aynı olmalı.
Test 2 — SC açık:   S_y drift'i iki yönde ZIT işaretli olmalı
          (B_sc = beta_beam × E_sc / c, beam yönü ters → B_sc ters).
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from integrator import integrate_particle, FieldParams

# --- Sabitler ---
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
eRatio  = 1.0
E0_V_m  = -eRatio * (p_magic * (p_magic / np.sqrt(p_magic**2 + M2**2)) / R0) * 1e9

t0, t_end, h = 0.0, 5e-3, 1e-11
return_steps  = 5000

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def make_fields(direction, N_particles=0.0):
    f = FieldParams()
    f.R0        = R0
    f.E0        = E0_V_m
    f.E0_power  = 1.0
    f.quadK1    = 0.6
    f.quadK0    = 0.6
    f.quadSwitch = 1.0
    f.sextSwitch = 0.0
    f.EDMSwitch  = 0.0
    f.direction  = float(direction)
    f.nFODO     = 24.0
    f.quadLen   = 0.4
    f.driftLen  = 2.0833
    f.E0ver     = 1e4
    f.N_particles   = N_particles
    f.beam_radius_a = 0.01
    return f

def run(direction, N_particles=0.0):
    fields = make_fields(direction, N_particles)
    s_local = [0.0, 0.0, float(direction)]
    y0 = [0.0, 0.0, 0.0,
          0.0, 0.0, p_mag * float(direction)] + s_local
    hist, _, _ = integrate_particle(y0, t0, t_end, h,
                                     fields=fields, return_steps=return_steps)
    return hist

def fft_peak(t_sec, sig, f_center=1000.0, bw=600.0):
    dt   = np.mean(np.diff(t_sec))
    N    = len(sig)
    freq = np.fft.rfftfreq(N, d=dt)
    win  = np.hanning(N)
    amp  = np.abs(np.fft.rfft((sig - sig.mean()) * win)) / (N * 0.5)
    mask = (freq >= f_center - bw) & (freq <= f_center + bw)
    if not mask.any():
        return None, None
    idx  = np.where(mask)[0]
    k    = idx[np.argmax(amp[idx])]
    df   = freq[1] - freq[0]
    denom = amp[k-1] - 2*amp[k] + amp[k+1] if 0 < k < len(amp)-1 else 1e-30
    dk    = 0.5*(amp[k-1] - amp[k+1]) / denom if abs(denom) > 1e-30 else 0.0
    return freq[k] + dk*df, amp[k]

def lin_slope(t, sig, trim=0.05):
    n = len(t)
    i0, i1 = int(n*trim), int(n*(1-trim))
    c = np.polyfit(t[i0:i1], sig[i0:i1], 1)
    return c[0]

print("=" * 65)
print("TEST 1  —  SC=0, dir=-1 vs dir=+1")
print("=" * 65)

h1m = run(-1)
h1p = run(+1)

t_sec = np.linspace(t0, t_end, return_steps)
sy_m = h1m[:, 8]
sy_p = h1p[:, 8]

# Spin normu
norm_m = (h1m[:, 6]**2 + h1m[:, 7]**2 + h1m[:, 8]**2)
norm_p = (h1p[:, 6]**2 + h1p[:, 7]**2 + h1p[:, 8]**2)
ok_norm_m = abs(norm_m.mean() - 1.0) < 1e-5 and norm_m.std() < 1e-5
ok_norm_p = abs(norm_p.mean() - 1.0) < 1e-5 and norm_p.std() < 1e-5
print(f"  Spin normu dir=-1:  ort={norm_m.mean():.7f}  std={norm_m.std():.2e}  → {PASS if ok_norm_m else FAIL}")
print(f"  Spin normu dir=+1:  ort={norm_p.mean():.7f}  std={norm_p.std():.2e}  → {PASS if ok_norm_p else FAIL}")

# Frekans aynı mı?
fm, am = fft_peak(t_sec, sy_m)
fp, ap = fft_peak(t_sec, sy_p)
ok_freq = abs(fm - fp) < 2.0 if (fm and fp) else False
print(f"  FFT freq dir=-1: {fm:.1f} Hz   dir=+1: {fp:.1f} Hz   fark={abs(fm-fp) if fm and fp else '?':.1f} Hz  → {PASS if ok_freq else FAIL}")

# Radyal orbit: X - R0 yakın 0 olmalı
x_m = h1m[:, 0] - R0
x_p = h1p[:, 0] - R0
ok_orb_m = abs(x_m.mean()) < 1e-4 and x_m.std() < 5e-3
ok_orb_p = abs(x_p.mean()) < 1e-4 and x_p.std() < 5e-3
print(f"  Radyal sapma dir=-1: ort={x_m.mean()*1e3:.3f} mm  std={x_m.std()*1e3:.3f} mm  → {PASS if ok_orb_m else FAIL}")
print(f"  Radyal sapma dir=+1: ort={x_p.mean()*1e3:.3f} mm  std={x_p.std()*1e3:.3f} mm  → {PASS if ok_orb_p else FAIL}")

print()
print("=" * 65)
print("TEST 2  —  SC açık (N=1e8), dir=-1 vs dir=+1")
print("İdeal: S_y slope işareti ZIT olmalı (B_sc = β×E/c yön bağımlı)")
print("=" * 65)

h2m = run(-1, N_particles=1e8)
h2p = run(+1, N_particles=1e8)

sy2_m = h2m[:, 8]
sy2_p = h2p[:, 8]

slope_m = lin_slope(t_sec, sy2_m)
slope_p = lin_slope(t_sec, sy2_p)

print(f"  S_y slope dir=-1: {slope_m:+.4e} rad/s")
print(f"  S_y slope dir=+1: {slope_p:+.4e} rad/s")

signs_opposite = (slope_m * slope_p) < 0
magnitudes_similar = abs(abs(slope_m) - abs(slope_p)) / (abs(slope_m) + abs(slope_p) + 1e-30) < 0.5
print(f"  İşaretler zıt mı?    {PASS if signs_opposite else FAIL}  (beklenen: EVET)")
print(f"  Büyüklükler benzer?  {PASS if magnitudes_similar else FAIL}  |Δ|/|Σ|={abs(abs(slope_m)-abs(slope_p))/(abs(slope_m)+abs(slope_p)+1e-30):.2f}")

print()
print("=" * 65)
print("TEST 3  —  SC açık, orbit ve spin normu bozulmuyor mu?")
print("=" * 65)
norm2_m = (h2m[:, 6]**2 + h2m[:, 7]**2 + h2m[:, 8]**2)
norm2_p = (h2p[:, 6]**2 + h2p[:, 7]**2 + h2p[:, 8]**2)
ok3m = abs(norm2_m.mean() - 1.0) < 1e-5
ok3p = abs(norm2_p.mean() - 1.0) < 1e-5
print(f"  Spin normu SC dir=-1: {norm2_m.mean():.7f}  → {PASS if ok3m else FAIL}")
print(f"  Spin normu SC dir=+1: {norm2_p.mean():.7f}  → {PASS if ok3p else FAIL}")

print()
if signs_opposite:
    print(">>> B_sc işaret testini GEÇTI. Kod SC için zaten doğru.")
else:
    print(">>> B_sc işaret testini GEÇEMEDİ. integrator.cpp'de B_sc *= dir düzeltmesi gerekiyor.")

print("=" * 65)
