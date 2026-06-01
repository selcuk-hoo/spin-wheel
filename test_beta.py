"""
Beta fonksiyon testi: CW ve CCW için dikey/yatay betaları Poincaré ölçümüyle karşılaştırır.

Beklenti:
- CW (dir=-1): QF yatayda (x) odaklayıcı → β_x(QF) > β_x(QD)
                QF dikeyde (z) defocusing → β_z(QF) < β_z(QD)
- CCW (dir=+1): v×B tersine döner → QF ve QD rolleri swap olur:
                QF artık dikeyde odaklayıcı → β_z(QF) > β_z(QD)
                QF artık yatayda defocusing → β_x(QF) < β_x(QD)

Yöntem: küçük başlangıç bozulmalarıyla 10+ tur atılır, Poincaré
noktalarındaki sapma genliğinden betalar hesaplanır.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from integrator import integrate_particle, FieldParams

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

# Devir süresi = 2πR0/v = 2π*95.49/(beta0*c)
T_rev = 2 * np.pi * R0 / (beta0 * C)
# 20 devir simüle edelim
t_end = 20 * T_rev
h     = 1e-11
return_steps = 2000

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def make_fields(direction, poincare_quad=-1):
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
    f.N_particles = 0.0
    f.poincare_quad_index = float(poincare_quad)
    return f

def run_with_kicks(direction, dx_mm=0.0, dz_mm=0.0, poincare_quad=-1):
    """Küçük başlangıç sapmasıyla parçacık izle."""
    fields = make_fields(direction, poincare_quad=poincare_quad)
    y0 = [dx_mm*1e-3, dz_mm*1e-3, 0.0,
          0.0, 0.0, p_mag * float(direction),
          0.0, 0.0, float(direction)]
    hist, poincare, _ = integrate_particle(y0, 0.0, t_end, h,
                                            fields=fields, return_steps=return_steps)
    return hist, poincare

def rms_amplitude(arr):
    """RMS genlik: sapmalar sıfır-ortalamalı kabul edilir."""
    return np.sqrt(np.mean((arr - arr.mean())**2)) * np.sqrt(2)

print("=" * 65)
print("BETA FONKSİYON TESTİ — CW vs CCW")
print("=" * 65)
print(f"Devir süresi T_rev = {T_rev*1e6:.3f} µs,  t_end = {t_end*1e3:.2f} ms")
print()

results = {}
for dir_label, direction in [("CW (dir=-1)", -1), ("CCW (dir=+1)", +1)]:
    print(f"--- {dir_label} ---")

    # poincare_quad_index=0 → hücre 0 ARC1 başına (CW'de QD çıkışı, CCW'de QF çıkışı)
    # poincare_quad_index=1 → hücre 0 ARC2 başına (CW'de QF çıkışı, CCW'de QD çıkışı)
    # Her iki yönde: pq=1 konumu x-odaklayıcı quad çıkışı → β_x büyük
    #                pq=0 konumu z-odaklayıcı quad çıkışı → β_z büyük

    amp_x = {}
    amp_z = {}
    for quad_name, pq_idx in [("QD", 0), ("QF", 1)]:
        # Radyal kick
        _, poc_x = run_with_kicks(direction, dx_mm=1.0, dz_mm=0.0, poincare_quad=pq_idx)
        # Dikey kick
        _, poc_z = run_with_kicks(direction, dx_mm=0.0, dz_mm=1.0, poincare_quad=pq_idx)

        if poc_x is not None and len(poc_x) > 4:
            # Poincaré lokal koord: kolon 0 = radyal sapma (R−R0), kolon 1 = dikey (z)
            x_dev = (poc_x[:, 0] - R0) * 1e3  # mm  (ortalama kayması RMS'yi etkilemez)
            z_dev  = poc_z[:, 1] * 1e3          # mm  (dikey konum, kolon 2 = yay uzunluğu!)
            ax = rms_amplitude(x_dev)
            az = rms_amplitude(z_dev)
        else:
            ax = az = 0.0

        amp_x[quad_name] = ax
        amp_z[quad_name] = az
        print(f"  {quad_name} (pq={pq_idx}): x genlik={ax:.3f} mm,  z genlik={az:.3f} mm")

    results[direction] = (amp_x, amp_z)

    # β orantılı: β ~ A²/ε (aynı emittans → β ~ A²)
    ratio_x = amp_x["QF"] / (amp_x["QD"] + 1e-30)
    ratio_z = amp_z["QD"] / (amp_z["QF"] + 1e-30)
    print(f"  Oran QF/QD x: {ratio_x:.3f}  QD/QF z: {ratio_z:.3f}")
    print()

print("=" * 65)
print("SONUÇ")
print("=" * 65)

amp_x_cw, amp_z_cw = results[-1]
amp_x_ccw, amp_z_ccw = results[+1]

# pq=1 (QF konumu) x-odaklayıcı quad çıkışı → β_x büyük → amp_x[QF] > amp_x[QD]
ok_cw_x   = amp_x_cw["QF"]  > amp_x_cw["QD"]
ok_ccw_x  = amp_x_ccw["QF"] > amp_x_ccw["QD"]
# pq=0 (QD konumu) z-odaklayıcı quad çıkışı → β_z büyük → amp_z[QD] > amp_z[QF]
ok_cw_z   = amp_z_cw["QD"]  > amp_z_cw["QF"]
ok_ccw_z  = amp_z_ccw["QD"] > amp_z_ccw["QF"]
# CW/CCW simetrik olmalı: x oran farkı < %20
ratio_cw  = amp_x_cw["QF"]  / (amp_x_cw["QD"]  + 1e-30)
ratio_ccw = amp_x_ccw["QF"] / (amp_x_ccw["QD"] + 1e-30)
ok_sym = abs(ratio_cw - ratio_ccw) / ((ratio_cw + ratio_ccw) / 2 + 1e-30) < 0.2

print(f"  CW  pq=1 β_x > pq=0 β_x (x-oda. quad çıkışı)?  {PASS if ok_cw_x else FAIL}")
print(f"  CCW pq=1 β_x > pq=0 β_x (x-oda. quad çıkışı)?  {PASS if ok_ccw_x else FAIL}")
print(f"  CW  pq=0 β_z > pq=1 β_z (z-oda. quad çıkışı)?  {PASS if ok_cw_z else FAIL}")
print(f"  CCW pq=0 β_z > pq=1 β_z (z-oda. quad çıkışı)?  {PASS if ok_ccw_z else FAIL}")
print(f"  CW/CCW β simetrik mi? (oran farkı <%%20)?         {PASS if ok_sym else FAIL}  "
      f"CW={ratio_cw:.3f} CCW={ratio_ccw:.3f}")

all_pass = ok_cw_x and ok_cw_z and ok_ccw_x and ok_ccw_z and ok_sym
print()
if all_pass:
    print(">>> Tüm testler GEÇTI. FODO beta fonksiyonu CW ve CCW için doğru.")
else:
    print(">>> Bazı testler BAŞARISIZ. Detayları kontrol edin.")
print("=" * 65)
