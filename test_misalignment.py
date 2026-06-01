"""
Hizalama hatası (B0hor) tek bir quad'a verildiğinde COD bump pozisyonu testi.
CW ve CCW için aynı fiziksel quad'a kick uygulanmalı; bump aynı yerde olmalı.
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

NFODO_OFF = 8        # 8. fiziksel quad'a misalignment
B0HOR     = 1e-4     # Tesla → quadYOffset üretir (DİKEY kick yapar)

t0, t_end, h = 0.0, 0.5e-3, 1e-11
return_steps  = 1000

def make_fields(direction):
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
    f.E0ver     = 0.0       # spin presesyonu test için gereksiz
    f.B0hor     = B0HOR     # misalignment kick
    f.nFODO_off = float(NFODO_OFF)
    f.N_particles = 0.0
    return f

def run(direction):
    fields = make_fields(direction)
    s_local = [0.0, 0.0, float(direction)]
    y0 = [0.0, 0.0, 0.0, 0.0, 0.0, p_mag * float(direction)] + s_local
    integrate_particle(y0, t0, t_end, h, fields=fields, return_steps=return_steps)
    # cod_data.txt okunur
    cod = np.loadtxt("cod_data.txt", skiprows=1)
    return cod

print("=" * 70)
print(f"Misalignment testi: B0hor={B0HOR} T, fiziksel cell {NFODO_OFF} QF'sinde")
print("=" * 70)

# Önce eski cod sil
if os.path.exists("cod_data.txt"):
    os.remove("cod_data.txt")
print("\n--- CW (direction=-1) ---")
cod_cw = run(-1)

if os.path.exists("cod_data.txt"):
    os.remove("cod_data.txt")
print("--- CCW (direction=+1) ---")
cod_ccw = run(+1)

# cod_data.txt formatı: s_m, x_mm, y_mm
# B0hor kuad'a quadYOffset verir → DİKEY (y) kick → COD'da y bump bekleniyor
print(f"\nCOD CW şekli: {cod_cw.shape}")
print(f"  max |x_radial|={np.max(np.abs(cod_cw[:,1])):.4f} mm,  max |y_vert|={np.max(np.abs(cod_cw[:,2])):.4f} mm")
print(f"COD CCW şekli: {cod_ccw.shape}")
print(f"  max |x_radial|={np.max(np.abs(cod_ccw[:,1])):.4f} mm,  max |y_vert|={np.max(np.abs(cod_ccw[:,2])):.4f} mm")

# Kick noktasındaki COD: cell NFODO_OFF, elem 2 (QF) → indeks = NFODO_OFF*8 + 2
i_qf8     = NFODO_OFF * 8 + 2
y_cw_qf8  = cod_cw[i_qf8, 2]
y_ccw_qf8 = cod_ccw[i_qf8, 2]
max_cw    = np.max(np.abs(cod_cw[:,2]))
max_ccw   = np.max(np.abs(cod_ccw[:,2]))

# Bekleneni hesapla:
L_def    = (2*np.pi*R0) / (2*24)
cell_len = 2*L_def + 4*2.0833 + 2*0.4
s_phys_qf = NFODO_OFF * cell_len + L_def + 2.0833
print(f"\nFiziksel QF_{NFODO_OFF} pozisyonu: s = {s_phys_qf:.3f} m")
print(f"Ring çevresi: {24*cell_len:.3f} m")
print(f"Kick noktası (idx={i_qf8}): CW y={y_cw_qf8:+.4f} mm,  CCW y={y_ccw_qf8:+.4f} mm")
print(f"Max |y_vert|: CW={max_cw:.4f} mm,  CCW={max_ccw:.4f} mm")

# Karşılaştırma: COD fiziksel varlığı ve işaret simetrisi
ok_cw_amp   = max_cw   > 0.1          # CW'de anlamlı COD var
ok_ccw_amp  = max_ccw  > 0.1          # CCW'de anlamlı COD var
ok_cw_kick  = abs(y_cw_qf8)  > 0.05  # CW kick noktasında COD sıfır değil
ok_ccw_kick = abs(y_ccw_qf8) > 0.05  # CCW kick noktasında COD sıfır değil
ok_sign     = (y_cw_qf8 * y_ccw_qf8) < 0  # CW/CCW işaretleri zıt (QF/QD rol değişimi)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
print(f"\nCW  COD genliği > 0.1 mm?              → {PASS if ok_cw_amp else FAIL}")
print(f"CCW COD genliği > 0.1 mm?              → {PASS if ok_ccw_amp else FAIL}")
print(f"CW  kick noktasında COD > 0.05 mm?     → {PASS if ok_cw_kick else FAIL}")
print(f"CCW kick noktasında COD > 0.05 mm?     → {PASS if ok_ccw_kick else FAIL}")
print(f"CW/CCW kick noktasında işaret zıt mı?  → {PASS if ok_sign else FAIL}")
print("=" * 70)

# COD profilini dosyaya kaydet
np.savez("cod_test_results.npz", cod_cw=cod_cw, cod_ccw=cod_ccw, s_phys_qf=s_phys_qf)
print("Sonuç: cod_test_results.npz")
