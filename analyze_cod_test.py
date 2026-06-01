"""
test_misalignment.py'den gelen cod_test_results.npz'yi okuyup
COD profilini ve bump yapısını detaylı analiz eder.
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

d = np.load("cod_test_results.npz")
cod_cw  = d["cod_cw"]    # (193, 3): s_m, x_mm, y_mm
cod_ccw = d["cod_ccw"]
s_qf    = float(d["s_phys_qf"])

L_def = (2*np.pi*95.49)/(2*24)
cell_len = 2*L_def + 4*2.0833 + 2*0.4

fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

axs[0].plot(cod_cw[:,0],  cod_cw[:,2],  'b-o', ms=3, lw=1.0, label='CW (dir=-1)')
axs[0].plot(cod_ccw[:,0], cod_ccw[:,2], 'r-x', ms=4, lw=1.0, label='CCW (dir=+1)')
axs[0].axvline(s_qf, color='k', ls='--', alpha=0.6, label=f'Fiz. QF_8 @ s={s_qf:.1f} m')
for k in range(25):
    axs[0].axvline(k*cell_len, color='gray', alpha=0.15)
axs[0].set_ylabel('y_vert (mm)')
axs[0].set_title(f'COD dikey profili — B0hor=1e-4 T misalignment, nFODO_off=8')
axs[0].legend()
axs[0].grid(True, alpha=0.4)

axs[1].plot(cod_cw[:,0],  cod_cw[:,1],  'b-o', ms=3, lw=1.0, label='CW')
axs[1].plot(cod_ccw[:,0], cod_ccw[:,1], 'r-x', ms=4, lw=1.0, label='CCW')
axs[1].axvline(s_qf, color='k', ls='--', alpha=0.6)
for k in range(25):
    axs[1].axvline(k*cell_len, color='gray', alpha=0.15)
axs[1].set_ylabel('x_radial (mm)')
axs[1].set_xlabel('s (m)')
axs[1].legend()
axs[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("cod_misalignment.png", dpi=120)
print(f"Grafik: cod_misalignment.png")

# Hesaplama: QF_8 civarındaki değerler
idx_qf = 8 * 8 + 2  # cell 8, elem 2 (QF)
print(f"\nQF_8 indeksi: {idx_qf}, s={cod_cw[idx_qf,0]:.3f} m")
print(f"  CW : x={cod_cw[idx_qf,1]:+.4f} mm,  y={cod_cw[idx_qf,2]:+.4f} mm")
print(f"  CCW: x={cod_ccw[idx_qf,1]:+.4f} mm,  y={cod_ccw[idx_qf,2]:+.4f} mm")

# RMS karşılaştırma
rms_cw  = np.sqrt(np.mean(cod_cw[:,2]**2))
rms_ccw = np.sqrt(np.mean(cod_ccw[:,2]**2))
print(f"\nRMS y_vert dikey:  CW={rms_cw:.4f} mm,  CCW={rms_ccw:.4f} mm,  oran CCW/CW={rms_ccw/rms_cw:.2f}")
print(f"Max y_vert dikey:  CW={np.max(np.abs(cod_cw[:,2])):.4f} mm,  CCW={np.max(np.abs(cod_ccw[:,2])):.4f} mm")
