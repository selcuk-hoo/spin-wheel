import json
import time
import threading
import numpy as np
import os
from integrator import integrate_particle, FieldParams


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    with open("params.json") as f:
        config = json.load(f)

    for i in range(5):
        if os.path.isfile(f"particle_{i}.txt"):
            os.remove(f"particle_{i}.txt")

    # --- Fiziksel sabitler ve sihirli momentum ---
    M2  = 0.938272046
    AMU = 1.792847356
    C   = 299792458.0
    M1  = 1.672621777e-27

    p_magic_base = M2 / np.sqrt(AMU)
    p_magic = p_magic_base * (1.0 + config.get("momError", 0.0))
    E_tot  = np.sqrt(p_magic**2 + M2**2)
    beta0  = p_magic / E_tot
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    R0     = config["R0"]
    eRatio = config.get("eRatio", 1.0)
    E0_V_m = -eRatio * (p_magic_base * (p_magic_base / np.sqrt(p_magic_base**2 + M2**2)) / R0) * 1e9

    direction = config.get("direction", -1)
    x0        = config.get("dev0", 0.0)
    y0_vert   = config.get("y0",   0.0)
    p_mag     = gamma0 * M1 * C * beta0

    spinHor = config.get("spinHorRotation", 0.0)
    s_local = [
        np.sin(spinHor * direction),
        0.0,
        np.cos(spinHor * direction) * direction,
    ]

    # --- Alan parametreleri ---
    alanlar = FieldParams()
    alanlar.R0                  = R0
    alanlar.E0                  = E0_V_m
    alanlar.E0_power            = config.get("E0_power", 1.0)
    alanlar.B0ver               = config.get("B0ver", 0.0)
    alanlar.B0rad               = config.get("B0rad", 0.0)
    alanlar.B0long              = config.get("B0long", 0.0)
    alanlar.quadK1              = config.get("k1", 0.0)
    alanlar.quadK0              = config.get("k0", alanlar.quadK1)
    alanlar.sextK1              = config.get("sextK1", 0.0)
    alanlar.quadSwitch          = float(config.get("quadSwitch", 1))
    alanlar.sextSwitch          = float(config.get("sextSwitch", 0))
    alanlar.EDMSwitch           = float(config.get("EDMSwitch", 0))
    alanlar.direction           = float(direction)
    alanlar.nFODO               = float(config.get("nFODO", 24))
    alanlar.quadLen             = float(config.get("quadLen", 0.4))
    alanlar.driftLen            = float(config.get("driftLen", 2.0))
    alanlar.poincare_quad_index = float(config.get("poincare_quad_index", 0))
    alanlar.rfSwitch            = float(config.get("rfSwitch", 0))
    alanlar.rfVoltage           = float(config.get("rfVoltage", 10000.0))
    alanlar.h                   = float(config.get("h", 1.0))
    alanlar.quadModA            = float(config.get("quadModA", 0.0))
    alanlar.quadModF            = float(config.get("quadModF", 0.0))
    alanlar.nFODO_off           = float(config.get("nFODO_off", -1))
    alanlar.B0hor               = float(config.get("B0hor", 0.0))
    alanlar.E0ver               = float(config.get("E0ver", 0.0))
    alanlar.EDM_ETA             = float(config.get("EDM_ETA", 0.0))
    alanlar.N_particles         = float(config.get("N_particles", 0.0))
    alanlar.beam_radius_a       = float(config.get("beam_radius_a", 0.01))
    alanlar.base_spin_freq      = float(config.get("base_spin_freq", 0.0))

    t0           = 0.0
    t_end        = config.get("t2", 1e-5)
    h            = config.get("dt", 1e-11)
    return_steps = config.get("return_steps", 10000)
    adim_sayisi  = int((t_end - t0) / h)
    save_interval = max(1, int(adim_sayisi / return_steps))

    # theta0: betatron açı genliği
    theta0 = config.get("theta0", config.get("theta0_hor", 1e-5))

    # 5 parçacık: (theta_hor, theta_ver)
    angles = [
        ( 0.0,    0.0   ),  # 0: ideal
        (+theta0, +theta0), # 1
        (+theta0, -theta0), # 2
        (-theta0, +theta0), # 3
        (-theta0, -theta0), # 4
    ]

    results = [None] * 5

    def run_particle(idx, theta_hor, theta_ver):
        p0_x = p_mag * np.sin(theta_hor) * np.cos(theta_ver)
        p0_y = p_mag * np.sin(theta_ver)
        p0_z = p_mag * np.cos(theta_hor) * np.cos(theta_ver) * direction
        y0 = [x0, y0_vert, 0.0, p0_x, p0_y, p0_z] + s_local
        results[idx] = integrate_particle(
            y0, t0, t_end, h, fields=alanlar, return_steps=return_steps
        )

    print(f"5 parçacık paralel koşuyor  (theta0 = {theta0:.2e} rad)...")
    print(f"  0: ideal  (θ=0, θ=0)")
    for i, (th, tv) in enumerate(angles[1:], 1):
        print(f"  {i}: θ_hor={th:+.2e}  θ_ver={tv:+.2e}")
    start = time.time()

    threads = [
        threading.Thread(target=run_particle, args=(i, th, tv))
        for i, (th, tv) in enumerate(angles)
    ]
    for thr in threads:
        thr.start()
    for thr in threads:
        thr.join()

    print(f"Tamamlandı ({time.time()-start:.1f} s). Veriler kaydediliyor...")

    t_array = t0 + np.arange(results[0][0].shape[0]) * (save_interval * h)
    header  = "Time(s)\tDev_X_m\tY_vert_m\tZ_long_m\tPx\tPy\tPz\tS_Rady\tS_Dikey\tS_Long"

    for i, (sonuclar, _, _) in enumerate(results):
        fname = f"particle_{i}.txt"
        with open(fname, "w") as f:
            f.write(header + "\n")
            for j in range(len(t_array)):
                lx, ly, lz = sonuclar[j, 0:3]
                px, py, pz = sonuclar[j, 3:6]
                sx, sy, sz = sonuclar[j, 6:9]
                f.write(
                    f"{t_array[j]:.6e}\t{lx:.6e}\t{ly:.6e}\t{lz:.6e}\t"
                    f"{px:.6e}\t{py:.6e}\t{pz:.6e}\t"
                    f"{sx:.6e}\t{sy:.6e}\t{sz:.6e}\n"
                )
        print(f"  -> particle_{i}.txt  ({len(t_array)} satır)")


if __name__ == "__main__":
    main()
