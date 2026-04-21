import json
import time
import numpy as np
import os
from integrator import integrate_particle, FieldParams

def load_parameters(param_file="params.json"):
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"{param_file} bulunamadı!")
    with open(param_file, "r") as f:
        config = json.load(f)
    return config

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    config = load_parameters("params.json")
    if os.path.isfile("rf.txt"):
        os.remove("rf.txt")
    if os.path.isfile("cod_data.txt"):
        os.remove("cod_data.txt")
    
    M2 = 0.938272046 
    AMU = 1.792847356 
    C = 299792458.0 
    M1 = 1.672621777e-27
    
    p_magic_base = M2 / np.sqrt(AMU)
    p_magic = p_magic_base * (1.0 + config.get("momError", 0.0))
    E_tot = np.sqrt(p_magic**2 + M2**2)
    beta0 = p_magic / E_tot
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    eRatio = config.get("eRatio", 1.0)
    R0 = config["R0"]
    E0_V_m = -eRatio * (p_magic_base * (p_magic_base / np.sqrt(p_magic_base**2 + M2**2)) / R0) * 1e9
    
    direction = config.get("direction", -1)
    x0 = config.get("dev0", 0.0)      
    y0_vert = config.get("z0", 0.0)   
    z0_long = 0.0                     
    r0_local = [x0, y0_vert, z0_long]
    
    theta0_hor = config.get("theta0_hor", 0.0)
    theta0_ver = config.get("theta0_ver", 0.0)
    
    p_mag = gamma0 * M1 * C * beta0
    p0_x = p_mag * np.sin(theta0_hor) * np.cos(theta0_ver)
    p0_y = p_mag * np.sin(theta0_ver)
    p0_z = p_mag * np.cos(theta0_hor) * np.cos(theta0_ver) * direction
    p_local = [p0_x, p0_y, p0_z]
    
    spinHor = config.get("spinHorRotation", 0.0)
    s0_x = np.sin(spinHor * direction)
    s0_y = 0.0
    s0_z = np.cos(spinHor * direction) * direction
    s_local = [s0_x, s0_y, s0_z]
    
    y0 = []
    y0.extend(r0_local)
    y0.extend(p_local)
    y0.extend(s_local)
    
    alanlar = FieldParams()
    alanlar.R0 = R0
    alanlar.E0 = E0_V_m
    alanlar.E0_power = config.get("E0_power", 1.0)
    alanlar.B0ver = config.get("B0ver", 0.0)
    alanlar.B0rad = config.get("B0rad", 0.0)
    alanlar.B0long = config.get("B0long", 0.0)
    alanlar.quadK1 = config.get("k1", 0.0)
    alanlar.sextK1 = config.get("sextK1", 0.0)
    alanlar.quadSwitch = float(config.get("quadSwitch", 1))
    alanlar.sextSwitch = float(config.get("sextSwitch", 0))
    alanlar.EDMSwitch = float(config.get("EDMSwitch", 0))
    alanlar.direction = float(direction)
    alanlar.nFODO = float(config.get("nFODO", 24))
    alanlar.quadLen = float(config.get("quadLen", 0.4))
    alanlar.driftLen = float(config.get("driftLen", 2.0))
    alanlar.poincare_quad_index = float(config.get("poincare_quad_index", 0))
    alanlar.rfSwitch = float(config.get("rfSwitch", 0))
    alanlar.rfVoltage = float(config.get("rfVoltage", 10000.0))
    alanlar.h = float(config.get("h", 1.0))
    alanlar.quadModA = float(config.get("quadModA", 0.0))
    alanlar.quadModF = float(config.get("quadModF", 0.0))
    
    t0 = 0.0
    t_end = config.get("t2", 1e-5)
    h = config.get("dt", 1e-11)
    return_steps = config.get("return_steps", 10000)
    adim_sayisi = int((t_end - t0) / h)
    
    print("\n================ SİMÜLASYON PARAMETRELERİ ================")
    print(f"R0 (Yarıçap)      : {R0} m")
    print(f"Sihirli Momentum  : {p_magic_base:.6f} GeV/c")
    print(f"Momentum Hatası   : {config.get('momError', 0.0):.1e} (dp/p)")
    print(f"Başlangıç dikey y : {y0_vert*1000:.2f} mm")
    print(f"Elektrik Alan (E0): {E0_V_m/1e6:.4f} MV/m")
    print(f"FODO Hücre Sayısı : {alanlar.nFODO}")
    print(f"Quadrupole (K1)   : {alanlar.quadK1}")
    print(f"Sextupole (S1)    : {alanlar.sextK1}")
    print("==========================================================\n")
    print(f"Simülasyon motoru çalışıyor (Toplam Adım: {adim_sayisi:,})")
    start_time = time.time()

    sonuclar_local, poin_local, poincare_t_arr = integrate_particle(
        y0, t0, t_end, h, fields=alanlar, return_steps=return_steps
    )
    end_time = time.time()
    
    print("==================================================")
    r_son = sonuclar_local[-1, 0:3]
    s_son = sonuclar_local[-1, 6:9]
    sz_son = s_son[2]
    s_yatay = np.sqrt(s_son[0]**2 + s_son[2]**2)
    s_norm = np.linalg.norm(s_son)
    
    print(f"-> Son Radyal Sapma (x)      : {r_son[0]*1000:.4f} mm")
    print(f"-> Son Dikey Yükseklik (y)   : {r_son[1]*1000:.4f} mm")
    tur_sayisi = r_son[2] / (2.0 * np.pi * R0)
    print(f"-> Toplam Atılan Tur Sayısı  : {tur_sayisi:.3f} tur")
    
    save_interval = max(1, int(adim_sayisi / return_steps))
    t_array = t0 + np.arange(sonuclar_local.shape[0]) * (save_interval * h)
    
    if poin_local.shape[0] > 1:
        x_pc = poin_local[:, 0] * 1000
        y_pc = poin_local[:, 1] * 1000
        pz_pc = poin_local[:, 5]
        xp_pc = (poin_local[:, 3] / pz_pc) * 1000
        yp_pc = (poin_local[:, 4] / pz_pc) * 1000
        
        var_x = np.var(x_pc)
        var_xp = np.var(xp_pc)
        cov_x = np.cov(x_pc, xp_pc)[0,1]
        eps_x = 2 * np.sqrt(max(0, var_x * var_xp - cov_x**2))
        
        var_y = np.var(y_pc)
        var_yp = np.var(yp_pc)
        cov_y = np.cov(y_pc, yp_pc)[0,1]
        eps_y = 2 * np.sqrt(max(0, var_y * var_yp - cov_y**2))
        
        print(f"-> Geometrik Emitans (x)     : {eps_x:.1e} pi*mm*mrad")
        print(f"-> Geometrik Emitans (y)     : {eps_y:.1e} pi*mm*mrad")

        if poin_local.shape[0] > 3:
            def _tune_full(u, up):
                uc = u - u.mean(); upc = up - up.mean()
                dphi = np.diff(np.unwrap(np.arctan2(upc, uc)))
                avg_dphi = abs(np.mean(dphi))
                if alanlar.poincare_quad_index < 0:
                    return (alanlar.nFODO * avg_dphi) / (2 * np.pi)
                return avg_dphi / (2 * np.pi)
            Qx = _tune_full(x_pc, xp_pc)
            Qy = _tune_full(y_pc, yp_pc)
            
            print(f"-> Betatron Tune Qx          : {Qx:.4f}")
            print(f"-> Betatron Tune Qy          : {Qy:.4f}")
        
    sx_arr = sonuclar_local[:, 6]
    sy_arr = sonuclar_local[:, 7]
    
    # Savitzky-Golay Filtresi ile Salinim Giderimi
    from scipy.signal import savgol_filter
    window_size = (len(sx_arr) // 4) * 2 + 1 
    if window_size < 5: window_size = 5
    sx_filtered = savgol_filter(sx_arr, window_length=window_size, polyorder=1)
    sy_filtered = savgol_filter(sy_arr, window_length=window_size, polyorder=1)
    
    trim = int(len(sx_filtered) * 0.1)
    if trim > 0 and len(sx_filtered) - 2 * trim > 10:
        fit_t = t_array[trim:-trim]
        fit_sx = sx_filtered[trim:-trim]
        fit_sy = sy_filtered[trim:-trim]
        slope_sx, _ = np.polyfit(fit_t, fit_sx, 1)
        slope_sy, _ = np.polyfit(fit_t, fit_sy, 1)
    else:
        slope_sx, _ = np.polyfit(t_array, sx_filtered, 1)
        slope_sy, _ = np.polyfit(t_array, sy_filtered, 1)
    
    print(f"-> Radyal Trend Eğimi (S_x-t): {slope_sx:.4e} rad/s")
    print(f"-> Dikey  Trend Eğimi (S_y-t): {slope_sy:.4e} rad/s")
    print("--------------------------------------------------")
    
    print("Sürekli (Continuous) veriler yazılıyor (simulation_data.txt)...")
    save_interval = max(1, int(adim_sayisi / return_steps))
    t_array = t0 + np.arange(sonuclar_local.shape[0]) * (save_interval * h)
    
    with open("simulation_data.txt", "w") as f:
        f.write("Time(s)\tDev_X_m\tY_vert_m\tZ_long_m\tPx\tPy\tPz\tS_Rady\tS_Dikey\tS_Long\n")
        for i in range(len(t_array)):
            lx, ly, lz = sonuclar_local[i, 0:3]
            px, py, pz = sonuclar_local[i, 3:6]
            sx, sy, sz = sonuclar_local[i, 6:9]
            f.write(f"{t_array[i]:.6e}\t{lx:.6e}\t{ly:.6e}\t{lz:.6e}\t"
                    f"{px:.6e}\t{py:.6e}\t{pz:.6e}\t"
                    f"{sx:.6e}\t{sy:.6e}\t{sz:.6e}\n")
                    
    print(f"{int(alanlar.poincare_quad_index)}. Quadrupole Poincare verileri yazılıyor (poincare_data.txt)...")
    with open("poincare_data.txt", "w") as f:
        f.write("Dev_X_m\tY_vert_m\tZ_long_m\tPx\tPy\tPz\tS_Rady\tS_Dikey\tS_Long\tT_sec\n")
        for i in range(poin_local.shape[0]):
            lx, ly, lz = poin_local[i, 0:3]
            px, py, pz = poin_local[i, 3:6]
            sx, sy, sz = poin_local[i, 6:9]
            t_pc = poincare_t_arr[i]
            f.write(f"{lx:.6e}\t{ly:.6e}\t{lz:.6e}\t"
                    f"{px:.6e}\t{py:.6e}\t{pz:.6e}\t"
                    f"{sx:.6e}\t{sy:.6e}\t{sz:.6e}\t{t_pc:.10e}\n")
    print(f"-> Toplam {poin_local.shape[0]} adet Poincare verisi kaydedildi.")
    if alanlar.rfSwitch > 0.0 and os.path.isfile("rf.txt"):
        with open("rf.txt") as rf_f:
            n_rf = max(0, sum(1 for _ in rf_f) - 1)
        print(f"-> RF kovuk geçişleri: rf.txt ({n_rf} satır, RF faz diyagramı için).")

if __name__ == "__main__":
    main()
