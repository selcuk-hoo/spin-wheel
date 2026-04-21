import numpy as np
import json
from scipy.optimize import minimize_scalar
from integrator import integrate_particle, FieldParams
import time

def load_parameters(param_file="params.json"):
    with open(param_file, "r") as f:
        return json.load(f)

def evaluate_spin_slope(sextK1_val, base_config):
    # Setup parameters
    M2 = 0.938272046 
    AMU = 1.792847356 
    C = 299792458.0 
    M1 = 1.672621777e-27
    
    p_magic_base = M2 / np.sqrt(AMU)
    p_magic = p_magic_base * (1.0 + base_config.get("momError", 1e-4))
    E_tot = np.sqrt(p_magic**2 + M2**2)
    beta0 = p_magic / E_tot
    gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
    R0 = base_config["R0"]
    eRatio = base_config.get("eRatio", 1.0)
    E0_V_m = -eRatio * (p_magic_base * (p_magic_base / np.sqrt(p_magic_base**2 + M2**2)) / R0) * 1e9
    
    direction = base_config.get("direction", -1)
    
    # User's target condition
    x0 = base_config.get("dev0", 0.0)      
    y0_vert = 1e-2  # 1 cm vertical excursion
    z0_long = 0.0                     
    r0_local = [x0, y0_vert, z0_long]
    
    theta0_hor = base_config.get("theta0_hor", 0.0)
    theta0_ver = base_config.get("theta0_ver", 0.0)
    
    p_mag = gamma0 * M1 * C * beta0
    p0_x = p_mag * np.sin(theta0_hor) * np.cos(theta0_ver)
    p0_y = p_mag * np.sin(theta0_ver)
    p0_z = p_mag * np.cos(theta0_hor) * np.cos(theta0_ver) * direction
    p_local = [p0_x, p0_y, p0_z]
    
    spinHor = base_config.get("spinHorRotation", 0.0)
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
    alanlar.E0_power = base_config.get("E0_power", 1.0)
    alanlar.B0ver = base_config.get("B0ver", 0.0)
    alanlar.B0rad = base_config.get("B0rad", 0.0)
    alanlar.B0long = base_config.get("B0long", 0.0)
    alanlar.quadK1 = base_config.get("k1", 0.0)
    
    if type(sextK1_val) is np.ndarray or type(sextK1_val) is list:
        alanlar.sextK1 = float(sextK1_val[0])
    else:
        alanlar.sextK1 = float(sextK1_val)
        
    alanlar.quadSwitch = float(base_config.get("quadSwitch", 1))
    alanlar.sextSwitch = 1.0
    alanlar.EDMSwitch = float(base_config.get("EDMSwitch", 0))
    alanlar.direction = float(direction)
    alanlar.nFODO = float(base_config.get("nFODO", 24))
    alanlar.quadLen = float(base_config.get("quadLen", 0.4))
    alanlar.poincare_quad_index = float(base_config.get("poincare_quad_index", 0))
    alanlar.rfSwitch = float(base_config.get("rfSwitch", 0))
    alanlar.rfVoltage = float(base_config.get("rfVoltage", 10000.0))
    alanlar.h = float(base_config.get("h", 1.0))
    
    t0 = 0.0
    t_end = 100e-6 
    h = base_config.get("dt", 1e-11)
    
    return_steps = 1000
    
    sonuclar_local, _, _ = integrate_particle(y0, t0, t_end, h, fields=alanlar, return_steps=return_steps)
    
    adim_sayisi = int((t_end - t0) / h)
    save_interval = max(1, int(adim_sayisi / return_steps))
    t_array = t0 + np.arange(sonuclar_local.shape[0]) * (save_interval * h)
    
    sx_arr = sonuclar_local[:, 6]
    slope_sx, _ = np.polyfit(t_array, sx_arr, 1)
    
    return abs(slope_sx)

def main():
    config = load_parameters("params.json")
    print("==========================================================")
    print("Sextupole (K2) Grid Search & L-BFGS-B Optimizasyonu")
    print("Hedef: Radyal spin salınım eğimini (S_x - t) minimize etmek.")
    print("Aralık: [-0.015, 0.015]")
    print("Simülasyon: 100 mikrosaniye, y0 = 1.0 cm")
    print("==========================================================\n")
    
    start_time = time.time()
    
    # 1. Coarse Grid Search
    print("Adım 1: Grid Search (Kaba Tarama) yapılıyor...")
    sext_values = np.linspace(-0.015, 0.015, 15)
    best_s = 0.0
    best_val = float('inf')
    
    for s_val in sext_values:
        val = evaluate_spin_slope(s_val, config)
        print(f"  Tarama: K2 = {s_val:8.4f}  => Spin Eğimi: {val*1e9:12.2f} nrad/s")
        if val < best_val:
            best_val = val
            best_s = s_val
            
    print(f"\nGrid Search En İyi K2: {best_s:8.4f} ({best_val*1e9:.2f} nrad/s)")
    
    # 2. Localized Refinement (Nelder-Mead)
    print("\nAdım 2: Nelder-Mead (Gradient-Free) Yöntemi ile hassas arama...")
    from scipy.optimize import minimize
    res = minimize(evaluate_spin_slope, x0=[best_s], method='Nelder-Mead', args=(config,), options={'xatol': 1e-5, 'maxiter': 30})
    
    if res.success or res.fun < best_val:
        optimum_k2 = float(res.x[0]) if hasattr(res.x, '__len__') else float(res.x)
        min_slope = float(res.fun)
        
        print("\n================ OPTİMİZASYON TAMAMLANDI ================")
        print(f"Optimum Sextupole (K2) Değeri : {optimum_k2:.6f}")
        print(f"Minimum Spin Eğim Hassasiyeti : {min_slope*1e9:.2f} nrad/s")
        print(f"Geçen Süre                    : {time.time() - start_time:.1f} saniye")
        print("==========================================================")
        
        config["sextK1"] = float(f"{optimum_k2:.4f}")
        with open("params.json", "w") as f:
            json.dump(config, f, indent=4)
        print("-> 'params.json' dosyası yeni optimize edilmiş 'sextK1' ile güncellendi.")
    else:
        print("\nOptimizasyon başarısız oldu:", res.message)

if __name__ == "__main__":
    main()
