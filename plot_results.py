import matplotlib
matplotlib.use('Agg') 
import numpy as np
import matplotlib.pyplot as plt
import os

_BASE = os.path.dirname(os.path.abspath(__file__))

def _p(*parts):
    return os.path.join(_BASE, *parts)

def main():
    sim_path = _p("simulation_data.txt")
    if not os.path.exists(sim_path):
        print("HATA: 'simulation_data.txt' bulunamadı.")
        return
        
    data = np.loadtxt(sim_path, skiprows=1)
    t    = data[:, 0] * 1e6  
    x    = data[:, 1] * 1000 
    y    = data[:, 2] * 1000 
    sx   = data[:, 7]
    sy   = data[:, 8]
    sz   = data[:, 9]
    
    x_pc, xp_pc, y_pc, yp_pc = [], [], [], []
    if os.path.exists(_p("poincare_data.txt")):
        try:
            pc_data = np.loadtxt(_p("poincare_data.txt"), skiprows=1)
            if len(pc_data) > 0:
                print(f"[{len(pc_data)} adet Poincare noktası Çiziliyor]")
                # Veri formatı: Dev_X, Y_vert, Z_long, Px, Py, Pz
                x_pc    = pc_data[:, 0] * 1000 
                y_pc    = pc_data[:, 1] * 1000 
                px_pc   = pc_data[:, 3]
                py_pc   = pc_data[:, 4]
                pz_pc   = pc_data[:, 5]
                
                # Açısal momentum sapmaları (mrad)
                x_prime_pc = (px_pc / pz_pc) * 1000
                y_prime_pc = (py_pc / pz_pc) * 1000
                
                x_pc, xp_pc = x_pc, x_prime_pc
                y_pc, yp_pc = y_pc, y_prime_pc
        except ValueError:
            pass

    fig, axs = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('6D Spin-Wheel Simülasyon Sonuçları', fontsize=18, fontweight='bold')
    
    axs[0,0].plot(t, x, 'k-', lw=1.5)
    axs[0,0].set_title("Radyal Konum (x-t)")
    axs[0,0].set_xlabel("Zaman (μs)")
    axs[0,0].set_ylabel("x (mm)")
    axs[0,0].grid(True, linestyle='--', alpha=0.7)
    
    axs[0,1].plot(t, y, 'k-', lw=1.5)
    axs[0,1].set_title("Dikey Konum (y-t)")
    axs[0,1].set_xlabel("Zaman (μs)")
    axs[0,1].set_ylabel("y (mm)")
    axs[0,1].grid(True, linestyle='--', alpha=0.7)
    
    import json
    with open(_p("params.json"), "r") as f:
        params = json.load(f)
    R0 = params.get("R0", 95.49)
    v_nom = 299792458.0 / np.sqrt(1.792847356 + 1.0)
    omega_rev = v_nom / R0
    h_rf = params.get("h", 1.0)
    
    t_sec = data[:, 0]
    z_local = data[:, 3] 
    pz_arr = data[:, 6]
    
    rf_ok = False
    rf_path = _p("rf.txt")
    if os.path.exists(rf_path):
        try:
            rf_data = np.loadtxt(rf_path, skiprows=1)
            if rf_data.ndim == 1:
                rf_data = rf_data.reshape(1, -1)
            nc = rf_data.shape[1]
            if rf_data.shape[0] > 0 and nc >= 7:
                # T, Phi_RF, Theta, Psi, Psi_wrap, P_long, dp_over_p
                phi_rf = rf_data[:, 1]
                psi_wrap = rf_data[:, 4]
                dp_over_p = rf_data[:, 6]
                # RF fazı = sinüsoidal voltaj fazı işte phi_rf; senkrotron fazı (psi_wrap)
                # particle phase-space için kullanılmalıdır (tek elips).
                psi_deg = (psi_wrap * 180.0 / np.pi + 180) % 360 - 180
                phi_rf_deg = (phi_rf * 180.0 / np.pi + 180) % 360 - 180
                dp_p = dp_over_p * 1e3
                # Tercih: psi_wrap kullanarak tek elips görün.
                axs[0, 2].plot(psi_deg, dp_p, "ko", markersize=4)
                # Alternatif için x ekseninde phi_rf kullanmak gerekiyorsa:
                # axs[0, 2].plot(phi_rf_deg, dp_p, "ro", markersize=2, alpha=0.5)
                rf_ok = True
                print(f"[RF faz diyagramı: rf.txt — Ψ_wrap vs dp/p (±{np.max(np.abs(psi_wrap)):.3e} rad) ve dp/p (‰)]")
            elif rf_data.shape[0] > 0 and nc >= 3:
                # Yeni format: T_sec, Phi_RF_rad, dp_over_p
                phi_rf = rf_data[:, 1]
                dp_over_p = rf_data[:, 2]
                phi_deg = (phi_rf * 180.0 / np.pi + 180) % 360 - 180
                dp_p = dp_over_p * 1e3
                axs[0, 2].plot(phi_deg, dp_p, "ko", markersize=4)
                rf_ok = True
                print(f"[RF faz diyagramı: rf.txt — Phi_RF vs dp/p (±{np.max(np.abs(phi_rf)):.3e} rad) ve dp/p (‰)]")
        except (ValueError, OSError, IndexError) as e:
            print(f"UYARI: rf.txt okunamadı ({e}); yedek yöntem kullanılıyor.")
    if not rf_ok:
        if "pc_data" in locals() and len(pc_data) > 0:
            z_long_pc = pc_data[:, 2] * 1000
            pz_pc_arr = pc_data[:, 5]
            
            if pc_data.shape[1] >= 10:
                t_pc = pc_data[:, 9]
            else:
                t_pc = np.interp(-z_long_pc, -z_local * 1000, t_sec)
            
            # RF fazı, sadece zamanla artan sinüsoidal voltaj fazı (parçacık konumundan bağımsız)
            phi_rad = h_rf * (omega_rev * t_pc)
            phi_deg = (phi_rad * 180.0 / np.pi + 180) % 360 - 180
            dp_p = (pz_pc_arr - np.mean(pz_pc_arr)) / np.mean(pz_pc_arr) * 1e3
            axs[0, 2].plot(phi_deg, dp_p, "ko", markersize=4)
        else:
            dp_p = (pz_arr - np.mean(pz_arr)) / np.mean(pz_arr) * 1e3
            phi_rad = h_rf * (omega_rev * t_sec)
            phi_deg = (phi_rad * 180.0 / np.pi + 180) % 360 - 180
            axs[0, 2].plot(phi_deg, dp_p, "k.", markersize=1)
    
    axs[0,2].set_title("RF Faz Diyagramı (Ψ vs dp/p)")
    axs[0,2].set_xlabel("Ψ (sarılı, derece)")
    axs[0,2].set_ylabel("dp/p ($10^{-3}$)")
    axs[0,2].grid(True, linestyle='--', alpha=0.7)
    
    axs[0,3].plot(t, sx, 'k-', lw=1.5, alpha=0.5, label='Ham S_x')
    
    # Savitzky-Golay Sx Filtreleme
    from scipy.signal import savgol_filter
    window_size = (len(sx) // 4) * 2 + 1
    if window_size >= 5:
        sx_filt = savgol_filter(sx, window_length=window_size, polyorder=1)
        axs[0,3].plot(t, sx_filt, 'r--', lw=2, label='Filtrelenmiş')
        axs[0,3].legend(loc='upper right', fontsize=8)
    else:
        sx_filt = sx
        
    axs[0,3].set_title("Radyal Spin ($S_x$-t)")
    axs[0,3].set_xlabel("Zaman (μs)")
    axs[0,3].set_ylabel("$S_x$")
    axs[0,3].grid(True, linestyle='--', alpha=0.7)
    
    # Kenar etkilerini yok etmek için %10'luk baştan sondan kırpmalı (robust) slope hesabı
    trim = int(len(sx_filt) * 0.1)
    if trim > 0 and len(sx_filt) - 2 * trim > 10:
        fit_t = data[trim:-trim, 0]
        fit_s = sx_filt[trim:-trim]
        slope_sx, _ = np.polyfit(fit_t, fit_s, 1)
    else:
        slope_sx, _ = np.polyfit(data[:, 0], sx_filt, 1)
        
    axs[0,3].text(0.05, 0.05, f"Eğim: {slope_sx:.2e} rad/s", transform=axs[0,3].transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # ---------------- POINCARE SECTIONS ----------------
    if len(x_pc) > 0:
        axs[1,0].plot(x_pc, xp_pc, 'ko', markersize=4)
        
        var_x = np.var(x_pc)
        var_xp = np.var(xp_pc)
        cov_x_xp = np.cov(x_pc, xp_pc)[0,1]
        eps_x = 2 * np.sqrt(max(0, var_x * var_xp - cov_x_xp**2))
        axs[1,0].text(0.05, 0.95, f"$\\epsilon_x = {eps_x:.1e}$ $\\pi \\cdot$mm$\\cdot$mrad", transform=axs[1,0].transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Fallback
        px = data[:, 4]
        pz = data[:, 6]
        axs[1,0].plot(x, (px/pz)*1000, 'ko', markersize=1)

    axs[1,0].set_title("Yatay Faz Uzayı (Poincare Kesiti)")
    axs[1,0].set_xlabel("x (mm)")
    axs[1,0].set_ylabel("x' (mrad)")
    axs[1,0].grid(True, linestyle='--', alpha=0.7)
    
    if len(y_pc) > 0:
        axs[1,1].plot(y_pc, yp_pc, 'ko', markersize=4)
        
        var_y = np.var(y_pc)
        var_yp = np.var(yp_pc)
        cov_y_yp = np.cov(y_pc, yp_pc)[0,1]
        eps_y = 2 * np.sqrt(max(0, var_y * var_yp - cov_y_yp**2))
        axs[1,1].text(0.05, 0.95, f"$\\epsilon_y = {eps_y:.1e}$ $\\pi \\cdot$mm$\\cdot$mrad", transform=axs[1,1].transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        py = data[:, 5]
        pz = data[:, 6]
        axs[1,1].plot(y, (py/pz)*1000, 'ko', markersize=1)
        
    axs[1,1].set_title("Dikey Faz Uzayı (Poincare Kesiti)")
    axs[1,1].set_xlabel("y (mm)")
    axs[1,1].set_ylabel("y' (mrad)")
    axs[1,1].grid(True, linestyle='--', alpha=0.7)
    # ---------------------------------------------------
    
    axs[1,2].plot(t, sy, 'k-', lw=1.5, alpha=0.5, label='Ham S_y')
    
    # Savitzky-Golay ile filtrelenmiş trend
    from scipy.signal import savgol_filter
    window_size = (len(sy) // 4) * 2 + 1
    if window_size >= 5:
        sy_filt = savgol_filter(sy, window_length=window_size, polyorder=1)
        axs[1,2].plot(t, sy_filt, 'r--', lw=2, label='Filtrelenmiş Trend')
        axs[1,2].legend(loc='best', fontsize=8)
        
    axs[1,2].set_title("Dikey Spin ($S_y$-t)")
    axs[1,2].set_xlabel("Zaman (μs)")
    axs[1,2].set_ylabel("$S_y$")
    axs[1,2].grid(True, linestyle='--', alpha=0.7)
    
    axs[1,3].plot(t, sz, 'k-', lw=1.5)
    axs[1,3].set_title("Longitudinal Spin ($S_z$-t)")
    axs[1,3].set_xlabel("Zaman (μs)")
    axs[1,3].set_ylabel("$S_z$")
    axs[1,3].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(_p("simulasyon_sonuclari.png"), dpi=300)
    print("Grafik 'simulasyon_sonuclari.png' olarak kaydedildi!")

if __name__ == "__main__":
    main()
