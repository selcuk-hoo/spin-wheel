import json
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def load_params(filename="params.json"):
    with open(filename, "r") as f:
        return json.load(f)

def save_params(params, filename="params.json"):
    with open(filename, "w") as f:
        json.dump(params, f, indent=4)

def run_2d_sweep(B0hor_vals, k0_start=0.1, k0_end=0.3, num_k0_steps=11, output_file="sweep_2d_results.npz"):
    """
    K-Modülasyon (Quadrupole Tarama) işlemini gerçekleştirir.
    Dış döngüde manyetik hata (B0hor), iç döngüde Quadrupole gücü (k0) taranır.
    Her adımda simülasyon koşturulur, RMS Kapalı Yörünge (COD) bozulması hesaplanır,
    ve her bir manyetik hata (B0hor) için halkanın hataya ne kadar duyarlı
    olduğunu (Slope / Eğim) çıkartır.
    """
    print(f"Starting 2D sweep: {len(B0hor_vals)} B0hor values x {num_k0_steps} k0 values.")
    
    original_params = load_params()
    k0_original = original_params.get("k0", 0.2)
    B0hor_original = original_params.get("B0hor", 0.0)
    
    k0_vals = np.linspace(k0_start, k0_end, num_k0_steps)
    
    # 2D arrays: shape (len(B0hor_vals), num_k0_steps)
    Qx_matrix = np.zeros((len(B0hor_vals), num_k0_steps))
    Qy_matrix = np.zeros((len(B0hor_vals), num_k0_steps))
    rms_y_matrix = np.zeros((len(B0hor_vals), num_k0_steps))
    
    # 3D array for COD shapes: shape (len(B0hor_vals), num_k0_steps, num_s_points)
    cod_y_3d = []
    s_m = None
    
    qx_regex = re.compile(r"Betatron Tune Qx\s*:\s*([\d\.\+\-e]+)")
    qy_regex = re.compile(r"Betatron Tune Qy\s*:\s*([\d\.\+\-e]+)")
    
    try:
        for i, b_hor in enumerate(B0hor_vals):
            print(f"\n--- B0hor = {b_hor} ({i+1}/{len(B0hor_vals)}) ---")
            cod_y_2d = []
            
            for j, k0 in enumerate(k0_vals):
                print(f"  [k0 = {k0:.4f}]...", end=" ", flush=True)
                start_time = time.time()
                
                params = load_params()
                params["k0"] = float(k0)
                params["B0hor"] = float(b_hor)
                save_params(params)
                
                result = subprocess.run(["python3", "run_simulation.py"], capture_output=True, text=True)
                output = result.stdout
                
                qx_match = qx_regex.search(output)
                qy_match = qy_regex.search(output)
                
                qx_val = float(qx_match.group(1)) if qx_match else np.nan
                qy_val = float(qy_match.group(1)) if qy_match else np.nan
                
                Qx_matrix[i, j] = qx_val
                Qy_matrix[i, j] = qy_val
                
                try:
                    cod_df = pd.read_csv("cod_data.txt", sep="\t")
                    y_cod = cod_df["y_mm"].values
                    rms_y = np.sqrt(np.mean(y_cod**2))
                    
                    rms_y_matrix[i, j] = rms_y
                    cod_y_2d.append(y_cod)
                    
                    if s_m is None:
                        s_m = cod_df["s_m"].values
                except Exception as e:
                    print(f"Error: {e}", end=" ")
                    rms_y_matrix[i, j] = np.nan
                    cod_y_2d.append(np.full_like(s_m, np.nan) if s_m is not None else [])
                
                elapsed = time.time() - start_time
                print(f"Done in {elapsed:.1f}s | Qx={qx_val:.4f}, Qy={qy_val:.4f}, RMS_Y={rms_y:.4f} mm")
                
            cod_y_3d.append(cod_y_2d)
            
    finally:
        print("Restoring original params.json...")
        original_params["k0"] = k0_original
        original_params["B0hor"] = B0hor_original
        save_params(original_params)
        
    cod_y_3d = np.array(cod_y_3d)
    
    # Calculate K-Modulation Sensitivity (Slope of RMS_Y vs k0)
    slopes = np.zeros(len(B0hor_vals))
    for i in range(len(B0hor_vals)):
        valid_idx = ~np.isnan(rms_y_matrix[i, :])
        if np.sum(valid_idx) > 1:
            slope, _ = np.polyfit(k0_vals[valid_idx], rms_y_matrix[i, valid_idx], 1)
            slopes[i] = slope
        else:
            slopes[i] = np.nan
    
    np.savez(output_file, 
             B0hor=np.array(B0hor_vals),
             k0=k0_vals, 
             Qx=Qx_matrix, 
             Qy=Qy_matrix, 
             rms_y=rms_y_matrix, 
             slopes=slopes,
             cod_y_3d=cod_y_3d, 
             s_m=s_m)
    print(f"2D Sweep Results saved to {output_file}")
    return output_file

def plot_2d_results(npz_file="sweep_2d_results.npz", plot_file="sweep_2d_plots.png"):
    """
    K-Modülasyon sonuçlarını (sweep_2d_results.npz) okuyarak grafikleştirir.
    Özellikle RMS Y-COD'un k0'a bağlı değişimi ve 'Sensitivity vs B0hor' 
    doğrusal trendini (linear fit) çizer.
    """
    data = np.load(npz_file)
    B0hor = data['B0hor']
    k0 = data['k0']
    Qx = data['Qx']
    Qy = data['Qy']
    rms_y = data['rms_y']
    slopes = data['slopes']
    
    fig = plt.figure(figsize=(14, 12))
    
    # 1. RMS Y-COD vs k0 for different B0hor
    ax1 = plt.subplot(2, 2, 1)
    for i, b_hor in enumerate(B0hor):
        ax1.plot(k0, rms_y[i, :], '.-', label=f'B0hor = {b_hor}')
    ax1.set_title("Resonance Amplification (RMS Y-COD vs k0)")
    ax1.set_xlabel("İlk Quadrupole Gücü k0 [T/m]")
    ax1.set_ylabel("RMS Y-COD [mm]")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 2. Linear Trend Check: K-Modulation Sensitivity vs B0hor
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(B0hor, np.abs(slopes), 'go-', linewidth=2, markersize=8)
    ax2.set_title("K-Modülasyon Duyarlılığı (Sensitivity vs B0hor)")
    ax2.set_xlabel("Manyetik Hata (B0hor) [T]")
    ax2.set_ylabel("Duyarlılık |d(RMS)/dk0| [mm / (T/m)]")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Add a linear fit line to ax2 to show how linear it is
    valid_slopes = ~np.isnan(slopes)
    if np.sum(valid_slopes) > 1:
        m, c = np.polyfit(B0hor[valid_slopes], np.abs(slopes)[valid_slopes], 1)
        b_range = np.array([0, np.max(B0hor)])
        ax2.plot(b_range, m*b_range + c, 'k--', alpha=0.5, label=f'Fit: y={m:.2e}x + {c:.2e}')
        ax2.legend()
    
    # 3. Tune Shift
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(k0, Qx[0, :], 'b.-', label='Qx (Radyal)')
    ax3.plot(k0, Qy[0, :], 'r.-', label='Qy (Dikey)')
    ax3.set_title("Betatron Tune Kayması (Tune Shift)")
    ax3.set_xlabel("k0 [T/m]")
    ax3.set_ylabel("Tune (Q)")
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    print(f"Plots saved to {plot_file}")

if __name__ == "__main__":
    # Define B0hor values to test (e.g. 0, 1e-6, 5e-6, 1e-5)
    B0hor_vals = [1e-6, 5e-6, 1e-5]
    
    output_data = "sweep_2d_results.npz"
    output_plot = "sweep_2d_plots.png"
    
    # Run the sweep (11 steps per B0hor)
    run_2d_sweep(B0hor_vals=B0hor_vals, k0_start=0.1, k0_end=0.3, num_k0_steps=11, output_file=output_data)
    
    # Plot
    plot_2d_results(npz_file=output_data, plot_file=output_plot)
