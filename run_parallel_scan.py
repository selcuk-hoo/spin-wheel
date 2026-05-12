import os
import sys
import json
import shutil
import subprocess
import re
import concurrent.futures
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_and_run(n_particles):
    run_dir_name = f"run_N_{float(n_particles):.0e}"
    run_dir = os.path.join(BASE_DIR, run_dir_name)
    
    # 1. Dizin oluştur
    os.makedirs(run_dir, exist_ok=True)
    
    # 2. Gerekli dosyaları kopyala
    files_to_copy = [
        "run_simulation.py",
        "plot_results.py",
        "integrator.py",
        "params.json",
        "integrator.dylib",  # macOS için
        "lib_integrator.so"  # Linux için
    ]
    
    for f in files_to_copy:
        src = os.path.join(BASE_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, run_dir)
            
    # 3. params.json içerisindeki N_particles değerini güncelle
    params_path = os.path.join(run_dir, "params.json")
    with open(params_path, "r") as f:
        params = json.load(f)
        
    params["N_particles"] = float(n_particles)
    
    with open(params_path, "w") as f:
        json.dump(params, f, indent=4)
        
    print(f"[{run_dir_name}] Simülasyon başlatıldı...")
    
    # 4. run_simulation.py çalıştır
    env = os.environ.copy()
    subprocess.run([sys.executable, "run_simulation.py"], cwd=run_dir, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"[{run_dir_name}] Simülasyon bitti, grafikler çiziliyor...")
    
    # 5. plot_results.py çalıştır ve çıktısını yakala
    result = subprocess.run([sys.executable, "plot_results.py"], cwd=run_dir, env=env, capture_output=True, text=True)
    
    # 6. Çıktıdan frekans değerini bul
    freq = "Bulunamadı"
    for line in result.stdout.split('\n'):
        if "-> Frekans" in line:
            # Örnek: "-> Frekans : 123.4567890123 Hz"
            match = re.search(r"-> Frekans\s*:\s*([\d\.]+)", line)
            if match:
                freq = match.group(1)
                break
                
    print(f"[{run_dir_name}] İşlem tamamlandı. Frekans: {freq} Hz")
    return n_particles, freq

def main():
    particles_list = [1e7, 1e8, 1e9]
    results = []
    
    print(f"Başlatılıyor: {particles_list} N_particles değerleri için paralel simülasyonlar...")
    
    # Paralel işlem için ThreadPoolExecutor kullanıyoruz
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(particles_list)) as executor:
        # submit all tasks
        futures = {executor.submit(setup_and_run, n): n for n in particles_list}
        
        # wait for results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                n_particles, freq = future.result()
                results.append((f"{n_particles:.0e}", freq))
            except Exception as exc:
                n = futures[future]
                print(f"[HATA] N={n} için simülasyon başarısız oldu: {exc}")
                results.append((f"{n:.0e}", "HATA"))
                
    # Sonuçları n_particles'a göre sırala (string parse problemine karşı orijinal listeye göre)
    results.sort(key=lambda x: float(x[0]))
    
    # Tabloyu ekrana yazdır
    print("\n" + "="*40)
    print("           SİMÜLASYON SONUÇLARI")
    print("="*40)
    
    try:
        # pip install tabulate (kullanıcıda olmayabilir, bu yüzden basic string formatı da kullanabiliriz ama tabulate çok şık durur)
        # Tabulate paketi yoksa normal yazdırma
        table_str = tabulate(results, headers=["N_particles", "S_y Frekansı (Hz)"], tablefmt="grid")
        print(table_str)
    except ImportError:
        # tabulate yüklü değilse fallback
        print(f"{'N_particles':<15} | {'S_y Frekansı (Hz)'}")
        print("-" * 40)
        for n, f in results:
            print(f"{n:<15} | {f}")
            
    print("="*40)

if __name__ == "__main__":
    main()
