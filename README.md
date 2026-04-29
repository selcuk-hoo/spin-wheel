# 6D Proton EDM Spin-Wheel Depolama Halkası Simülatörü

**Yazar:** Selcuk H.

Bu proje, Proton Elektrik Dipol Momenti (EDM) deneyleri için tasarlanmış, tam 6-Boyutlu (6D) ve yüksek hassasiyetli bir depolama halkası simülasyonudur. Parçacık (beam) dinamiği ve Thomas-BMT denklemi üzerinden spin presesyonu hesaplamaları, aşırı yüksek kararlılık gerektiren **GL4 (Gauss-Legendre 4. Derece) Simplektik Entegratör** ile C++ ortamında çözülür. Parametre yönetimi, simülasyon orkestrasyonu ve grafiksel analizler (Betatron Tune NAFF hesaplaması, K-Modülasyon duyarlılık analizi vs.) Python betikleri aracılığıyla sağlanır.

## Temel Fiziksel Özellikler
* **6D Faz Uzayı Takibi:** $x, p_x, y, p_y, z, p_z$ değişkenleri ile tam kuple (coupled) ışın takibi.
* **Thomas-BMT Spin Takibi:** Elektrik ve manyetik alanlarda spin vektörünün $(S_x, S_y, S_z)$ tam çözümü.
* **Simplektik Yapı:** Enerji ve faz uzayı hacmini koruyan 4. dereceden (GL4) örtük (implicit) entegrasyon.
* **K-Modülasyon & Tune Analizi:** `nafflib` aracılığıyla yüksek hassasiyetli Tune ($Q_x, Q_y$) çıkartılması ve Quadrupole hatalarının K-Modülasyon tekniğiyle (Closed Orbit Distortion üzerinden) haritalanması.

## Dosya Yapısı
* `integrator.cpp`: Yüksek performanslı hesaplama motoru (C++ dinamik kütüphanesi).
* `integrator.py`: Python ile C++ kütüphanesi (`ctypes`) arasında veri köprüsü kuran modül.
* `params.json`: Simülasyonun tüm donanımsal (FODO uzunlukları, manyetik alanlar, RF voltajı vb.) parametrelerini barındıran konfigürasyon dosyası.
* `run_simulation.py`: Tekil simülasyonu başlatan, zaman verilerini toplayan ve txt dosyalarına yazan ana Python betiği.
* `plot_results.py`: Elde edilen verilerin spin trendlerini, kapalı yörüngesini (COD) ve Poincaré kesitlerini çizen analiz betiği.
* `sweep_k0.py`: Sistemdeki global hataları ($B_{rad}$ vb.) ölçmek için K-Modülasyon taraması yapan, RMS Y-COD değerlerinin duyarlılığını çıkaran betik.

## Gereksinimler (Prerequisites)
Simülasyonları çalıştırabilmek için sisteminizde C++ derleyicisi ve Python 3 (ilgili kütüphaneler ile) bulunmalıdır:
```bash
pip install numpy scipy pandas matplotlib nafflib
```

## Kurulum ve Derleme (Compilation)
Performans kritik olduğu için `integrator.cpp` dosyasının ilk kullanımdan önce sisteminize uygun bir şekilde dinamik kütüphane (.so veya .dylib) olarak derlenmesi gerekmektedir.

**Linux Ortamında:**
```bash
g++ -shared -o lib_integrator.so -fPIC -O3 integrator.cpp
```

**macOS Ortamında:**
```bash
clang++ -dynamiclib -o lib_integrator.dylib -O3 integrator.cpp
# Not: integrator.py dosyasındaki ctypes.CDLL satırının ".dylib" uzantısını yükleyecek şekilde ayarlandığından emin olun. (Örn: ctypes.CDLL("./lib_integrator.dylib"))
```

## Nasıl Çalıştırılır?

### 1. Tekil Simülasyon Koşusu
Fiziksel parametreleri `params.json` dosyasından ayarladıktan sonra ana simülasyonu çalıştırın:
```bash
python3 run_simulation.py
```
Bu işlem sonunda halkanın farklı noktalarına dair `simulation_data.txt`, `cod_data.txt` ve `poincare_data.txt` veri dosyaları oluşturulacaktır.

### 2. Görselleştirme ve Analiz
Oluşan txt dosyalarındaki verileri işleyip profesyonel grafikler elde etmek için:
```bash
python3 plot_results.py
```

### 3. K-Modülasyon (Quadrupole Sweep) Analizi
Belirli bir $B_{hor}$ hizalama veya global dipol hatası varlığında, ilk Quadrupole'un gücünü ($k_0$) tarayarak halkanın hataya duyarlılığını (Sensitivity) ölçmek için:
```bash
python3 sweep_k0.py
```
Bu betik arka planda çok sayıda simülasyon koşturacak, RMS yörünge sapmalarını bulacak ve `sweep_2d_results.npz` ile analiz grafiğini üretecektir.
