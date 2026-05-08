# Spin-Wheel 6D Tracking Simulation

Bu proje, bir depolama halkasındaki (storage ring) protonların yörünge (orbital) ve spin dinamiklerini yüksek hassasiyetle modellemek üzere geliştirilmiş **6D Parçacık İzleme (Particle Tracking)** simülasyonudur. Projenin ana odak noktası, protonun **Elektrik Dipol Momentini (EDM)** ölçmek için önerilen **"Spin-Wheel"** metodunun fiziksel ve nümerik analizidir.

## 🚀 Temel Özellikler

- **Yüksek Hassasiyetli C++ Motoru:** Parçacığın yörünge (Lorentz kuvveti) ve spin (Thomas-BMT denklemi) hareketleri, 4. dereceden örtük (implicit) **Gauss-Legendre (GL4)** entegratörü ile eşzamanlı çözülür. Bu sympletik benzeri yapı, uzun süreli simülasyonlarda faz uzayı hacmini ve spin uzunluğunu mükemmel şekilde korur.
- **Gerçekçi Örgü (Lattice) Modellemesi:** FODO hücreleri, Quadrupole, Sextupole ve RF kaviteleri eksiksiz modellenir.
- **Spin-Wheel EDM Ölçümü:** Sisteme eklenen dikey elektrik alanı (`E0ver`) sayesinde spinin yavaş salınımları tetiklenir. `EDM_ETA` ($\eta$) duyarlılık parametresi dışarıdan ayarlanabilir.
- **Gelişmiş Sinyal İşleme (Curve Fitting):** Sadece birkaç milisaniyelik kısa simülasyon verilerinden dahi ~110 Hz gibi düşük frekanslı spin salınımlarını (Spin-Wheel frekansını) bulabilmek için **Hareketli Ortalama (Moving Average)** filtrelemesi ve non-lineer eğri uydurma algoritmaları kullanılır. Yüksek frekanslı betatron gürültüleri filtre ile tamamen yok edilir.

## ⚙️ Kurulum ve Kullanım

Simülasyon motorunun kalbi C++ ile yazılmıştır ve Python üzerinden çağrılır (ctypes/shared library).

### 1. C++ Motorunu Derleme
Simülasyonu ilk kez çalıştırıyorsanız veya `integrator.cpp` dosyasında değişiklik yaptıysanız motoru derlemeniz gerekir:
```bash
g++ -O3 -shared -fPIC -o integrator.dylib integrator.cpp
```
*(Linux/Windows kullanıcıları uzantıyı `.so` veya `.dll` olarak değiştirmelidir)*

### 2. Simülasyonu Çalıştırma
Fiziksel parametreleri `params.json` dosyasından ayarladıktan sonra simülasyonu başlatın:
```bash
python run_simulation.py
```
Bu işlem sonucunda `simulation_data.txt` ve `poincare_data.txt` dosyaları üretilir.

### 3. Sonuçları Analiz Etme ve Görselleştirme
```bash
python plot_results.py
```
Bu kod, devasa bir 3x4 panel oluşturur, faz uzaylarını, spinin zamanla değişimini ve FFT grafiklerini çizer. Aynı zamanda dikey spin ($S_y$) üzerine "Curve Fit" algoritmasını uygulayarak terminale tespit edilen EDM spin-wheel frekansını basar. Grafikler `simulasyon_sonuclari.png` olarak kaydedilir.

## 🛠 `params.json` Parametreleri Hakkında

Simülasyon tamamen dışa açık bir JSON konfigürasyonuyla yönetilir. Öne çıkan bazı parametreler:
- `t2`: Simülasyonun toplam süresi (saniye). Spin-wheel analizi için (Curve Fit algoritmasının doğru çalışması adına) minimum **0.01 (10 ms)** olması tavsiye edilir.
- `E0ver`: Deflektörlerdeki dikey elektrik alanı (V/m).
- `EDM_ETA`: Protonun EDM duyarlılık katsayısı ($\eta$). (Standart model beklentisi ~1.88e-15 civarındadır. Gözlemlenebilirliği test etmek için bu değer artırılabilir).
- `poincare_quad_index`: `-1` yapıldığında her FODO hücresinin başında kayıt alınır, bu sayede Tune aliasing (frekans katlanması) hataları önlenir.
- `dev0`: Parçacığın yatay (x) eksenindeki başlangıç sapması. Eğer sıfır olursa, yatay hareket tamamen dikey hareketin enerji kuplajından (dispersion) kaynaklanan bir gölgeye dönüşür. Gerçek yatay Tune değerini okumak için ufak bir değer (örneğin `0.001` mm) verilmelidir.

## 🔬 Spin-Wheel Sinyal Analizi
Simülasyondaki yatay (betatron) salınımları tipik olarak 10-100 kHz mertebesindedir. Spin-Wheel metodunun tetiklediği EDM salınımı ise 100-200 Hz aralığındadır. Algoritma, yüksek frekanslı gürültüyü ekarte etmek için **10 kHz yutma kapasiteli 0.1 ms pencere boyutlu Hareketli Ortalama (Moving Average) Filtresi** kullanır ve filtreli veri üzerinden `scipy.optimize.curve_fit` ile ana frekansı mükemmel hassasiyette çeker.

---
*Geliştirici Notu: Bu kod, EDM araştırmaları için parçacık izleme simülasyonu temel alınarak geliştirilmiştir.*
