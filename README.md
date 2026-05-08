# 6D Proton EDM & Spin-Wheel Depolama Halkası Simülatörü

**Yazar:** Selcuk H.

Bu proje, Proton Elektrik Dipol Momenti (EDM) deneyleri için tasarlanmış tam 6 boyutlu bir depolama halkası simülasyonudur. Parçacık dinamiği ve spin presesyonu C++ ile yüksek hassasiyetle çözülür; parametre yönetimi, sinyal analizi ve görselleştirme Python katmanında yapılır. Simülatör, yavaş EDM sinyallerinin tespitini (Spin-Wheel metodu) yüksek hassasiyetle simüle edebilmektedir.

---

## İçindekiler

1. [Fiziksel Arkaplan](#1-fiziksel-arkaplan)
2. [Halka Geometrisi: FODO Örgüsü](#2-halka-geometrisi-fodo-örgüsü)
3. [Koordinat Sistemi](#3-koordinat-sistemi)
4. [C++ Entegratör: `integrator.cpp`](#4-c-entegratör-integratorcpp)
5. [Python Köprüsü: `integrator.py`](#5-python-köprüsü-integratorpy)
6. [Simülasyon Orkestrasyonu: `run_simulation.py`](#6-simülasyon-orkestrasyonu-run_simulationpy)
7. [Spin-Wheel Sinyal İşleme ve Analiz](#7-spin-wheel-sinyal-i̇şleme-ve-analiz)
8. [Görselleştirme: `plot_results.py`](#8-görselleştirme-plot_resultspy)
9. [Parametreler: `params.json`](#9-parametreler-paramsjson)
10. [Kurulum ve Çalıştırma](#10-kurulum-ve-çalıştırma)

---

## 1. Fiziksel Arkaplan

### Neden bu simülasyon?
Proton EDM deneyi, protonun elektrik dipol momentini ölçerek CP-simetri ihlalini aramayı hedefler. Deney, dolaşan protonların spinini radyal elektrik alanla "dondurarak" küçük bir EDM sinyali arar. Bunu yapabilmek için halkadaki zayıf spin sinyalleri izole edilmelidir (Spin-Wheel).

### Sihirli Momentum
Proton EDM deneyinin can alıcı koşulu:
$$p_{\text{magic}} = \frac{m_p c}{\sqrt{G_p}} \approx 0.701\ \text{GeV/c}$$
Bu momentumda, elektrik alandan kaynaklanan spin presesyonu tam olarak sıfırlanır (Thomas terimi ile Larmor terimi birbirini götürür). Böylece spin, radyal yönde donmuş kalır ve yalnızca EDM varlığında dikey bileşen kazanır.

### Spin-Wheel Metodu
Spin-Wheel metodu, EDM ölçümünü kolaylaştırmak için deflektörlere dışarıdan kontrollü bir dikey elektrik alanı ($E_{0ver}$) uygulanması esasına dayanır. Bu alan, proton spininin yavaşça dönmesine (presesyon) sebep olur. Bu dönme frekansındaki ince sapmalar (modulation), doğrudan protonun EDM duyarlılık katsayısına ($\eta$) bağlıdır.

---

## 2. Halka Geometrisi: FODO Örgüsü

Halka, 24 özdeş **FODO hücresi**nden oluşur. Her hücre 8 elemandan ibarettir:
```
ARC1 → DRIFT → QF → DRIFT → ARC2 → DRIFT → QD → DRIFT
elem=0   =1    =2    =3    =4    =5    =6    =7
```

| Eleman | Tipi | Görevi |
|--------|------|--------|
| ARC1, ARC2 | Silindirik kapasitör | Parçacığı büküp halka boyunca taşır ve dikey elektrik alan (Spin-Wheel) uygular |
| QF | Odaklayan quadrupol (G₁ > 0) | Radyal düzlemde odaklar |
| QD | Ayrıştıran quadrupol (−G₁) | Dikey düzlemde odaklar |
| DRIFT | Serbest yol | Saha yok, parçacık düz ilerler |

### Betatron Tune
FODO örgüsündeki odaklama gücü, parçacığın halkayı her dolaşımında kaç salınım yaptığını belirler:
$$Q_x \approx 2.69 \qquad Q_y \approx 2.36 \quad (G_1 = 0.21\ \text{T/m için})$$

---

## 3. Koordinat Sistemi

Simülatör **global Kartezyen** koordinat kullanır:
- **X**: Halka düzleminde radyal yön (halka merkezinden dışa doğru)
- **Y**: Halka düzleminde azimutal yön (parçacık bu yönde hareket eder)
- **Z**: Dikey yön

Her yay elemanından sonra `rotate_all()` C++ fonksiyonu koordinat çerçevesini `−Φ_def` kadar döndürür. Böylece parçacık her eleman girişinde `(X ≈ R₀, Y ≈ 0)` konumundan başlıyor gibi görünür (dönen çerçeve).

---

## 4. C++ Entegratör: `integrator.cpp`

### GL4 Simplektik Entegratör
Hareket denklemleri (Newton + Thomas-BMT) **Gauss–Legendre 4. derece örtük Runge–Kutta** yöntemiyle çözülür. GL4 enerjiyi ve faz uzayı hacmini uzun vadede korur, bu da EDM gibi minik kümülatif etkilerin simülasyonunda hayati öneme sahiptir.

### Elektromanyetik Alanlar: `get_electromagnetic_fields()`
**Yay (ARC):** Silindirik kapasitör alanı ve Spin-Wheel için eklenen dikey elektrik alanı:
$$E_r(R,Z) = E_0 \left(\frac{R_0}{R}\right)^n \dots \qquad E_Z = E_{0ver}$$

**Quadrupol (QF/QD):** Kaçıklık bileşenleri dahil saf quadrupol alanı:
$$B_r = G_1\,(Z - d_y) \qquad B_Z = G_1\,(X - R_0 - d_x)$$

### Kapalı Yörünge Verisi (COD) ve Poincaré Kesiti
Her tur ortalaması alınarak kapalı yörünge sapması (COD) `cod_data.txt` dosyasına yazılır.
Ayrıca, `poincare_quad_index = -1` seçildiğinde her FODO hücresinin başında kayıt alınır. Bu sayede aliasing (frekans katlanması) önlenir ve Betatron Tune ($Q_x, Q_y$) kusursuz şekilde hesaplanır.

### Thomas-BMT Spin Dinamiği
Spin vektörü $\mathbf{S}$, Thomas-BMT denklemiyle evrilir. Simülatörde EDM duyarlılık katsayısı olan $\eta$ (`EDM_ETA`) sabit kodlanmamıştır; Python katmanından dinamik olarak çekilir ve Thomas-BMT denklemlerine anlık dahil edilir.

---

## 5. Python Köprüsü: `integrator.py`

C++ motoru ile Python arasındaki iletişimi sağlar. `FieldParams` sınıfı, `R0`, `E0ver`, `EDM_ETA`, `dev0` gibi tüm fizik parametrelerini tutar ve `to_c_array()` metoduyla ardışık bir `ctypes.c_double` dizisine dönüştürerek `_lib.run_integration(...)` fonksiyonuna gönderir.

---

## 6. Simülasyon Orkestrasyonu: `run_simulation.py`

Simülasyonun ana akışını kontrol eder:
1. `params.json` dosyasından tüm girdileri okur.
2. Sihirli momentumdaki parçacığı yörüngede tutacak ideal `E0` elektrik alanını otomatik hesaplar.
3. İstenilen quad kaçıklıkları ve deflektör eğimlerini (tilt) dizi olarak oluşturur.
4. Simülasyon motorunu çalıştırır ve Tune ile Spin trendi eğimlerini ekrana basar.

---

## 7. Spin-Wheel Sinyal İşleme ve Analiz

Spin-Wheel metodunda hedef, `E0ver` ile tetiklenen yavaş dikey spin ($S_y$) presesyon frekansını (örneğin ~110 Hz) son derece yüksek bir hassasiyetle ölçmektir. Ancak parçacık başlangıçta kapalı yörüngeye oturmadığı için (transient state) ~10 kHz mertebesinde devasa betatron salınımları yapar. Bu gürültü EDM sinyalinden binlerce kat daha büyüktür.

**Sinyal İşleme Çözümü:**
- **Hareketli Ortalama (Moving Average) Filtresi:** Veriye 0.1 ms pencere boyutuna sahip bir MA filtresi uygulanır. Bu filtre, betatron gürültülerini faz kaymasına (edge effect) yol açmadan mükemmel şekilde yutar.
- **Non-Lineer Eğri Uydurma (Curve Fit):** Hızlı Fourier Dönüşümü (FFT), 10 ms gibi kısa simülasyon sürelerinde frekansları ayırmak için kör kalır ($\Delta f = 1/T = 100$ Hz). Bu yüzden kod, `scipy.optimize.curve_fit` kullanarak doğrudan zaman uzayında $f(t) = A \cdot \sin(2\pi f t + \phi) + C$ fonksiyonunu veriye oturtur.
Bu sayede sadece 10 ms'lik bir simülasyon verisinden dahi Spin-Wheel frekansı %99.9 doğrulukla (örn: 111.56 Hz) terminale basılır!

---

## 8. Görselleştirme: `plot_results.py`

3×4'lük devasa bir analiz paneli oluşturur:
- **Satır 1 & 2:** Radyal ve Dikey düzlemlerde zamana bağlı yörünge, COD grafiği, Faz Uzayları ve FFT spektrumu.
- **Satır 3:** Spin vektörlerinin ($S_x, S_y, S_z$) zaman içindeki evrimi. $S_y$ panelinde doğrudan MA filtresi ve Eğri Uydurma sonucu üst üste çizilir ve hesaplanan frekans (Hz) grafiğin içine yazılır.
- RF verisi mevcutsa `rf.png` olarak faz diyagramını kaydeder.

---

## 9. Parametreler: `params.json`

Sistem tamamen `params.json` üzerinden yönetilir:

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `t2` | Toplam simülasyon süresi [s] (Spin-Wheel için min: 0.01) | 0.01 |
| `dev0` / `y0` | Radyal/Dikey başlangıç tepmesi [m] | 0.0 |
| `E0ver` | Spin-Wheel dikey elektrik alanı [V/m] | 1000.0 |
| `EDM_ETA` | Proton EDM duyarlılık katsayısı $\eta$ | 1.88e-15 |
| `poincare_quad_index`| Poincaré kesiti kayıt indeksi (−1 = her hücre) | −1 |

---

## 10. Kurulum ve Çalıştırma

### Gereksinimler
Sinyal işleme kütüphaneleri için Python `p39` environment'ının aktif olması önerilir:
```bash
pip install numpy scipy matplotlib
```

### Derleme
Simülatörün C++ motorunu derleyin:
```bash
# macOS için:
g++ -O3 -shared -fPIC -o integrator.dylib integrator.cpp
```

### Adım Adım Kullanım
1. **Fizik Simülasyonunu Çalıştırma** (Spin ve yörünge entegrasyonu):
   ```bash
   python run_simulation.py
   ```
2. **Görselleştirme ve Spin-Wheel Analizi**:
   ```bash
   python plot_results.py
   ```
