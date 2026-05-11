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
| QF | Odaklayan quadrupol (G₁ > 0) | Radyal düzlemde odaklar. (K-Modülasyon ve misalignment hataları desteklenir) |
| QD | Ayrıştıran quadrupol (−G₁) | Dikey düzlemde odaklar |
| DRIFT | Serbest yol | Saha yok, parçacık düz ilerler |

> **Not:** Halkanın ilk FODO hücresinin (elem=0) girişinde, isteğe bağlı olarak parçacıklara uzunlamasına bir enerji değişimi uygulayan bir **RF Kovuğu (RF Cavity)** tanımlanabilmektedir.

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
- **Dinamik Eğri Uydurma (Adaptive Curve Fit):** Sinyalin uzunluğuna göre kod otomatik olarak en doğru yöntemi seçer:
  1. **Kısa Sinyaller (Sıfır geçişi < 2):** Eğer simülasyon çok kısaysa ve henüz tam bir sinüs dalgası oluşmamışsa, sinüse fit etmek hatalı sonuçlar verir. Bu durumda kod transient etkileri tıraşlar ve veriye en uygun **doğrusal doğruyu (linear fit)** oturtarak spin dönüşüm hızını (rad/s) bulur.
  2. **Uzun Sinyaller (Sıfır geçişi $\ge$ 2):** Simülasyon birden fazla çevrim içeriyorsa, lokal minimum hatalarından kaçınmak için önce **Hızlı Fourier Dönüşümü (FFT)** ile baskın frekans tahmin edilir. Bu tahmin, `scipy.optimize.curve_fit` için başlangıç noktası olarak kullanılır ve zaman uzayında $f(t) = A \cdot \sin(2\pi f t + \phi) + C$ fonksiyonu veriye kusursuz şekilde oturtulur.
Bu hibrit yapı sayesinde Spin-Wheel frekansı her simülasyon uzunluğu için en yüksek doğrulukla tespit edilir.

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
| `rfSwitch` / `rfVoltage`| RF Kovuğu aç/kapat ve voltajı [V] | 0 / 1000000 |
| `N_particles` | Demetteki toplam parçacık sayısı (Space Charge için) | 0e8 |
| `beam_radius_a` | Demet yarıçapı [m] (Space Charge için) | 0.01 |
| `B0hor` / `nFODO_off`| Hata analizi için yatay manyetik alan ve uygulanacağı quad | 0 / 8 |
| `quadModA` / `quadModF` | Parametrik rezonans için Quadrupole modülasyon genliği ve frekansı | 0.0 / 10000.0 |

---

## 10. Demet İçi Etkileşimler (Collective Effects)

Gerçek bir depolama halkasında tek bir proton değil, milyarlarca protondan oluşan bir demet (bunch) döner. Bu protonların birbirleriyle etkileşimi, EDM sinyali üzerinde iki farklı ve kritik bozucu etki yaratır. Simülatör, bu makroskopik ve mikroskopik etkileri **Weak-Strong (Zayıf-Güçlü)** modelleme yaklaşımıyla analitik olarak çözebilecek altyapıya sahiptir:

### 1. Boşluk Yükü (Space Charge) - Sistematik Hata (False EDM)
Milyarlarca protonun oluşturduğu demet, pürüzsüz ve sürekli bir şarj bulutu gibi davranır. Test parçacığı bu bulutun içinde salınım yaparken dışarı doğru itici bir Coulomb kuvveti (Defocusing) hisseder.
- **Tune Kayması:** İtici kuvvet, kuadrupollerin odaklama gücünü zayıflattığı için betatron salınım frekansı düşer ($\Delta Q_y < 0$).
- **Sahte EDM Sinyali:** Eğer demetin merkezi ideal yörüngeden dikey olarak sapmışsa (örneğin hizalama hataları nedeniyle), test parçacığı asimetrik bir elektromanyetik alan hisseder. Thomas-BMT denklemine giren bu sürekli asimetrik radyal manyetik alan, dikey spini ($S_y$) yavaşça yukarı kaldırır. Bu durum, tamamen sahte bir EDM sinyali üretir (Systematic Error).
- **Simülasyon Modeli:** Gauss yasası kullanılarak demetin makroskopik elektrik ve manyetik alanları analitik olarak hesaplanır ve dış alanların üzerine eklenerek Lorentz kuvvetine dahil edilir.

### 2. Demet İçi Saçılma (IBS) - Faz Uyumsuzluğu (Spin Decoherence)
Demetin içindeki protonların mikroskopik ölçekte birebir, rastgele Coulomb çarpışmaları yapmasıdır (Rutherford saçılması).
- **Sihirli Momentumun Bozulması:** Çarpışmalar nedeniyle protonların enerjisi (momentumu) rastgele zıplamalarla değişir. Proton EDM deneyi tamamen $p \approx 0.701$ GeV/c "Sihirli Momentum" koşuluna bağlıdır. Momentumu sapan parçacıkların Thomas presesyonu artık sıfırlanmaz ve spin yatay düzlemde ($S_x, S_z$) çılgınca dönmeye başlar (g-2 precession).
- **Sinyalin Yok Olması:** Trilyonlarca protonun spini farklı yönlere dağıldığı için toplam demet polarizasyonu (okunabilir sinyal) eriyip sıfıra iner. Buna **Spin Decoherence** denir. IBS sahte sinyal üretmez, var olan sinyali yok eder.
- **Simülasyon Modeli:** IBS, Langevin stokastik diferansiyel denklemleri (Rastgele Yürüyüş) kullanılarak, parçacığın momentumuna her entegrasyon adımında rastgele bir difüzyon gürültüsü ($\Delta p$) eklenmesiyle modellenir. *(Not: IBS modülü henüz C++ motoruna tam entegre edilmemiştir, aktif geliştirme aşamasındadır.)*

---

## 11. İleri Seviye Spin Dinamikleri ve Fiziksel Gözlemler

Simülasyon sonuçları incelenirken, özellikle tek parçacık (single-particle tracking) analizlerinde ilk bakışta anomali gibi görünen ancak tamamen fiziksel olan bazı hassas kuantum/klasik mekanik etkiler gözlemlenebilir. C++ motorunun yüksek simplektik hassasiyeti sayesinde yakalanan bu iki önemli fenomen aşağıda açıklanmıştır:

### 1. Dikey Spinde ($S_y$) Beklenmeyen DC Ofset (Geometrik Faz Etkisi)
Elektrik alan ($E0_{ver} = 0$) ve EDM sinyali (`EDMSwitch = 0`) kapalı olmasına rağmen dikey spinde (örneğin $\sim 0.33$ mrad genliğinde) bir kayma (ofset) görülebilir.
- **Nedeni:** Parçacık simülasyona dikey konumda tam merkezden ($y=0$) ancak küçük bir dikey hızla ($\theta_{ver} \neq 0$) başladığında dikey betatron salınımı bir sinüs fonksiyonu olur: $y(t) \propto \sin(\omega_\beta t)$.
- Thomas-BMT denklemine göre bu hareketin yarattığı radyal manyetik alan, spinin dikey düzlemdeki değişim hızını belirler: $dS_y/dt \propto \sin(\omega_\beta t)$.
- Spinin anlık durumunu bulmak için bu hızın integrali alındığında $S_y(t) = \int \sin(\omega_\beta t) dt = 1 - \cos(\omega_\beta t)$ elde edilir.
- $1 - \cos$ fonksiyonu her zaman pozitiftir ve genliğinin yarısı kadar devasa bir **DC ofset** barındırır. Bu durum sahte bir EDM değil, tamamen başlangıç fazından kaynaklı klasik bir geometrik faz (geometric phase) birikimidir. Gerçek bir hızlandırıcıdaki trilyonlarca rastgele fazlı protonun ortalaması alındığında bu ofsetler birbirini mükemmel şekilde sönümleyerek sıfırlanır.

### 2. Hareketli Ortalama Sonrası Ortaya Çıkan ~1462 Hz Sinyali (Spin-Yörünge Vuruntusu)
`plot_results.py` içindeki Hareketli Ortalama (Moving Average) filtresi, yüksek frekanslı ana betatron ($\sim 1.76$ MHz) ve ana spin presesyon ($\sim 490$ kHz) sinyallerini tamamen ezip yok eder. Ancak filtreden sızan ve 1462 Hz gibi çok düşük bir frekansta salınan minik bir sinüs dalgası kalır.
- **Nedeni:** Yatay (radyal) betatron salınımı ($Q_x \approx 8.23$), kuadrupollerdeki manyetik alanı modüle ederek spinin dönüş fazını (Spin Tune $\nu_s \approx 2.237$) etkiler. Bu doğrusal olmayan etkileşim, faz uzayında bir "vuruntu" (beating) frekansı yaratır.
- Halkanın ayrık (discrete) yapısı nedeniyle bu frekanslar devir frekansının ($f_{rev} \approx 218.97$ kHz) harmoniklerinde katlanır (aliasing). 6. harmoniğe göre frekans farkı alındığında: $\Delta Q = |Q_x - \nu_s - 6| \approx |8.230 - 2.237 - 6| \approx 0.0067$ elde edilir.
- Bu küsuratlı rezonans farkını devir frekansıyla çarptığımızda: $0.0067 \times 218974 \text{ Hz} \approx \mathbf{1462 \text{ Hz}}$ frekansına ulaşılır.
- Kısacası bu 1462 Hz sinyali bir hata veya gürültü değil, simülasyonun ikinci dereceden **Intrinsic Spin Resonance (Doğal Spin Rezonansı)** yan bantlarını bile kusursuz şekilde yakalayabildiğinin net bir ispatıdır.

---

## 12. Kurulum ve Çalıştırma

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
