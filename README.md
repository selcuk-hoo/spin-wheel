# 6D Proton EDM & Spin-Wheel Depolama Halkası Simülatörü

**Yazar:** Selcuk H.  
**Güncel Sürüm:** v3.3

Bu proje, Proton Elektrik Dipol Momenti (EDM) deneyleri için tasarlanmış tam 6 boyutlu bir depolama halkası simülasyonudur. Parçacık dinamiği ve spin presesyonu C++ ile yüksek hassasiyetle çözülür; parametre yönetimi, sinyal analizi ve görselleştirme Python katmanında yapılır.

---

## İçindekiler

1. [Fiziksel Arkaplan](#1-fiziksel-arkaplan)
2. [Halka Geometrisi: FODO Örgüsü](#2-halka-geometrisi-fodo-örgüsü)
3. [Koordinat Sistemi](#3-koordinat-sistemi)
4. [C++ Entegratör: `integrator.cpp`](#4-c-entegratör-integratorcpp)
5. [Python Köprüsü: `integrator.py`](#5-python-köprüsü-integratorpy)
6. [Simülasyon Orkestrasyonu: `run_simulation.py`](#6-simülasyon-orkestrasyonu-run_simulationpy)
7. [Diferansiyel Spin Analizi: İdeal Referans Yöntemi](#7-diferansiyel-spin-analizi-ideal-referans-yöntemi)
8. [Görselleştirme: `plot_results.py`](#8-görselleştirme-plot_resultspy)
9. [Parametreler: `params.json`](#9-parametreler-paramsjson)
10. [Demet İçi Etkileşimler (Collective Effects)](#10-demet-i̇çi-etkileşimler-collective-effects)
11. [Betatron Spin Analizi: 5 Parçacık Yöntemi](#11-betatron-spin-analizi-5-parçacık-yöntemi)
12. [İleri Seviye Spin Dinamikleri ve Fiziksel Gözlemler](#12-i̇leri-seviye-spin-dinamikleri-ve-fiziksel-gözlemler)
13. [Kurulum ve Çalıştırma](#13-kurulum-ve-çalıştırma)
14. [Sürüm Geçmişi](#14-sürüm-geçmişi)

---

## 1. Fiziksel Arkaplan

### Neden bu simülasyon?
Proton EDM deneyi, protonun elektrik dipol momentini ölçerek CP-simetri ihlalini aramayı hedefler. Deney, dolaşan protonların spinini radyal elektrik alanla "dondurarak" küçük bir EDM sinyali arar. Bunu yapabilmek için halkadaki zayıf spin sinyalleri izole edilmelidir (Spin-Wheel).

### Sihirli Momentum
Proton EDM deneyinin can alıcı koşulu:
$$p_{\text{magic}} = \frac{m_p c}{\sqrt{G_p}} \approx 0.701\ \text{GeV/c}$$
Bu momentumda, elektrik alandan kaynaklanan spin presesyonu tam olarak sıfırlanır (Thomas terimi ile Larmor terimi birbirini götürür). Böylece spin, radyal yönde donmuş kalır ve yalnızca EDM varlığında dikey bileşen kazanır.

### Spin-Wheel Metodu
Spin-Wheel metodu, EDM ölçümünü kolaylaştırmak için deflektörlere dışarıdan kontrollü bir dikey elektrik alanı ($E_{0ver}$) uygulanması esasına dayanır. Bu alan, proton spininin yavaşça dönmesine (presesyon) sebep olur; ~1115 Hz spin-wheel frekansı oluşur. Bu frekanstaki ince sapmalar doğrudan EDM duyarlılık katsayısına ($\eta$) bağlıdır.

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
| DRIFT | Serbest yol | Alan yok, parçacık düz ilerler |

### Betatron Tune
$$Q_x \approx 2.69 \qquad Q_y \approx 2.36 \quad (G_1 = 0.21\ \text{T/m için})$$

---

## 3. Koordinat Sistemi

Simülatör **global Kartezyen** koordinat kullanır:
- **X**: Halka düzleminde radyal yön (halka merkezinden dışa doğru)
- **Y**: Halka düzleminde azimutal yön (parçacık bu yönde hareket eder)
- **Z**: Dikey yön

Her yay elemanından sonra `rotate_all()` C++ fonksiyonu koordinat çerçevesini −Φ_def kadar döndürür. Python çıktısında sütunlar şu anlama gelir:
- **S_Rady** (sütun 7): Radyal spin bileşeni (~0, ideal durumda sabit)
- **S_Dikey** (sütun 8): Dikey spin bileşeni (= −sin Ωt, spin-wheel salınımı)
- **S_Long** (sütun 9): Boylamsal spin bileşeni (= −cos Ωt)

---

## 4. C++ Entegratör: `integrator.cpp`

### GL4 Simplektik Entegratör
Hareket denklemleri (Newton + Thomas-BMT) **Gauss–Legendre 4. derece örtük Runge–Kutta** yöntemiyle çözülür. GL4 enerjiyi ve faz uzayı hacmini uzun vadede korur. Thomas-BMT spin normunu daima |S|=1'de tutar.

### Elektromanyetik Alanlar
**Yay (ARC):** Silindirik kapasitör + Spin-Wheel dikey alanı:
$$E_r(R,Z) = E_0 \left(\frac{R_0}{R}\right)^n \qquad E_Z = E_{0ver}$$

**Quadrupol (QF/QD):**
$$B_r = G_1\,(Z - d_y) \qquad B_Z = G_1\,(X - R_0 - d_x)$$

### Thomas-BMT Spin Dinamiği
Spin vektörü **S**, Thomas-BMT denklemiyle evrilir. EDM duyarlılık katsayısı $\eta$ (`EDM_ETA`) Python katmanından dinamik olarak çekilir.

> **Not:** `field_params[29]` (C++ dönen çerçeve parametresi) her zaman 0.0 olarak gönderilir; C++ dönen çerçeve devre dışıdır. Diferansiyel ölçüm bu parametreye ihtiyaç duymaz.

---

## 5. Python Köprüsü: `integrator.py`

C++ motoru ile Python arasındaki iletişimi sağlar. `FieldParams` sınıfı tüm fizik parametrelerini tutar ve `to_c_array()` metoduyla `ctypes.c_double` dizisine dönüştürerek `_lib.run_integration(...)` fonksiyonuna gönderir.

`ctypes` C çağrıları sırasında Python GIL'ini serbest bırakır; bu sayede aynı anda birden fazla `integrate_particle()` çağrısı gerçek paralel çok-çekirdekli yürütme sağlar.

---

## 6. Simülasyon Orkestrasyonu: `run_simulation.py`

Ana simülasyon akışını kontrol eder:
1. `params.json` dosyasından tüm girdileri okur.
2. Sihirli momentumda parçacığı yörüngede tutacak ideal E₀ elektrik alanını otomatik hesaplar.
3. **`simulate_ideal = 1` ise:** İdeal referans parçacığını (EDM=0, space charge=0) ana simülasyonla **paralel** koşturur; sonucu `simulation_data_ideal.txt`'e kaydeder.
4. Tune, emitans ve spin trendi eğimlerini ekrana basar.

---

## 7. Diferansiyel Spin Analizi: İdeal Referans Yöntemi

### Motivasyon
Spin-wheel frekansını (~1115 Hz) doğrudan ölçerek EDM etkisini (~μHz) tespit etmek son derece güçtür. İki farklı ölçüm yöntemi (sinüs eğri uydurma, kompleks fazör analizi) sistematik olarak ~17 mHz farklı sonuçlar verir. Bu fark, S_y(t)'deki ~300 Hz fazlı modülasyondan kaynaklanır: iki yöntem bu modülasyonu farklı ağırlıklarla ortalar. Frekans alanında çalışmak yerine doğrudan diferansiyel spin çıktısı ölçmek bu sorunları ortadan kaldırır.

### Yöntem
`simulate_ideal = 1` olduğunda iki simülasyon paralel koşturulur:

| | Ana simülasyon | İdeal referans |
|---|---|---|
| Lattice / E₀ver | params.json | params.json (aynı) |
| EDM (η) | params.json değeri | 0 |
| Space charge (N) | params.json değeri | 0 |
| IBS | aktif olabilir | kapalı |

Her örgü elemanında alınan fark:
$$\Delta S_y(t) = S_y^{\text{ana}}(t) - S_y^{\text{ideal}}(t)$$

### Neden çalışır?
Her iki simülasyon da aynı lattice'i ve aynı E₀ver alanını kullandığı için spin-wheel taşıyıcısı (1115 Hz) ve lattice kaynaklı modülasyon büyük ölçüde iptal olur. Geri kalan ΔS_y yalnızca ölçmek istediğimiz pertürbasyonu (space charge, EDM, IBS) yansıtır.

> **Önemli not:** ΔS_y'nin FFT'si hâlâ 1115 Hz civarında bir tepe içerir çünkü cos(ω·t+ε) − cos(ω·t) ≈ sin(ω·t)·ε(t) ifadesinde taşıyıcı tam iptal olmaz; yalnızca genliği ε kadar küçülür. Ortak-mod baskılanması DC ofset ve yavaş drift için geçerlidir, sinüsoidal taşıyıcı için değil. Bu nedenle FFT analizi ham S_y üzerinden yapılmaktadır.

---

## 8. Görselleştirme: `plot_results.py`

3×4'lük analiz paneli oluşturur:

**Satır 1 & 2:** Radyal/Dikey yörünge (zaman), COD, faz uzayları, FFT spektrumu.

**Satır 3 (Spin panelleri):**

| Panel | `simulate_ideal=0` | `simulate_ideal=1` |
|---|---|---|
| [2,0] | Ham S_x (radyal) | Ham S_x (radyal) |
| [2,1] | Ham S_y (dikey) | ΔS_y = S_y − S_y^ideal |
| [2,2] | Ham S_z (boylamsal) | Ham S_z (boylamsal) |
| [2,3] | FFT(S_y) | FFT(S_y) |

Her panelde Savitzky-Golay filtresi ve doğrusal eğim gösterilir.

### FFT Tepe Analizi
`_sy_fft_peaks()` fonksiyonu S_y FFT'sinde 500–1500 Hz penceresi içindeki ana tepeyi ve side band'leri tespit eder:
- **Hanning penceresi** yan lobları baskılar
- **Parabolik interpolasyon** ile sub-bin hassasiyet: ~±0.2 Hz (T=0.02 s için)
- Konsola frekans, genlik ve Δf değerleri yazdırılır

### Spin Korunumu Kontrolü
Her çalıştırmada `|S|² = Sx² + Sy² + Sz²` aralığı yazdırılır; 1.000000'dan sapma sayısal integrasyon hatasını gösterir.

---

## 9. Parametreler: `params.json`

| Parametre | Açıklama | Varsayılan |
|-----------|----------|------------|
| `t2` | Toplam simülasyon süresi [s] | 0.02 |
| `dev0` / `y0` | Radyal/Dikey başlangıç sapması [m] | 0.0 |
| `theta0_hor` / `theta0_ver` | Başlangıç açısal sapması [rad] | 1e-7 |
| `theta0` | Betatron analizi için açı genliği [rad] (`run_5_particles.py`) | theta0_hor |
| `E0ver` | Spin-Wheel dikey elektrik alanı [V/m] | 1e4 |
| `EDM_ETA` | Proton EDM duyarlılık katsayısı η | 1.88e-15 |
| `EDMSwitch` | EDM etkisini aç/kapat | 0 |
| `simulate_ideal` | İdeal referans parçacığını paralel koştur (0/1) | 0 |
| `N_particles` | Demetteki toplam parçacık sayısı (Space Charge) | 0 |
| `beam_radius_a` | Demet yarıçapı [m] | 0.01 |
| `poincare_quad_index` | Poincaré kesiti indeksi (−1 = her hücre) | −1 |
| `rfSwitch` / `rfVoltage` | RF Kovuğu aç/kapat ve voltajı [V] | 0 / 1e6 |
| `base_spin_freq` | Referans spin frekansı [Hz] (yalnızca kayıt amaçlı) | 1115.74 |

---

## 10. Demet İçi Etkileşimler (Collective Effects)

### Boşluk Yükü (Space Charge) — Sahte EDM Sinyali

Milyarlarca protondan oluşan demetin yarattığı Coulomb alanı, test parçacığı üzerinde dışa doğru itici bir kuvvet (defocusing) uygular.

**Sahte EDM mekanizması:** Eğer demetin merkezi ideal yörüngeden dikey olarak sapmışsa, test parçacığı asimetrik bir elektromanyetik alan görür. Bu asimetri sürekli bir radyal manyetik alan bileşeni oluşturur; Thomas-BMT denklemine giren bu alan dikey spini yavaşça kaydırır. Bu kaydırma, gerçek bir EDM varlığındaki sinyal ile ayırt edilemez — tamamen sahte bir EDM sinyali üretir.

**Simülasyon modeli:** Gauss yasası ile demetin makroskopik E ve B alanları analitik olarak hesaplanıp Lorentz kuvvetine eklenir.

### Demet İçi Saçılma (IBS) — Spin Dekoheransı

Protonların rastgele Coulomb çarpışmaları momentumu dağıtır. Momentumu sihirli değerden sapan parçacıkların Thomas presesyonu sıfırlanmaz; spinler farklı yönlere döner ve toplam polarizasyon sıfıra iner (Spin Decoherence). IBS sahte sinyal üretmez, var olan sinyali yok eder.

---

## 11. Betatron Spin Analizi: 5 Parçacık Yöntemi

### Motivasyon

Space charge'ın ürettiği sahte EDM sinyalinin başlıca kaynağı, betatron salınımları sırasında parçacığın asimetrik bir Coulomb alanı görmesidir. Eğer farklı başlangıç açılarına sahip parçacıkların spinleri uygun şekilde birleştirilirse, betatron salınımına bağlı ortak-mod etkilerin kısmen iptal olması beklenir.

### Parçacık Konfigürasyonları

`run_5_particles.py` beş parçacığı aynı anda paralel olarak çalıştırır:

| İndeks | θ_hor | θ_ver | Açıklama |
|--------|-------|-------|----------|
| 0 | 0 | 0 | İdeal referans (betatron yok) |
| 1 | +θ₀ | +θ₀ | |
| 2 | +θ₀ | −θ₀ | |
| 3 | −θ₀ | +θ₀ | |
| 4 | −θ₀ | −θ₀ | |

θ₀ değeri `params.json` dosyasındaki `theta0` (yoksa `theta0_hor`) anahtarından okunur.

### Spin Kombinasyonları

`plot_5_particle_results.py` beş farklı simetri kombinasyonunu çizer:

| Kombinasyon | İfade | Test edilen simetri |
|-------------|-------|---------------------|
| 1 | $s_{1y} - s_{0y}$ | Tek parçacık farkı |
| 2 | $\frac{1}{2}(s_{1y}+s_{2y}) - s_{0y}$ | Dikey yansıma ortalaması |
| 3 | $\frac{1}{2}(s_{1y}+s_{3y}) - s_{0y}$ | Yatay yansıma ortalaması |
| 4 | $\frac{1}{2}(s_{1y}+s_{4y}) - s_{0y}$ | Köşegen yansıma ortalaması |
| 5 | $\frac{1}{4}(s_{1y}+s_{2y}+s_{3y}+s_{4y}) - s_{0y}$ | Tam simetri ortalaması |

### Sonuç ve Gözlem

Simülasyonlar, symmetric betatron kombinasyonlarının space charge kaynaklı sahte EDM sinyalini **yaklaşık bir mertebe** (~10×) azalttığını göstermektedir. Ancak yüksek N_particles değerlerinde kalan sistematik hata hâlâ **~10²⁰ e·cm** mertebesindedir — bu değer fiziksel EDM hedefinin (~10²⁹ e·cm, Standart Model ötesi fizik için beklenen üst sınır) çok üzerindedir.

**Fiziksel yorum:** Space charge alanı betatron faz uzayında doğrusal değildir; simetrik başlangıç koşullarının ortalaması alınırken doğrusal olmayan terimler tam iptal olmaz. Geriye kalan sahte EDM, space charge yoğunluğuyla (N/a²) ölçeklenir. Gerçek deneyde demeti seyrekleştirmek (düşük N) veya demet yarıçapını artırmak (büyük a) bu sistematik hatayı düşürmenin temel yoludur.

---

## 12. İleri Seviye Spin Dinamikleri ve Fiziksel Gözlemler

### Dikey Spinde DC Ofset (Geometrik Faz)
Elektrik alanı ve EDM kapalı olmasına rağmen dikey başlangıç hızı olan bir parçacıkta ($\theta_{ver} \neq 0$) S_y'de DC ofset gözlemlenebilir. Bunun nedeni:
$$S_y(t) \propto \int \sin(\omega_\beta t)\, dt = 1 - \cos(\omega_\beta t)$$
Bu ifadenin sıfırdan büyük DC bileşeni vardır. Gerçek bir demetin trilyonlarca rastgele fazlı protonunun ortalaması alındığında bu ofsetler birbirini sönümler.

### Spin-Yörünge Yan Bantları (~1462 Hz)
Hareketli ortalama filtresi sonrası ~1462 Hz'de ortaya çıkan sinyal bir hata değildir. Yatay betatron salınımı ($Q_x$) kuadrupollerdeki manyetik alanı modüle ederek Spin Tune ($\nu_s$) ile vuruntuya (beating) girer. 6. devir harmonikle katlanma (aliasing) sonucu:
$$\Delta Q = |Q_x - \nu_s - 6| \approx 0.0067 \quad \Rightarrow \quad f = 0.0067 \times f_{rev} \approx 1462\ \text{Hz}$$
Bu simülatörün ikinci mertebe **Intrinsic Spin Resonance** yan bantlarını yakalayabildiğinin göstergesidir.

---

## 13. Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install numpy scipy matplotlib
```

### Derleme
```bash
# Linux:
g++ -O3 -shared -fPIC -o lib_integrator.so integrator.cpp

# macOS:
g++ -O3 -shared -fPIC -o integrator.dylib integrator.cpp
```

### Temel Kullanım

**Tekil simülasyon:**
```bash
python run_simulation.py    # → simulation_data.txt
python plot_results.py      # → simulasyon_sonuclari.png
```

**İdeal referans ile diferansiyel analiz** (`params.json`: `"simulate_ideal": 1`):
```bash
python run_simulation.py    # → simulation_data.txt + simulation_data_ideal.txt
python plot_results.py      # → ΔS_y paneli otomatik aktif olur
```

**5 parçacık betatron analizi** (`params.json`: `"theta0": 1e-5`):
```bash
python run_5_particles.py          # → particle_0.txt … particle_4.txt
python plot_5_particle_results.py  # → betatron_spin.png
```

### `params.json` Hızlı Referans

```json
{
    "simulate_ideal": 1,
    "theta0": 1e-5,
    "E0ver": 1e4,
    "EDMSwitch": 0,
    "EDM_ETA": 1.88e-15,
    "N_particles": 1e9,
    "t2": 0.02,
    "return_steps": 10000
}
```

---

## 14. Sürüm Geçmişi

### v3.3 — CW/CCW Yön Tutarlılığı ve Spin Doğrulama Araçları

v3.2, tek yönlü (CW) simülasyonlar için geliştirilmişti. Simülatöre CCW (`direction = +1`) desteği eklendiğinde üç kritik fizik hatası gün yüzüne çıktı: her ikisi de aynı halkanın iki farklı yönde gezilen parçacıkları olmasına karşın kod bu simetriyi doğru yansıtmıyordu. Bu sürümde söz konusu hatalar giderildi, kapsamlı bir test altyapısı kuruldu ve CW/CCW spin davranışını görsel olarak incelemeye yarayan yeni bir analiz betiği eklendi.

---

#### 14.1 `integrator.cpp` — Düzeltilen Fizik Hataları

**1. Uzay Yükü Manyetik Alanı (`B_sc`) İşaret Hatası**

Uzay yükü manyetik alanı Thomas-BMT denklemine `B_sc = β × E_sc / c` olarak girer. CW ve CCW parçacıklarının hız vektörleri zıt yönlü olduğundan (`β_CCW = −β_CW`), `B_sc` alanı da iki yönde zıt işaretli olmalıdır — aksi hâlde her iki ışın da "aynı yönden" uzay yükü etkisi görüyor gibi davranır ve sahte EDM sinyali her iki yönde aynı işarette çıkar.

v3.2'de `direction` faktörü gereksiz yere `B_sc` formülüne eklenmişti; bu, dönen referans çerçevesi düzeltmesiyle çakışarak hatalı bir katlama üretiyordu. Düzeltme: `β`'nın yöne bağımlı işareti formüle zaten doğal olarak yansıdığından ayrıca `direction` çarpımına gerek yoktu; kaldırıldı. Doğrulama: uzay yükü açıkken CW ve CCW'de `S_y` kayması zıt işaretli — sağlandı ✓

**2. CCW Yönünde FODO Hücre Geçiş Sırası**

Depolama halkasında parçacık CW gidişte hücreleri 0→1→…→23 sırasıyla, CCW gidişte ise 23→22→…→0 sırasıyla geçer. İki yönde QF ve QD quadrupollerinin rolleri değişir: bir yönde yatayda odaklayıcı olan mıknatıs, diğer yönde dikeyde odaklayıcıdır. v3.2'de CCW parçacığı da CW ile özdeş hücre sırasını izliyordu; bu nedenle hem odaklama dinamikleri hem de hizalama hatası tepkileri yanlış hesaplanıyordu.

Düzeltme: `current_fodo` hesabında CCW için ayna-eşleme, hücre içi eleman sırasında da ters çevrim uygulandı. Sonuçlar: her iki yönde QF/QD rolleri doğru; hizalama hatasına verilen COD tepkisi kick noktasında CW ve CCW'de zıt işaretli ✓; beta fonksiyonu oranları her iki yönde `<2%` farkla simetrik ✓

**3. Boylamsal Manyetik Alan (`B0long`) Yön İşareti**

`B0long`, ışın boyunca tanımlanan boylamsal bir manyetik alandır. CW ve CCW için teğet yönler fiziksel olarak zıt olduğundan bu alanın lab çerçevesindeki gösterimi de işaret değiştirmelidir. `long_sign = −dir_field` çarpımı eklenerek simetri sağlandı.

---

#### 14.2 Spin Sütun Sözleşmesi (Referans)

`integrator.py`'nin döndürdüğü `hist` dizisinde global Kartezyen koordinatlar yerel Frenet-Serret çerçevesine dönüştürülür. İsim benzerliğinden kaynaklanan iki karışıklık dikkat gerektirir:

| `hist` sütunu | Fiziksel anlam | Dikkat |
|:---:|---|---|
| `[:,1]` | **Dikey konum** Z (m) | — |
| `[:,2]` | Yay uzunluğu s (m) | Dikey konum **değil**; 1 devir ≈ 600 m büyür |
| `[:,7]` | **Dikey spin** Sz | EDM sinyali burada yavaşça birikir; başlangıçta 0 |
| `[:,8]` | **Boylamsal spin** Sy | Momentum yönünde; başlangıçta ±1 |

`hist[:,8]` global Y eksenine karşılık gelir ve θ=0'da teğet yönü gösterir. İsmi "Sy" olsa da fiziksel anlamı **boylamsal polarizasyon**dur; EDM ölçümünde izlenmesi gereken **dikey** bileşen `hist[:,7]`'dir. Boylamsal başlangıç polarizasyonu için başlangıç koşulu: `y0_local = [0, 0, 0, 0, 0, p_mag × direction, 0, 0, direction]`.

---

#### 14.3 Yeni Test Betikleri

Üç doğrulama betiği ve bir yardımcı görselleştirme betiği eklendi:

**`test_direction.py`** — CW/CCW temel simetri testlerini çalıştırır: spin normu her iki yönde `|S| = 1.000000` (GL4 simplektik korunumu), spin presesyon frekansı CW = CCW (sihirli momentumda yöne bağımsız), uzay yükü açıkken `S_y` kayması CW ve CCW'de zıt işaretli.

**`test_misalignment.py`** — 8. quadrupole hizalama hatası (`B0hor = 10⁻⁴ T`) uygulanır ve kapalı yörünge bozulması (COD) incelenir. v3.2'de hatalı bir fizik varsayımı vardı: FODO halkasında COD maksimumu kick noktasında değil, beta fonksiyonunun yüksek olduğu noktada oluşur. Test, bu gerçeği yansıtacak şekilde yeniden yazıldı: her iki yönde anlamlı COD genliği oluştuğunu, kick quadrupolünde sıfırdan farklı sapma olduğunu ve CW/CCW değerlerinin kick noktasında zıt işaretli olduğunu doğrular.

**`test_beta.py`** — Küçük başlangıç sapmaları ile Poincaré kesitlerinden FODO beta fonksiyonu ölçülür. `β_x(QF)/β_x(QD) ≈ 2.85`; CW ve CCW arasında `<2%` fark ✓. Bu betik geliştirilirken üç hata giderildi: yanlış başlangıç koşulu sözleşmesi (teğet momentum `pz` yerine dikey momentum `py` olarak ayarlanmıştı), dikey konum yerine yay uzunluğu sütununun kullanılması ve QF/QD Poincaré etiketlerinin yer değiştirmesi.

**`analyze_cod_test.py`** — `test_misalignment.py` çıktısından CW/CCW COD profillerini çizer; hücre sınırları ve kick quadrupolünün konumu işaretlenir.

---

#### 14.4 Yeni Analiz Betiği: `plot_sy_cw_ccw.py`

CW ve CCW tek parçacık simülasyonlarını art arda çalıştırır ve üç panelli karşılaştırma grafiği üretir: (1) dikey spin `S_y(t)` her iki yön için, (2) fark `ΔS_y = S_y^CCW − S_y^CW` ile lineer drift tahmini, (3) boylamsal spin `S_s(t)` ve FFT spin frekansı.

```bash
python plot_sy_cw_ccw.py               # Temel: saf elektrik halkası, EDM/SC kapalı
python plot_sy_cw_ccw.py --edm         # EDM açık (η = 1.88×10⁻¹⁵)
python plot_sy_cw_ccw.py --sc          # Uzay yükü açık (N = 10⁸)
python plot_sy_cw_ccw.py --fodo        # FODO quadrupolleri açık
python plot_sy_cw_ccw.py --E0ver 1e4   # Spin-Wheel sürücüsü
python plot_sy_cw_ccw.py --t 10        # t_end = 10 ms
```

**`E0ver` ve frozen spin sözleşmesi:** Magic momentum koşulu, spin presesyonunu yalnızca *radyal* elektrik alan (`E₀`) için dondurur. `E0ver ≠ 0` dikey bileşen ise dengelenmediğinden yaklaşık 5000 rad/s MDM presesyonu üretir; bu, EDM sinyalini tamamen eze ve CW/CCW yön simetrisini bozar. Varsayılan `E0ver = 0`'dır. EDM karakterizasyonu için bu değer korunmalıdır; Spin-Wheel sürücüsü analizleri için `--E0ver 1e4` açıkça belirtilmelidir.

**EDM yön doğrulaması** (`η = 10⁻³`, `E0ver = 0`): İki yönde momentum yönünde başlatılan spinler, radyal elektrik alanı birbirlerine göre zıt yönden gördüklerinden dikey bileşen karşıt yönlerde büyür:

```
CW  dikey spin: Δ = −0.254  (negatif — elektrik alan sol taraftan etkiyor)
CCW dikey spin: Δ = +0.254  (pozitif — elektrik alan sağ taraftan etkiyor)
```

Zıt yönde ✓ · Tam simetrik ✓ · Beklenen fizikle örtüşüyor ✓
