import ctypes
import numpy as np
import os
import sys

# ==============================================================================
# C++ PAYLAŞIMLI KÜTÜPHANE YÜKLEME (CTYPES)
# Linux için .so, macOS için .dylib uzantılı derlenmiş kütüphane yüklenir.
# Bu sayede performans kritik işlemler C++ üzerinde çalıştırılırken,
# kontrol Python üzerinden sağlanır.
# ==============================================================================

try:
    if sys.platform == "darwin":
        lib_name = "integrator.dylib"
    else:
        lib_name = "lib_integrator.so"
        
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), lib_name))
    _lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise RuntimeError(f"C++ kütüphanesi ({lib_name}) bulunamadı. Lütfen derleyin.")

_lib.run_integration.argtypes = [
    ctypes.POINTER(ctypes.c_double), # y_init
    ctypes.POINTER(ctypes.c_double), # field_params
    ctypes.c_double,                 # t0
    ctypes.c_double,                 # t_end
    ctypes.c_double,                 # h
    ctypes.c_int,                    # dim
    ctypes.c_int,                    # return_steps
    ctypes.POINTER(ctypes.c_double), # history_out
    ctypes.c_int,                    # max_poincare
    ctypes.POINTER(ctypes.c_double), # poincare_out
    ctypes.POINTER(ctypes.c_double), # poincare_t_out
    ctypes.POINTER(ctypes.c_int),    # poincare_count
]
_lib.run_integration.restype = None

_lib.run_integration.restype = None

# ==============================================================================
# FIZIKSEL PARAMETRE SINIFI (FieldParams)
# C++ motoruna gönderilecek manyetik alanlar, RF özellikleri ve simülasyon
# yapılandırmalarını tutan Python nesnesidir. Bu sınıf içindeki değişkenler,
# C++ tarafındaki `field_params` dizisine aktarılır.
# ==============================================================================
class FieldParams:
    def __init__(self):
        self.R0 = 95.49
        self.E0 = 0.0
        self.E0_power = 1.0
        self.B0ver = 0.0
        self.B0rad = 0.0
        self.B0long = 0.0
        self.quadK1 = 0.0
        self.sextK1 = 0.0
        self.quadSwitch = 1.0
        self.sextSwitch = 0.0
        self.EDMSwitch = 0.0
        self.direction = -1.0
        self.nFODO = 24.0
        self.quadLen = 0.4
        self.driftLen = 2.0
        self.poincare_quad_index = 0.0
        self.rfSwitch = 0.0
        self.rfVoltage = 10000.0
        self.h = 1.0
        self.quadModA = 0.0
        self.quadModF = 0.0
        self.nFODO_off = -1.0
        self.B0hor = 0.0
        self.quadYOffset = 0.0
        self.quadK0 = 0.0

        self.quadK0 = 0.0

    def to_c_array(self):
        """
        Sınıf içindeki tüm parametreleri, C++ fonksiyonuna gönderilebilecek
        formatta (`ctypes.c_double` array) ardışık bir diziye dönüştürür.
        """
        params = [
            self.R0, self.E0, self.E0_power,
            self.B0ver, self.B0rad, self.B0long,
            self.quadK1, self.sextK1,
            self.quadSwitch, self.sextSwitch,
            self.EDMSwitch, self.direction,
            self.nFODO, self.quadLen,
            self.poincare_quad_index,
            self.rfSwitch, self.rfVoltage, self.h, self.driftLen,
            self.quadModA, self.quadModF, self.nFODO_off,
            self.B0hor, self.quadYOffset, self.quadK0
        ]
        return (ctypes.c_double * len(params))(*params)

        return (ctypes.c_double * len(params))(*params)

# ==============================================================================
# KOORDİNAT DÖNÜŞÜMLERİ
# C++ tarafında tüm simülasyon Kartezyen global koordinatlarda çözülür.
# Ancak kullanıcı (analiz) açısından halkanın yerel koordinatları (S, X, Y)
# daha anlamlıdır. Bu fonksiyon Global -> Yerel (Frenet-Serret) dönüşümünü yapar.
# ==============================================================================
def convert_global_to_local_matrix(history_global_np, R0, initial_z):
    """
    Global kartezyen (X, Y, Z, Px, Py, Pz, Sx, Sy, Sz) matrisini,
    yerel ışın izleme koordinatlarına dönüştürür (X_sapma, Y_dikey, S_boylamsal).
    """
    X_c  = history_global_np[:, 0]
    S_c  = history_global_np[:, 1]
    Z_c  = history_global_np[:, 2]
    
    Px_c = history_global_np[:, 3]
    Py_c = history_global_np[:, 4]
    Pz_c = history_global_np[:, 5]
    
    Sx_c = history_global_np[:, 6]
    Sy_c = history_global_np[:, 7]
    Sz_c = history_global_np[:, 8]
    
    history_local = np.zeros_like(history_global_np)
    
    history_local[:, 0] = X_c - R0
    history_local[:, 1] = Z_c
    history_local[:, 2] = S_c + initial_z
    
    history_local[:, 3] = Px_c
    history_local[:, 4] = Pz_c
    history_local[:, 5] = Py_c
    
    history_local[:, 6] = Sx_c
    history_local[:, 7] = Sz_c
    history_local[:, 8] = Sy_c
    
    return history_local

    return history_local

# ==============================================================================
# ANA ENTEGRASYON ÇAĞRISI
# ==============================================================================
def integrate_particle(y0_local, t0, t_end, h, fields=None, return_steps=1000):
    """
    Bir parçacığı verilen başlangıç koşulları ve zaman aralığı için C++ 
    motorunu (GL4 Simplektik Entegratör) kullanarak takip eder.
    
    Parametreler:
    - y0_local: Yerel faz uzayı başlangıç koşulları [x, y, z, px, py, pz, sx, sy, sz]
    - t0: Başlangıç zamanı (saniye)
    - t_end: Bitiş zamanı (saniye)
    - h: Zaman adımı (saniye)
    - fields: FieldParams nesnesi
    - return_steps: Python tarafına kaç adımda bir veri kaydedileceği
    
    Döndürdükleri:
    - hist_local: Tüm kaydedilen adımlar için yerel koordinatlar
    - poin_local: Poincaré kesit noktaları (Belirli bir QF'den geçişler)
    - poincare_t_np: Poincaré geçiş zamanları
    """
    if fields is None: fields = FieldParams()
    R0 = fields.R0

    x, y, z = y0_local[0:3]
    px, py, pz = y0_local[3:6]
    sx, sy, sz = y0_local[6:9]

    theta = z / R0
    R_G = R0 + x
    X_G = R_G * np.cos(theta)
    Y_G = R_G * np.sin(theta)

    y0_global = [X_G, Y_G, y,
                 px * np.cos(theta) - pz * np.sin(theta),
                 px * np.sin(theta) + pz * np.cos(theta), py,
                 sx * np.cos(theta) - sz * np.sin(theta),
                 sx * np.sin(theta) + sz * np.cos(theta), sy]

    y0_arr = (ctypes.c_double * 9)(*y0_global)
    field_arr = fields.to_c_array()

    # C++ hafıza bloklarının (array) Python tarafında tahsis edilmesi.
    # C++ fonksiyonu değerleri bu pointer'ların gösterdiği belleğe yazacaktır.    history_c = (ctypes.c_double * (9 * return_steps))()
    max_poincare = 200000
    poincare_c = (ctypes.c_double * (9 * max_poincare))()
    poincare_count = (ctypes.c_int * 1)(0)
    poincare_t_c = (ctypes.c_double * max_poincare)()

    # Asıl C++ fonksiyon çağrısı (Performans kritik bölüm)
    _lib.run_integration(y0_arr, field_arr, t0, t_end, h, 9, return_steps, history_c, max_poincare, poincare_c, poincare_t_c, poincare_count)

    # C++ bellek bloklarını numpy dizilerine (array) çevirme
    num_p = poincare_count[0]
    poincare_t_np = np.ctypeslib.as_array(poincare_t_c).copy()[:num_p]
    poincare_np = np.ctypeslib.as_array(poincare_c).reshape((max_poincare, 9))[:num_p]

    hist_local = convert_global_to_local_matrix(history_np, R0, z)
    if num_p > 0:
        poin_local = convert_global_to_local_matrix(poincare_np, R0, z)
    else:
        poin_local = np.array([])

    return hist_local, poin_local, poincare_t_np
