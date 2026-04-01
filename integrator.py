import ctypes
import numpy as np
import os
import sys

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
    ctypes.c_double,                 # prev_theta_uw_init
    ctypes.POINTER(ctypes.c_double), # final_theta_uw_out
]
_lib.run_integration.restype = None

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

    def to_c_array(self):
        params = [
            self.R0, self.E0, self.E0_power,
            self.B0ver, self.B0rad, self.B0long,
            self.quadK1, self.sextK1,
            self.quadSwitch, self.sextSwitch,
            self.EDMSwitch, self.direction,
            self.nFODO, self.quadLen,
            self.poincare_quad_index,
            self.rfSwitch, self.rfVoltage, self.h, self.driftLen
        ]
        return (ctypes.c_double * len(params))(*params)

def convert_global_to_local_matrix(history_global_np, R0, initial_z):
    X_g = history_global_np[:, 0]
    Y_g = history_global_np[:, 1]
    Z_g = history_global_np[:, 2]
    
    theta_arr = np.arctan2(Y_g, X_g)
    theta_unwrapped = np.unwrap(theta_arr)
    z_local_arr = (theta_unwrapped - theta_unwrapped[0]) * R0 + initial_z
    
    R_g_arr = np.sqrt(X_g**2 + Y_g**2)
    history_local = np.zeros_like(history_global_np)
    
    history_local[:, 0] = R_g_arr - R0
    history_local[:, 1] = Z_g
    history_local[:, 2] = z_local_arr
    
    P_X_g = history_global_np[:, 3]
    P_Y_g = history_global_np[:, 4]
    P_Z_g = history_global_np[:, 5]
    
    history_local[:, 3] = P_X_g * np.cos(theta_arr) + P_Y_g * np.sin(theta_arr)
    history_local[:, 4] = P_Z_g
    history_local[:, 5] = -P_X_g * np.sin(theta_arr) + P_Y_g * np.cos(theta_arr)
    
    S_X_g = history_global_np[:, 6]
    S_Y_g = history_global_np[:, 7]
    S_Z_g = history_global_np[:, 8]
    
    history_local[:, 6] = S_X_g * np.cos(theta_arr) + S_Y_g * np.sin(theta_arr)
    history_local[:, 7] = S_Z_g
    history_local[:, 8] = -S_X_g * np.sin(theta_arr) + S_Y_g * np.cos(theta_arr)
    
    return history_local

def integrate_particle(y0_local, t0, t_end, h, fields=None, return_steps=1000, prev_theta_uw=0.0):
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
    
    history_c = (ctypes.c_double * (9 * return_steps))()
    max_poincare = 200000
    poincare_c = (ctypes.c_double * (9 * max_poincare))()
    poincare_count = (ctypes.c_int * 1)(0)
    poincare_t_c = (ctypes.c_double * max_poincare)()
    
    final_theta_uw_c = (ctypes.c_double * 1)(0.0)
    _lib.run_integration(y0_arr, field_arr, t0, t_end, h, 9, return_steps, history_c, max_poincare, poincare_c, poincare_t_c, poincare_count, ctypes.c_double(prev_theta_uw), final_theta_uw_c)

    history_np = np.ctypeslib.as_array(history_c).reshape((return_steps, 9))
    num_p = poincare_count[0]
    poincare_t_np = np.ctypeslib.as_array(poincare_t_c).copy()[:num_p]
    poincare_np = np.ctypeslib.as_array(poincare_c).reshape((max_poincare, 9))[:num_p]

    hist_local = convert_global_to_local_matrix(history_np, R0, z)
    if num_p > 0:
        poin_local = convert_global_to_local_matrix(poincare_np, R0, z)
    else:
        poin_local = np.array([])

    final_theta_uw = float(final_theta_uw_c[0])

    # Extract actual final global state (y0_arr was modified in-place by C++)
    # convert_global_to_local_matrix can't compute correct z_long for a single
    # point (np.unwrap does nothing). Use final_theta_uw directly instead.
    fg = list(y0_arr)
    X_g, Y_g, Z_g = fg[0], fg[1], fg[2]
    R_g   = np.sqrt(X_g**2 + Y_g**2)
    th    = np.arctan2(Y_g, X_g)
    cos_t = np.cos(th); sin_t = np.sin(th)
    final_local_arr = np.array([
        R_g - R0,                                              # x_rad
        Z_g,                                                   # y_vert
        final_theta_uw * R0,                                   # z_long (unwrapped)
        fg[3]*cos_t + fg[4]*sin_t,                            # p_rad
        fg[5],                                                 # p_vert
        -fg[3]*sin_t + fg[4]*cos_t,                           # p_tang
        fg[6]*cos_t + fg[7]*sin_t,                            # s_rad
        fg[8],                                                 # s_vert
        -fg[6]*sin_t + fg[7]*cos_t,                           # s_tang
    ])

    return hist_local, poin_local, poincare_t_np, final_theta_uw, final_local_arr
