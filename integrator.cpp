#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

extern "C" {

const double C_LIGHT = 299792458.0;
const double M_P     = 1.672621777e-27;
const double Q_E     = 1.602176565e-19;
const double G_P     = 1.792847356;
const double EDM_ETA = 1.88e-15;

inline void cross_product(const double* a, const double* b, double* res) {
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}

inline double dot_product(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void rotate_all(double* y, double theta) {
    double X = y[0], Y = y[1];
    double Px = y[3], Py = y[4];
    double Sx = y[6], Sy = y[7];
    
    double cos_th = std::cos(theta);
    double sin_th = std::sin(theta);
    
    y[0] = X * cos_th - Y * sin_th;
    y[1] = X * sin_th + Y * cos_th;
    
    y[3] = Px * cos_th - Py * sin_th;
    y[4] = Px * sin_th + Py * cos_th;
    
    y[6] = Sx * cos_th - Sy * sin_th;
    y[7] = Sx * sin_th + Sy * cos_th;
}

// element_type: 0 = DEFLECTOR, 1 = DRIFT, 2 = QUAD_F, 3 = QUAD_D, 4 = QUAD_F_MOD (cell-0 modulated)
void get_electromagnetic_fields(double t, const double* r, const double* field_params, int element_type, double* E, double* B) {
    double R0       = field_params[0];
    double E0       = field_params[1];
    double n        = field_params[2];
    double B0ver    = field_params[3];
    double B0rad    = field_params[4];
    double B0long   = field_params[5];
    double quadK1   = field_params[6];
    double sextK1   = field_params[7];
    
    double X = r[0], Y = r[1], Z = r[2];
    double R = std::sqrt(X*X + Y*Y);

    E[0] = 0.0; E[1] = 0.0; E[2] = 0.0;
    B[0] = 0.0; B[1] = 0.0; B[2] = 0.0;

    if (element_type == 0) {
        // DEFLECTOR
        double cos_th = X / R;
        double sin_th = Y / R;
        double E_r = 0.0, E_z = 0.0;
        if (R > 1e-6) {
            if (n == 1.0) {
                E_r = E0 * R0 / R;
            } else {
                double zR  = Z / R;
                double zR2 = zR * zR;
                double pow_n = std::pow(R0/R, n);
                E_r = E0 * pow_n * (1.0 - 0.5*(n*n-1.0)*zR2 + (n*n-1.0)*(n+1.0)*(n+3.0)*zR2*zR2/24.0);
                E_z = E0 * pow_n * ((n-1.0)*zR - (n*n-1.0)*(n+1.0)*zR2*zR/6.0);
            }
        }
        E[0] = E_r * cos_th;
        E[1] = E_r * sin_th;
        E[2] = E_z;
        
        B[0] = -B0rad * cos_th + B0long * sin_th;
        B[1] = -B0rad * sin_th - B0long * cos_th;
        B[2] = B0ver;
    } else if (element_type == 2 || element_type == 3) {
        // QUAD
        double current_K1 = (element_type == 2) ? quadK1 : -quadK1;
        
        // Quad lies on a straight path, so lateral deviation is purely horizontal X minus R0.
        double dev_quad = X - R0; 
        
        double B_quad_r = current_K1 * Z;
        double B_quad_z = current_K1 * dev_quad;
        
        // Sextupole 
        double sextSwitch = field_params[9];
        if (sextSwitch > 0.0) {
            double current_sK1 = (element_type == 2) ? sextK1 : -sextK1;
            B_quad_r += current_sK1 * dev_quad * Z;
            B_quad_z += current_sK1 * (dev_quad*dev_quad - Z*Z);
        }
        
        // fields acting transversely
        B[0] = B_quad_r;
        B[1] = 0.0;
        B[2] = B_quad_z;
    } else if (element_type == 4) {
        // QUAD_F_MOD: focusing quad with time-modulated strength (cell 0 only)
        double A_mod  = field_params[19];
        double f_mod  = field_params[20];
        double K1_eff = quadK1 * (1.0 + A_mod * std::cos(2.0 * M_PI * f_mod * t));

        double dev_quad = X - R0;
        double B_quad_r = K1_eff * Z;
        double B_quad_z = K1_eff * dev_quad;

        double sextSwitch = field_params[9];
        if (sextSwitch > 0.0) {
            B_quad_r += sextK1 * dev_quad * Z;
            B_quad_z += sextK1 * (dev_quad*dev_quad - Z*Z);
        }

        B[0] = B_quad_r;
        B[1] = 0.0;
        B[2] = B_quad_z;
    }
}

void compute_rhs(double t, const double* y, const double* field_params, int element_type, double* dydt, int dim) {
    const double* r = &y[0];
    const double* p = &y[3];
    const double* s = &y[6];

    double p_sq  = dot_product(p, p);
    double mc    = M_P * C_LIGHT;
    double gamma = std::sqrt(1.0 + p_sq / (mc * mc));

    double v[3], beta[3];
    for (int i = 0; i < 3; ++i) {
        v[i]    = p[i] / (gamma * M_P);
        beta[i] = v[i] / C_LIGHT;
    }

    double E[3], B[3];
    get_electromagnetic_fields(t, r, field_params, element_type, E, B);

    double v_cross_B[3];
    cross_product(v, B, v_cross_B);

    double dpdt[3];
    for (int i = 0; i < 3; ++i)
        dpdt[i] = Q_E * (E[i] + v_cross_B[i]);

    double EDMSwitch   = field_params[10];
    double beta_dot_B  = dot_product(beta, B);
    double beta_dot_E  = dot_product(beta, E);
    double beta_cross_E[3], beta_cross_B[3];
    cross_product(beta, E, beta_cross_E);
    cross_product(beta, B, beta_cross_B);

    double AMU = G_P;
    double Omega[3];
    for (int i = 0; i < 3; ++i) {
        double mdm_term = (B[i] * (AMU + 1.0/gamma))
                        - (beta[i] * beta_dot_B * AMU * gamma/(gamma+1.0))
                        - (beta_cross_E[i] * (AMU + 1.0/(gamma+1.0)) / C_LIGHT);
        double edm_term = 0.0;
        if (EDMSwitch > 0.0)
            edm_term = (beta_cross_B[i] + E[i]/C_LIGHT
                        - beta[i]*beta_dot_E * gamma/(gamma+1.0)/C_LIGHT) * 0.5 * EDM_ETA;
        Omega[i] = -(mdm_term + edm_term) * (Q_E / M_P);
    }

    double dsdt[3];
    cross_product(Omega, s, dsdt);

    for (int i = 0; i < 3; ++i) {
        dydt[i]   = v[i];
        dydt[i+3] = dpdt[i];
        dydt[i+6] = dsdt[i];
    }
}

void gl4_step_element(double t, double* y, const double* field_params, int element_type, double h, int dim) {
    const double sq3 = std::sqrt(3.0) / 6.0;
    const double c1  = 0.5 - sq3, c2 = 0.5 + sq3;
    const double a11 = 0.25, a12 = 0.25 - sq3;
    const double a21 = 0.25 + sq3, a22 = 0.25;

    double k1[9], k2[9], y1[9], y2[9];

    compute_rhs(t, y, field_params, element_type, k1, dim);
    for (int i = 0; i < dim; ++i) k2[i] = k1[i];

    for (int iter = 0; iter < 4; ++iter) {
        for (int i = 0; i < dim; ++i) {
            y1[i] = y[i] + h * (a11*k1[i] + a12*k2[i]);
            y2[i] = y[i] + h * (a21*k1[i] + a22*k2[i]);
        }
        compute_rhs(t + c1*h, y1, field_params, element_type, k1, dim);
        compute_rhs(t + c2*h, y2, field_params, element_type, k2, dim);
    }

    for (int i = 0; i < dim; ++i)
        y[i] += h * (0.5*k1[i] + 0.5*k2[i]);
}

void run_integration(double* y_init, const double* field_params,
                     double t0, double t_end, double h, int dim,
                     int return_steps, double* history_out,
                     int max_poincare, double* poincare_out,
                     double* poincare_t, int* poincare_count) {
    
    long long total_steps_est = (long long)((t_end - t0) / h);
    if (total_steps_est <= 0) return;
    long long save_interval = total_steps_est / return_steps;
    if (save_interval == 0) save_interval = 1;

    double R0       = field_params[0];
    int    nFODO    = (int)(field_params[12] + 0.5);
    double quadLen  = field_params[13];
    double dir      = field_params[11];
    double driftLen = field_params[18];

    int    target_quad  = (int)std::round(field_params[14]);

    const bool rf_on = (field_params[15] > 0.0);
    double V_rf      = field_params[16];
    double h_rf      = field_params[17];

    double M_GeV   = 0.938272046;
    double p_magic = M_GeV / std::sqrt(G_P);
    double E_magic = std::sqrt(p_magic*p_magic + M_GeV*M_GeV);
    double beta_magic = p_magic / E_magic;
    double circumference = 2.0 * M_PI * R0 + 4.0 * nFODO * driftLen + 2.0 * nFODO * quadLen; 
    double omega_rf = h_rf * 2.0 * M_PI * beta_magic * C_LIGHT / circumference;

    std::ofstream rf_out;
    rf_out.open("rf.txt", std::ios::app);
    if (rf_out.is_open() && rf_out.tellp() == 0)
        rf_out << "T_sec\tPhi_RF_rad\tdp_over_p\n";

    double t = t0;
    int save_idx = 0;
    int p_saved = 0;
    long long global_step = 0;
    long long print_interval = total_steps_est / 10;
    if (print_interval == 0) print_interval = 1;

    double p_tang_ref = 0.0;
    bool have_ref = false;

    int total_fodo_cells = 0;
    double global_S = 0.0;
    // Lengths & Angles
    double L_def = (2.0 * M_PI * R0) / (2.0 * nFODO);
    double Phi_def = L_def / R0;

    // Exact s-coordinate at the entry of each of the 8 elements within a FODO cell
    double cell_len_exact = 2.0*L_def + 4.0*driftLen + 2.0*quadLen;
    double elem_s_offset[8] = {
        0.0,
        L_def,
        L_def + driftLen,
        L_def + driftLen + quadLen,
        L_def + 2.0*driftLen + quadLen,
        2.0*L_def + 2.0*driftLen + quadLen,
        2.0*L_def + 3.0*driftLen + quadLen,
        2.0*L_def + 3.0*driftLen + 2.0*quadLen
    };

    // COD accumulators: sum x and y at each of the nFODO*8 element entries
    int     n_lat    = nFODO * 8;
    double* cod_x_sum = new double[n_lat]();   // zero-initialised
    double* cod_y_sum = new double[n_lat]();
    int*    cod_cnt   = new int[n_lat]();

    while (t < t_end) {
        int current_fodo = total_fodo_cells % nFODO;

        // Sequence internal to one FODO cell
        for (int elem = 0; elem < 8; ++elem) {
            if (t >= t_end) break;
            
            // RF flag execution
            if (current_fodo == 0 && elem == 0) {
                double phi_rf = omega_rf * t;
                while (phi_rf >= 2.0*M_PI) phi_rf -= 2.0*M_PI;
                while (phi_rf <  0.0     ) phi_rf += 2.0*M_PI;

                if (rf_on) {
                    double p_sq = y_init[3]*y_init[3] + y_init[4]*y_init[4] + y_init[5]*y_init[5];
                    double E_J  = std::sqrt(p_sq*C_LIGHT*C_LIGHT + M_P*M_P*C_LIGHT*C_LIGHT);
                    double beta = std::sqrt(p_sq) * C_LIGHT / E_J;
                    double dp   = Q_E * V_rf * std::sin(phi_rf) / (beta * C_LIGHT);
    
                    y_init[4] += dp * dir; 
                }

                if (rf_out.is_open()) {
                    double p_tang = y_init[4]; 
                    if (!have_ref) { p_tang_ref = p_tang; have_ref = true; }
                    double dp_over_p = (std::abs(p_tang_ref) > 1e-40) ? (p_tang - p_tang_ref) / p_tang_ref : 0.0;
                    rf_out << std::scientific << std::setprecision(16)
                           << t << "\t" << phi_rf << "\t" << dp_over_p << "\n";
                }
            }
            
            // Poincare trigger
            bool is_poincare_mark = false;
            if (target_quad < 0) {
                if (elem == 0) is_poincare_mark = true;
            } else {
                if ((target_quad % 2 == 0) && elem == 0) {
                    if (current_fodo == (target_quad / 2)) is_poincare_mark = true;
                } else if ((target_quad % 2 == 1) && elem == 4) {
                    if (current_fodo == (target_quad / 2)) is_poincare_mark = true;
                }
            }
            if (is_poincare_mark && p_saved < max_poincare) {
                poincare_t[p_saved] = t;
                for (int i = 0; i < dim; ++i) poincare_out[p_saved*dim + i] = y_init[i];
                double true_x = 0.0;
                if (elem == 0 || elem == 4) {
                    double R = std::sqrt(y_init[0]*y_init[0] + y_init[1]*y_init[1]);
                    true_x = R - R0;
                } else {
                    true_x = y_init[0] - R0;
                }
                poincare_out[p_saved*dim + 0] = true_x + R0;
                poincare_out[p_saved*dim + 1] = global_S;
                p_saved++;
            }

            // Accumulate element-entry position for COD (skip first turn)
            if (total_fodo_cells >= nFODO) {
                int idx = current_fodo * 8 + elem;
                cod_x_sum[idx] += (y_init[0] - R0) * 1000.0;  // mm
                cod_y_sum[idx] += y_init[2] * 1000.0;
                cod_cnt[idx]++;
            }

            int type = 0;
            double target_val = 0;
            if (elem == 0 || elem == 4) { type = 0; target_val = Phi_def; }
            else if (elem == 1 || elem == 3 || elem == 5 || elem == 7) { type = 1; target_val = driftLen; }
            else if (elem == 2) { type = (current_fodo == 0) ? 4 : 2; target_val = quadLen; }
            else if (elem == 6) { type = 3; target_val = quadLen; }

            double start_metric = (type == 0) ? std::atan2(y_init[1], y_init[0]) : y_init[1];
            double accumulated = 0.0;

            while (accumulated < target_val && t < t_end) {
                double h_step = h;
                
                double p_sq = y_init[3]*y_init[3] + y_init[4]*y_init[4] + y_init[5]*y_init[5];
                double gam = std::sqrt(1.0 + p_sq / (M_P * M_P * C_LIGHT * C_LIGHT));
                double vx = y_init[3]/(gam*M_P);
                double vy = y_init[4]/(gam*M_P);
                
                double val_rate = 0.0;
                if (type == 0) {
                    double R2 = y_init[0]*y_init[0] + y_init[1]*y_init[1];
                    val_rate = (y_init[0]*vy - y_init[1]*vx) / R2; 
                } else {
                    val_rate = vy;
                }
                
                // fractionally approach bound
                bool break_after = false;
                if (std::abs(val_rate) > 1e-12) {
                    double time_remaining = (target_val - accumulated) / std::abs(val_rate);
                    if (time_remaining <= h_step) {
                        h_step = time_remaining;
                        if (h_step < 0.0) h_step = 0.0;
                        break_after = true;
                    }
                }
                if (t + h_step > t_end) {
                    h_step = t_end - t;
                    break_after = true;
                }
                
                double old_metric = (type == 0) ? std::atan2(y_init[1], y_init[0]) : y_init[1];
                
                gl4_step_element(t, y_init, field_params, type, h_step, dim);
                t += h_step;
                global_step++;
                global_S += vy * h_step;

                if (global_step % print_interval == 0) {
                    int pct = (int)std::round(t * 100.0 / t_end);
                    std::printf("  t = %.4f ms  |  %%%d\n", t*1000.0, pct);
                    std::fflush(stdout);
                }

                if (global_step % save_interval == 0 && save_idx < return_steps) {
                    for (int i = 0; i < dim; ++i) history_out[save_idx*dim + i] = y_init[i];
                double true_x = 0.0;
                if (elem == 0 || elem == 4) {
                    double R = std::sqrt(y_init[0]*y_init[0] + y_init[1]*y_init[1]);
                    true_x = R - R0;
                } else {
                    true_x = y_init[0] - R0;
                }
                history_out[save_idx*dim + 0] = true_x + R0;
                    history_out[save_idx*dim + 1] = global_S;
                    save_idx++;
                }

                double new_metric = (type == 0) ? std::atan2(y_init[1], y_init[0]) : y_init[1];
                if (type == 0) {
                    double dth = new_metric - old_metric;
                    while(dth < -M_PI) dth += 2.0*M_PI;
                    while(dth >  M_PI) dth -= 2.0*M_PI;
                    accumulated += std::abs(dth);
                } else {
                    accumulated += std::abs(new_metric - old_metric);
                }
                
                if (break_after) break;
                if (target_val - accumulated <= 1e-11) break;
            }
            
            // EXACT physical transformation to place origin of next element
            if (type == 0) {
                // We rotated geometrically by Phi_def exactly
                double sign = (y_init[1] >= 0) ? 1.0 : -1.0;
                rotate_all(y_init, -sign * Phi_def);
            } else {
                // We translated geometrically by the length exactly
                double sign = (y_init[1] >= 0) ? 1.0 : -1.0;
                y_init[1] -= sign * target_val;
            }
        }
        total_fodo_cells++;
    }
    
    // Write per-element averaged COD to file
    {
        FILE* cod_file = std::fopen("cod_data.txt", "w");
        if (cod_file) {
            std::fprintf(cod_file, "s_m\tx_mm\ty_mm\n");
            for (int k = 0; k < nFODO; ++k) {
                for (int e = 0; e < 8; ++e) {
                    int idx = k * 8 + e;
                    double s_here = k * cell_len_exact + elem_s_offset[e];
                    double avg_x = (cod_cnt[idx] > 0) ? cod_x_sum[idx] / cod_cnt[idx] : 0.0;
                    double avg_y = (cod_cnt[idx] > 0) ? cod_y_sum[idx] / cod_cnt[idx] : 0.0;
                    std::fprintf(cod_file, "%.6f\t%.6f\t%.6f\n", s_here, avg_x, avg_y);
                }
            }
            std::fclose(cod_file);
        }
    }
    delete[] cod_x_sum;
    delete[] cod_y_sum;
    delete[] cod_cnt;

    printf("target_quad: %d, p_saved: %d\n", target_quad, p_saved);
    poincare_count[0] = p_saved;
    std::printf("  t = %.4f ms  |  %%100\n", t * 1000.0);
    std::fflush(stdout);
}

} // extern "C"
