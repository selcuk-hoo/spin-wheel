#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

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

void get_electromagnetic_fields(double t, const double* r, const double* field_params, double* E, double* B) {
    double R0       = field_params[0];
    double E0       = field_params[1];
    double n        = field_params[2];
    double B0ver    = field_params[3];
    double B0rad    = field_params[4];
    double B0long   = field_params[5];
    double quadK1   = field_params[6];
    double sextK1   = field_params[7];
    double quadSwitch = field_params[8];
    double sextSwitch = field_params[9];
    int    nFODO    = (int)(field_params[12] + 0.5);
    double quadLen  = field_params[13];

    double X_g = r[0], Y_g = r[1], Z_g = r[2];
    double R   = std::sqrt(X_g*X_g + Y_g*Y_g);
    double dev = R - R0;

    // cos/sin doğrudan X/R, Y/R — atan2+cos/sin çağrısından kaçınılır
    double cos_th = X_g / R;
    double sin_th = Y_g / R;

    double E_r = 0.0, E_z = 0.0;
    if (R > 1e-6) {
        if (n == 1.0) {
            E_r = E0 * R0 / R;
        } else {
            double zR  = Z_g / R;
            double zR2 = zR * zR;
            double pow_n = std::pow(R0/R, n);
            E_r = E0 * pow_n * (1.0 - 0.5*(n*n-1.0)*zR2
                + (n*n-1.0)*(n+1.0)*(n+3.0)*zR2*zR2/24.0);
            E_z = E0 * pow_n * ((n-1.0)*zR
                - (n*n-1.0)*(n+1.0)*zR2*zR/6.0);
        }
    }
    E[0] = E_r * cos_th;
    E[1] = E_r * sin_th;
    E[2] = E_z;

    B[0] = -B0rad * cos_th + B0long * sin_th;
    B[1] = -B0rad * sin_th - B0long * cos_th;
    B[2] = B0ver;

    if ((quadSwitch > 0.0 || sextSwitch > 0.0) && nFODO > 0 && quadLen > 0.0) {
        // Quad sektörü tespiti için atan2 sadece burada hesaplanır
        double theta_pos = std::atan2(Y_g, X_g);
        if (theta_pos < 0) theta_pos += 2.0 * M_PI;

        double sector_angle = M_PI / nFODO;
        int quad_index = (int)(theta_pos / sector_angle + 0.5);
        if (quad_index >= 2 * nFODO) quad_index = 0;

        double quad_center_theta = quad_index * sector_angle;
        double diff_theta = theta_pos - quad_center_theta;
        while (diff_theta >  M_PI) diff_theta -= 2.0 * M_PI;
        while (diff_theta < -M_PI) diff_theta += 2.0 * M_PI;

        double dist_from_center = R0 * std::abs(diff_theta);

        if (dist_from_center <= quadLen / 2.0) {
            if (quadSwitch > 0.0) {
                double current_K1 = ((quad_index % 2) == 0) ? quadK1 : -quadK1;
                double B_quad_r   = current_K1 * Z_g;
                double B_quad_z   = current_K1 * dev;
                B[0] += B_quad_r * cos_th;
                B[1] += B_quad_r * sin_th;
                B[2] += B_quad_z;
            }
            if (sextSwitch > 0.0) {
                double current_sK1 = ((quad_index % 2) == 0) ? sextK1 : -sextK1;
                double B_sext_r    = current_sK1 * dev * Z_g;
                double B_sext_z    = current_sK1 * (dev*dev - Z_g*Z_g);
                B[0] += B_sext_r * cos_th;
                B[1] += B_sext_r * sin_th;
                B[2] += B_sext_z;
            }
        }
    }
}

void compute_rhs(double t, const double* y, const double* field_params, double* dydt, int dim) {
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
    get_electromagnetic_fields(t, r, field_params, E, B);

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

void gl4_step(double t, double* y, const double* field_params, double h, int dim) {
    const double sq3 = std::sqrt(3.0) / 6.0;
    const double c1  = 0.5 - sq3, c2 = 0.5 + sq3;
    const double a11 = 0.25, a12 = 0.25 - sq3;
    const double a21 = 0.25 + sq3, a22 = 0.25;

    // dim her zaman 9 — heap allocation yerine stack dizileri
    double k1[9], k2[9], y1[9], y2[9];

    compute_rhs(t, y, field_params, k1, dim);
    for (int i = 0; i < dim; ++i) k2[i] = k1[i];

    for (int iter = 0; iter < 2; ++iter) {
        for (int i = 0; i < dim; ++i) {
            y1[i] = y[i] + h * (a11*k1[i] + a12*k2[i]);
            y2[i] = y[i] + h * (a21*k1[i] + a22*k2[i]);
        }
        compute_rhs(t + c1*h, y1, field_params, k1, dim);
        compute_rhs(t + c2*h, y2, field_params, k2, dim);
    }

    for (int i = 0; i < dim; ++i)
        y[i] += h * (0.5*k1[i] + 0.5*k2[i]);
}

// RF kovuğu quad indeks 0'da: theta_rf = 0 * (pi/nFODO) = 0
// Parçacık bu quad'a her girişinde anlık bir momentum kick'i alır.
static bool is_in_rf_quad(double theta, double R0, double quadLen) {
    // theta [0, 2pi) aralığında; RF merkezi theta_rf = 0
    // Wrap-around: theta = 2pi - epsilon da merkezin yakınında sayılır
    double d = theta;  // zaten [0,2pi) aralığında
    if (d > M_PI) d = 2.0*M_PI - d;  // |theta - 0| mod 2pi, kısa yol
    return (R0 * d <= quadLen / 2.0);
}

void run_integration(double* y_init, const double* field_params,
                     double t0, double t_end, double h, int dim,
                     int return_steps, double* history_out,
                     int max_poincare, double* poincare_out,
                     double* poincare_t, int* poincare_count) {
    long long total_steps = (long long)((t_end - t0) / h);
    if (total_steps <= 0) return;
    long long save_interval = total_steps / return_steps;
    if (save_interval == 0) save_interval = 1;

    double R0      = field_params[0];
    int    nFODO   = (int)(field_params[12] + 0.5);
    double quadLen = field_params[13];
    double dir     = field_params[11];

    int    target_quad  = (int)(field_params[14] + 0.5);
    double target_angle = target_quad * (M_PI / nFODO);

    const bool rf_on = (field_params[15] > 0.0);
    double V_rf      = field_params[16];
    double h_rf      = field_params[17];  // harmonik sayı

    double M_GeV   = 0.938272046;
    double p_magic = M_GeV / std::sqrt(G_P);               // GeV/c
    double E_magic = std::sqrt(p_magic*p_magic + M_GeV*M_GeV); // GeV
    double beta_magic = p_magic / E_magic;
    double circumference = 2.0 * M_PI * R0;
    double omega_rf = h_rf * 2.0 * M_PI * beta_magic * C_LIGHT / circumference;

    bool   prev_in_rf = false;
    double p_tang_ref  = 0.0;
    bool   have_ref    = false;

    std::ofstream rf_out;
    if (rf_on) {
        rf_out.open("rf.txt", std::ios::out | std::ios::trunc);
        if (rf_out.is_open())
            rf_out << "T_sec\tPhi_RF_rad\tdp_over_p\n";
    }

    double t           = t0;
    double prev_theta_uw = 0.0;
    int    save_idx    = 0;
    int    p_saved     = 0;

    for (long long step = 0; step < total_steps; ++step) {
        double old_X = y_init[0], old_Y = y_init[1];
        double old_theta = std::atan2(old_Y, old_X);
        if (old_theta < 0) old_theta += 2.0 * M_PI;

        gl4_step(t, y_init, field_params, h, dim);
        t += h;

        double cur_X = y_init[0], cur_Y = y_init[1];
        double cur_theta = std::atan2(cur_Y, cur_X);
        if (cur_theta < 0) cur_theta += 2.0 * M_PI;

        // Unwrapped theta (Poincaré için)
        double dtheta = cur_theta - old_theta;
        while (dtheta >  M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        double cur_theta_uw = prev_theta_uw + dtheta;

        // --- RF kick: quad indeks 0'a ilk girişte ---
        if (rf_on) {
            bool cur_in_rf = is_in_rf_quad(cur_theta, R0, quadLen);

            if (!prev_in_rf && cur_in_rf) {
                // RF faz
                double phi_rf = omega_rf * t;
                while (phi_rf >= 2.0*M_PI) phi_rf -= 2.0*M_PI;
                while (phi_rf <  0.0     ) phi_rf += 2.0*M_PI;

                // Teğetsel yön (dolaşım yönüne göre)
                double R      = std::sqrt(cur_X*cur_X + cur_Y*cur_Y);
                double cos_th = cur_X / R;
                double sin_th = cur_Y / R;

                // Momentum kick: dp = q*V*sin(phi) / (beta * c)
                double p_sq = y_init[3]*y_init[3] + y_init[4]*y_init[4] + y_init[5]*y_init[5];
                double mc   = M_P * C_LIGHT;
                double E_J  = std::sqrt(p_sq*C_LIGHT*C_LIGHT + mc*mc*C_LIGHT*C_LIGHT);
                double beta = std::sqrt(p_sq) * C_LIGHT / E_J;
                double dp   = Q_E * V_rf * std::sin(phi_rf) / (beta * C_LIGHT);

                y_init[3] += dp * dir * (-sin_th);
                y_init[4] += dp * dir * ( cos_th);

                // Kayıt
                if (rf_out.is_open()) {
                    double p_tang = -y_init[3]*sin_th + y_init[4]*cos_th;
                    if (!have_ref) { p_tang_ref = p_tang; have_ref = true; }
                    double dp_over_p = (std::abs(p_tang_ref) > 1e-40)
                                     ? (p_tang - p_tang_ref) / p_tang_ref : 0.0;
                    rf_out << std::scientific << std::setprecision(16)
                           << t << "\t" << phi_rf << "\t" << dp_over_p << "\n";
                }
            }
            prev_in_rf = cur_in_rf;
        }

        // --- Poincaré kesiti ---
        double p_rel = prev_theta_uw - target_angle;
        double c_rel = cur_theta_uw  - target_angle;
        if (std::floor(c_rel/(2.0*M_PI)) != std::floor(p_rel/(2.0*M_PI)) && p_saved < max_poincare) {
            poincare_t[p_saved] = t;
            for (int i = 0; i < dim; ++i)
                poincare_out[p_saved*dim + i] = y_init[i];
            p_saved++;
        }
        prev_theta_uw = cur_theta_uw;

        // --- Geçmiş kaydı ---
        if (step % save_interval == 0 && save_idx < return_steps) {
            for (int i = 0; i < dim; ++i)
                history_out[save_idx*dim + i] = y_init[i];
            save_idx++;
        }
    }
    poincare_count[0] = p_saved;
}

} // extern "C"
