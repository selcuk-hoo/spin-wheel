// integrator.cpp — proton EDM storage-ring particle and spin tracker
//
// Coordinate system (local rotating frame):
//   X = radial (outward from ring centre)
//   Y = azimuthal / tangential (along the nominal orbit)
//   Z = vertical (out of the median plane)
//
// After each lattice element the frame is reset so the particle enters the
// next element with Y ≈ 0 and X ≈ R0 ("rotating-frame" convention).
//
// FODO cell layout (8 elements, repeated nFODO times):
//   elem 0: ARC1   (deflector, angle Phi_def)
//   elem 1: DRIFT
//   elem 2: QF     (focusing quad; QF_MOD for cell 0 if modulated)
//   elem 3: DRIFT
//   elem 4: ARC2   (deflector, angle Phi_def)
//   elem 5: DRIFT
//   elem 6: QD     (defocusing quad)
//   elem 7: DRIFT
//
// Integrator: 4th-order implicit Gauss-Legendre (GL4), symplectic,
//             4 fixed-point iterations per step.
// Spin:       Thomas-BMT equation, optional EDM term.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdio>

extern "C" {

// Physical constants (SI)
const double C_LIGHT = 299792458.0;       // speed of light [m/s]
const double M_P     = 1.672621777e-27;   // proton rest mass [kg]
const double Q_E     = 1.602176565e-19;   // proton charge [C]
const double G_P     = 1.792847356;       // proton anomalous magnetic moment G = (g-2)/2
const double EDM_ETA = 1.88e-15;          // proton EDM sensitivity parameter η

inline void cross_product(const double* a, const double* b, double* res) {
    res[0] = a[1]*b[2] - a[2]*b[1];
    res[1] = a[2]*b[0] - a[0]*b[2];
    res[2] = a[0]*b[1] - a[1]*b[0];
}

inline double dot_product(const double* a, const double* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

// Rotate position, momentum and spin vectors together around the Z-axis by
// angle theta.  Called after each arc element with theta = -sign*Phi_def to
// undo the angular advance of the orbit, keeping X ≈ R0 at every element
// entry (the rotating-frame convention).
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

// Compute E and B at position r for the given element type.
//
// field_params layout (indices):
//   [0] R0       ring radius [m]
//   [1] E0       electric field at R0 [V/m]
//   [2] n        field index (E_r ∝ (R0/R)^n)
//   [3] B0ver    uniform vertical B [T]
//   [4] B0rad    uniform radial B [T]
//   [5] B0long   uniform longitudinal B [T]
//   [6] quadK1   quadrupole gradient K1 [T/m]
//   [7] sextK1   sextupole strength K2 [T/m²]
//   [9] sextSwitch   1 = sextupole on
//   [19] quadModA    modulation amplitude (type 4 only)
//   [20] quadModF    modulation frequency [Hz] (type 4 only)
//   [21] nFODO_off   apply quad offset at elem=2 of this cell (0-based), <0 disables
//   [22] B0hor       equivalent horizontal field [T] for offset formula y_off = B0hor / K1
//   [23] quadYOffset per-element vertical quad centre shift [m] (internal runtime override)
//   [24] engeSwitch  0 = hard-edge, 1 = Enge soft-edge fringe
//   [25] engeA       Enge polynomial coefficient a0 (default 2.0)
//   [26] engeG       electrode gap [m] (default 0.05)
//   [27] engeNorm    orbit-closure normalization (computed by run_integration)
//
// element_type: 0 = DEFLECTOR, 1 = DRIFT, 2 = QUAD_F, 3 = QUAD_D, 4 = QUAD_F_MOD
void get_electromagnetic_fields(double t, const double* r, const double* field_params, int element_type, double* E, double* B) {
    double R0       = field_params[0];
    double E0       = field_params[1];
    double n        = field_params[2];
    double B0ver    = field_params[3];
    double B0rad    = field_params[4];
    double B0long   = field_params[5];
    double quadK1   = field_params[6];
    double sextK1   = field_params[7];
    double quadYOffset = field_params[23];
    
    double X = r[0], Y = r[1], Z = r[2];
    double R = std::sqrt(X*X + Y*Y);

    E[0] = 0.0; E[1] = 0.0; E[2] = 0.0;
    B[0] = 0.0; B[1] = 0.0; B[2] = 0.0;

    if (element_type == 0) {
        // DEFLECTOR: cylindrical capacitor, hard-edge (no fringe fields).
        //
        // Radial electric field with Z-expansion ensuring ∇·E = 0:
        //   E_r(R,Z) = E0*(R0/R)^n * [1 - (n²-1)/2*(Z/R)² + ...]
        //   E_Z(R,Z) = E0*(R0/R)^n * [(n-1)*(Z/R) - ...]
        // For n=1 the series collapses to E_r = E0*R0/R, E_Z = 0 exactly.
        //
        // Stray B fields (B0ver, B0rad, B0long) are included for COD studies.
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

        // Enge soft-edge fringe: scale E only (electric deflector, not magnetic).
        // F(φ) = F_entrance(φ) · F_exit(φ) · engeNorm
        // where F_entrance = sigmoid(a·φ/φ_g), F_exit = sigmoid(a·φ_remain/φ_g)
        // and φ_g = engeG/R0 converts the gap length to an angle.
        if (field_params[24] > 0.5 && R > 1e-6) {
            double engeA    = field_params[25];
            double engeG    = field_params[26];
            double engeNorm = field_params[27];
            double phi_g    = engeG / R0;
            int    nFODO_e  = (int)(field_params[12] + 0.5);
            double Phi_def_e = M_PI / nFODO_e;
            double phi_prog  = std::abs(std::atan2(Y, X));
            double phi_rem   = Phi_def_e - phi_prog;
            double F_ent = 1.0 / (1.0 + std::exp(-engeA * phi_prog / phi_g));
            double F_ext = 1.0 / (1.0 + std::exp(-engeA * phi_rem  / phi_g));
            double F = F_ent * F_ext * engeNorm;
            E[0] *= F;
            E[1] *= F;
            E[2] *= F;
        }

        // Stray B projected onto (radial, tangential, vertical) unit vectors
        B[0] = -B0rad * cos_th + B0long * sin_th;
        B[1] = -B0rad * sin_th - B0long * cos_th;
        B[2] = B0ver;
    } else if (element_type == 2 || element_type == 3) {
        // QUADRUPOLE (QF or QD, normal — no time modulation).
        //
        // dev_quad = X - R0: radial deviation from the quad magnetic centre.
        // Pure quadrupole (satisfies ∇·B = 0 and ∇×B = 0):
        //   B_r =  K1 * Z
        //   B_Z =  K1 * dev
        // QF (type 2): K1 > 0 → horizontally focusing, vertically defocusing.
        // QD (type 3): K1 → -K1 (sign flip).
        double current_K1 = (element_type == 2) ? quadK1 : -quadK1;
        double dev_quad = X - R0;

        double y_rel = Z - quadYOffset;
        double B_quad_r = current_K1 * y_rel;
        double B_quad_z = current_K1 * dev_quad;

        // Optional sextupole overlay.  Maxwell's ∇·B = 0 requires:
        //   B_r =  K2 * dev * Z
        //   B_Z = (K2/2) * (dev² - Z²)    ← factor 0.5 is mandatory
        // QF: K2 = +sextK1 ; QD: K2 = -sextK1
        double sextSwitch = field_params[9];
        if (sextSwitch > 0.0) {
            double current_sK1 = (element_type == 2) ? sextK1 : -sextK1;
            B_quad_r += current_sK1 * dev_quad * y_rel;
            B_quad_z += 0.5 * current_sK1 * (dev_quad*dev_quad - y_rel*y_rel);
        }

        B[0] = B_quad_r;
        B[1] = 0.0;     // no longitudinal field in an ideal quad
        B[2] = B_quad_z;
    } else if (element_type == 4) {
        // QUAD_F_MOD: focusing quad with time-modulated K1 (cell 0 only).
        // Used for parametric resonance studies.
        //   K1_eff(t) = K1 * (1 + A_mod * cos(2π * f_mod * t))
        double A_mod  = field_params[19];
        double f_mod  = field_params[20];
        double K1_eff = quadK1 * (1.0 + A_mod * std::cos(2.0 * M_PI * f_mod * t));

        double dev_quad = X - R0;
        double y_rel = Z - quadYOffset;
        double B_quad_r = K1_eff * y_rel;
        double B_quad_z = K1_eff * dev_quad;

        // Same Maxwell-correct sextupole overlay as in type 2/3
        double sextSwitch = field_params[9];
        if (sextSwitch > 0.0) {
            B_quad_r += sextK1 * dev_quad * y_rel;
            B_quad_z += 0.5 * sextK1 * (dev_quad*dev_quad - y_rel*y_rel);
        }

        B[0] = B_quad_r;
        B[1] = 0.0;
        B[2] = B_quad_z;
    }
}

// Equations of motion — right-hand side of the ODE system.
//
// State vector y[9]:
//   y[0..2] = (X, Y, Z)    position [m]
//   y[3..5] = (Px, Py, Pz) relativistic 3-momentum [kg·m/s]
//   y[6..8] = (Sx, Sy, Sz) spin unit vector (|S| = 1)
//
//   dr/dt = v = p / (γm)
//   dp/dt = q(E + v×B)                  [Lorentz force]
//   dS/dt = Ω × S                        [Thomas-BMT precession]
//
// Thomas-BMT angular velocity:
//   Ω = -(q/m){ [G + 1/γ]B - [Gγ/(γ+1)](β·B)β - [G+1/(γ+1)](β×E)/c }
//       + EDM term (if EDMSwitch = 1)
//
// At magic momentum (G = 1/γ²) the MDM term for an on-energy particle in
// a pure radial electric field vanishes → horizontal spin is "frozen".
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

// Single step of the 4th-order implicit Gauss-Legendre (GL4) symplectic
// integrator.
//
// GL4 Butcher tableau (2-stage implicit Runge-Kutta):
//   c1 = 1/2 - √3/6    a11 = 1/4          a12 = 1/4 - √3/6
//   c2 = 1/2 + √3/6    a21 = 1/4 + √3/6   a22 = 1/4
//                       b1  = 1/2          b2  = 1/2
//
// The implicit stage equations are solved by 4 fixed-point iterations,
// starting from the explicit Euler derivative as the initial guess.
//
// GL4 preserves symplecticity: phase-space volume is conserved and |S| = 1
// is maintained to machine precision over millions of turns.
//
// Step size h is NOT adaptive — it equals dt from params.json for every step
// except the last step within each lattice element, which is shortened to
// land exactly on the element boundary.
void gl4_step_element(double t, double* y, const double* field_params, int element_type, double h, int dim) {
    // GL4 nodes and coupling coefficients
    const double sq3 = std::sqrt(3.0) / 6.0;
    const double c1  = 0.5 - sq3, c2 = 0.5 + sq3;
    const double a11 = 0.25, a12 = 0.25 - sq3;
    const double a21 = 0.25 + sq3, a22 = 0.25;

    double k1[9], k2[9], y1[9], y2[9];

    // Zeroth iterate: explicit Euler derivative at current point
    compute_rhs(t, y, field_params, element_type, k1, dim);
    for (int i = 0; i < dim; ++i) k2[i] = k1[i];

    // Fixed-point iterations to converge the implicit stage values
    for (int iter = 0; iter < 4; ++iter) {
        for (int i = 0; i < dim; ++i) {
            y1[i] = y[i] + h * (a11*k1[i] + a12*k2[i]);
            y2[i] = y[i] + h * (a21*k1[i] + a22*k2[i]);
        }
        compute_rhs(t + c1*h, y1, field_params, element_type, k1, dim);
        compute_rhs(t + c2*h, y2, field_params, element_type, k2, dim);
    }

    // Final update with equal weights b1 = b2 = 1/2
    for (int i = 0; i < dim; ++i)
        y[i] += h * (0.5*k1[i] + 0.5*k2[i]);
}

// Main tracking loop — traverses the FODO lattice element by element.
//
// For each element:
//   1. Integrate with GL4 until the progress variable reaches its target.
//   2. Apply frame-reset so the particle enters the next element at Y ≈ 0.
//
// Progress variables (how element traversal is measured):
//   Arc (type 0):     φ = atan2(Y,X),  rate dφ/dt = (X·vy - Y·vx) / R²
//   Straight (other): Y coordinate,    rate dY/dt  = vy
//
// Frame reset after each element:
//   Arc:      rotate_all(y, -sign*Phi_def)   — undoes the arc rotation
//   Straight: y[1] -= sign * target_len      — shifts Y back to zero
//   sign = +1 if Y ≥ 0, −1 if Y < 0
//
// COD (Closed-Orbit Distortion):
//   Samples x = X - R0 and Z at every element entry from turn 2 onward.
//   Turn-averaged values are written to cod_data.txt at the end.
//
// Poincaré sections:
//   Recorded at the chosen quadrupole (target_quad ≥ 0) or every cell
//   entry (target_quad < 0).
//
// RF cavity (thin kick, once per turn at cell-0 arc entry):
//   Δpy = q·V_RF·sin(φ_RF) / (β·c),  φ_RF = ω_RF·t
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
    int    nFODO_off = (int)std::round(field_params[21]);
    double B0hor = field_params[22];

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

    int total_fodo_cells = 0;  // total elements / 8; div nFODO gives current cell index
    double global_S = 0.0;    // accumulated arc-length along the nominal orbit [m]

    // Half-cell arc extent: each FODO cell has 2 arcs of Phi_def each
    double L_def = (2.0 * M_PI * R0) / (2.0 * nFODO);  // arc length per half-cell [m]
    double Phi_def = L_def / R0;                          // arc angle per half-cell [rad]

    // Enge normalization: compute once so ∫₀^{Phi_def} F_ent·F_exit dφ = Phi_def
    double enge_norm = 1.0;
    if (field_params[24] > 0.5) {
        double engeA = field_params[25];
        double engeG = field_params[26];
        double phi_g = engeG / R0;
        if (phi_g > 1e-12) {
            const int nint = 2000;
            double dp = Phi_def / nint, sum = 0.0;
            for (int i = 0; i < nint; i++) {
                double phi = (i + 0.5) * dp;
                double fe = 1.0 / (1.0 + std::exp(-engeA * phi / phi_g));
                double fx = 1.0 / (1.0 + std::exp(-engeA * (Phi_def - phi) / phi_g));
                sum += fe * fx;
            }
            double integral = sum * dp;
            if (integral > 0.0) enge_norm = Phi_def / integral;
        }
        std::printf("[Enge] a=%.2f  g=%.4f m  phi_g=%.4e rad  norm=%.6f\n",
                    engeA, engeG, phi_g, enge_norm);
        std::fflush(stdout);
    }

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

    // Staging buffer: filled during each revolution, committed only when the
    // revolution completes.  Partial last revolution is silently discarded,
    // ensuring cod_cnt[i] is identical for all lattice positions.
    double* stage_x = new double[n_lat]();
    double* stage_y = new double[n_lat]();
    bool past_first_rev = false;

    while (t < t_end) {
        int current_fodo = total_fodo_cells % nFODO;  // cell index within current revolution

        // Start each revolution from the first drift element (elem=1),
        // then wrap around: 1,2,3,4,5,6,7,0.
        const int start_elem = 1;
        for (int elem_iter = 0; elem_iter < 8; ++elem_iter) {
            int elem = (start_elem + elem_iter) % 8;
            if (t >= t_end) break;

            // ---- RF thin kick (once per revolution at cell-0 arc entry) ----
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

            // ---- Poincaré section trigger ----
            // target_quad < 0 → every cell-0 arc entry (full-turn section)
            // target_quad ≥ 0 → specific quad: even=QF(elem2), odd=QD(elem6)
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

            // ---- COD: stage element-entry position for this revolution ----
            // The staging buffer is committed to cod_sum only at revolution end.
            // This guarantees cod_cnt[i] is uniform across all lattice positions,
            // so s=0 and s=circumference carry the same number of averaged turns.
            if (past_first_rev) {
                int idx = current_fodo * 8 + elem;
                double cod_x = 0.0;
                if (elem == 0 || elem == 4) {
                    double R = std::sqrt(y_init[0]*y_init[0] + y_init[1]*y_init[1]);
                    cod_x = R - R0;
                } else {
                    cod_x = y_init[0] - R0;
                }
                stage_x[idx] = cod_x * 1000.0;  // mm (overwrite — one visit per rev)
                stage_y[idx] = y_init[2] * 1000.0;
            }

            int type = 0;
            double target_val = 0;
            if (elem == 0 || elem == 4) { type = 0; target_val = Phi_def; }
            else if (elem == 1 || elem == 3 || elem == 5 || elem == 7) { type = 1; target_val = driftLen; }
            else if (elem == 2) { type = (current_fodo == 0) ? 4 : 2; target_val = quadLen; }
            else if (elem == 6) { type = 3; target_val = quadLen; }

            double field_params_local[28];
            for (int fp = 0; fp < 28; ++fp) field_params_local[fp] = field_params[fp];
            field_params_local[23] = 0.0;
            bool is_target_first_quad = (elem == 2) && (nFODO_off >= 0) && (current_fodo == nFODO_off);
            if (is_target_first_quad && std::abs(field_params[6]) > 1e-20) {
                field_params_local[23] = B0hor / field_params[6];
            }
            field_params_local[27] = enge_norm;

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
                
                // Shorten the last step to land exactly on the element boundary
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
                
                gl4_step_element(t, y_init, field_params_local, type, h_step, dim);
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
            
            // ---- Frame reset: place origin at start of next element ----
            if (type == 0) {
                // Arc: rotate by the actual current angle atan2(Y,X) so that Y
                // becomes exactly zero regardless of any small overshoot or
                // undershoot in `accumulated` relative to Phi_def.
                // Using the nominal Phi_def would leave a residual Y ∝ (dev/R0)*Phi_def
                // that accumulates turn-by-turn and drives spurious oscillations.
                double actual_angle = std::atan2(y_init[1], y_init[0]);
                rotate_all(y_init, -actual_angle);
            } else {
                // Straight: zero Y exactly, analogous to the arc atan2-based reset.
                // Drift and quad fields depend only on X and Z, not on Y, so
                // translating the frame origin to the actual exit Y is exact.
                // Subtracting the nominal target_val would leave a residual
                // proportional to the last-step overshoot.
                y_init[1] = 0.0;
            }
        }
        total_fodo_cells++;

        // Commit staged COD data only when a complete revolution just finished.
        // Partial last revolution stays in stage_x/y and is simply discarded.
        if (total_fodo_cells % nFODO == 0) {
            if (!past_first_rev) {
                past_first_rev = true;  // first revolution done; begin staging next
            } else {
                for (int i = 0; i < n_lat; i++) {
                    cod_x_sum[i] += stage_x[i];
                    cod_y_sum[i] += stage_y[i];
                    cod_cnt[i]++;  // same increment for every lattice position
                }
            }
        }
    }

    delete[] stage_x;
    delete[] stage_y;

    // ---- Write turn-averaged COD to cod_data.txt ----
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
