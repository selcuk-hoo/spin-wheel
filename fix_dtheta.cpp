// Correct calculation:
// State vector in integrator is GLOBAL: [X, Y, Z, Px_g, Py_g, Pz_g, ...]
// where:
//   X = (R0 + x_local)*cos(theta_local)
//   Y = (R0 + x_local)*sin(theta_local)  
//   Z = y_local
//   Px_g = px_local*cos(theta) - pz_local*sin(theta)
//   Py_g = px_local*sin(theta) + pz_local*cos(theta)
//   Pz_g = py_local
//
// Tangential velocity: v_tang = (-Px_g*sin(θ) + Py_g*cos(θ)) / (γ*M)
// But (-Px_g*sin(θ) + Py_g*cos(θ)) = pz_local (from the transformation)
// So: dθ/dt = pz_local / (γ * M1 * R0)
//
// In global frame:
// X = R*cos(θ), Y = R*sin(θ) 
// Px_g ~ -R*sin(θ)*ω + dR/dt*cos(θ)
// Py_g ~ R*cos(θ)*ω + dR/dt*sin(θ)
// Where ω = dθ/dt
//
// Solving: (-Px_g*sin(θ) + Py_g*cos(θ)) = R*ω
// So: ω = dθ/dt = (-Px_g*sin(θ) + Py_g*cos(θ)) / R

double X = y_init[0];
double Y = y_init[1];
double Px_g = y_init[3];
double Py_g = y_init[4];

double theta = std::atan2(Y, X);
double R = std::sqrt(X*X + Y*Y);

// Tangential momentum component
double p_tang = -Px_g * std::sin(theta) + Py_g * std::cos(theta);

// Angular velocity: dθ/dt = p_tang / (R * γ * M)
// For highly relativistic: p ≈ E/c, so dθ/dt ≈ p_tang * c / (R * E)
// But we need to account for mass. Using γ*M*c ≈ E:
// dθ/dt = p_tang / (R * γ * M1 / c²) = p_tang * c / (R * E_kinetic)
// Simpler: dθ/dt = p_tang * c / (R * m_gamma) where m_gamma = γ*M

// For the magic momentum particle:
double E_total = 1.171064;  // GeV (from earlier calc)
double m_gamma = E_total / (299792458 * 299792458) * 1e9;  // Convert to SI units

// Actually simpler: use that the particle moves with velocity v ≈ β*c
// And tangential component of momentum is pz_local from coordinate transform
// Which we can get back, but it's easier to just use:
// dθ/dt = p_tang / (R * p_mag) * (β*c / mass_energy_relation)

// Most direct: use the fact that in curvilinear coords:
// dθ/dt = v_tangential / R = (p_tangential / p_total) * (v_total / R)
// where v_total ≈ β*c for relativistic particle

double p_total = std::sqrt(y_init[3]*y_init[3] + y_init[4]*y_init[4] + y_init[5]*y_init[5]);
double dtheta_dt = (p_tang / R) * (beta_ideal * C_LIGHT) / p_total;

double dtheta = dtheta_dt * h;
