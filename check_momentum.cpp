// Test: initial state'de momentum values nedir?
#include <iostream>
#include <cmath>
#include <ctypes>

// From integrator.py initial conditions:
// Initial local: [x=0, y=0.00345, z=0, px=0, py=0, pz~p_mag*beta*gamma*cos_theta]
// theta = z/R0 = 0
// So: cos(theta) = 1, sin(theta) = 0
// X_G = R0*cos(0) = R0 = 95.49
// Y_G = R0*sin(0) = 0
// Px_g = px*1 - pz*0 = 0
// Py_g = px*0 + pz*1 = pz
// Pz_g = py = 0

// So initial: [X=95.49, Y=0, Z=0.00345, Px=0, Py=pz_local, Pz=0]

double R0 = 95.49;
double M2 = 0.938272046;
double AMU = 1.792847356;
double p_magic = M2 / std::sqrt(AMU);
double E_tot = std::sqrt(p_magic*p_magic + M2*M2);
double beta = p_magic / E_tot;
double gamma = 1.0 / std::sqrt(1 - beta*beta);
double p_mag = gamma * 0.938272046 / 299792458 * 1e9;  // Convert to SI

std::cout << "p_magic = " << p_magic << " GeV/c\n";
std::cout << "E_tot = " << E_tot << " GeV\n";
std::cout << "beta = " << beta << "\n";
std::cout << "gamma = " << gamma << "\n";
std::cout << "p_mag (magnitude) ~= " << p_mag << "\n";

// Now at initial conditions:
double X = R0, Y = 0;
double Px = 0, Py = p_mag * beta * gamma;  // approximately pz_local
double theta = std::atan2(Y, X);  // = 0
double R = std::sqrt(X*X + Y*Y);  // = 95.49

double p_tang = -Px * std::sin(theta) + Py * std::cos(theta);  // = 0 - Py*0 + Py*1 = Py
double p_total = std::sqrt(Px*Px + Py*Py + Py*Py);  // ≈ sqrt(2)*Py

std::cout << "\nInitial state:\n";
std::cout << "X = " << X << ", Y = " << Y << "\n";
std::cout << "theta = " << theta << "\n";
std::cout << "R = " << R << "\n";
std::cout << "Px = " << Px << ", Py = " << Py << "\n";
std::cout << "p_tang = " << p_tang << "\n";
std::cout << "p_total = " << p_total << "\n";

double C_LIGHT = 299792458;
double dtheta_dt = (p_tang / R) * (beta * C_LIGHT / p_total);
std::cout << "\ndtheta/dt = " << dtheta_dt << " rad/s\n";
std::cout << "dtheta per h=1e-11 = " << dtheta_dt * 1e-11 << " rad\n";

double dt_for_one_fodo = (2 * M_PI / 24) / dtheta_dt;
std::cout << "Time to advance 1 FODO cell = " << dt_for_one_fodo << " s\n";
