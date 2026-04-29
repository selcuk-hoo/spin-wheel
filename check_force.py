import numpy as np

# C++ constants
C_LIGHT = 299792458.0
M_P     = 1.672621777e-27
Q_E     = 1.602176565e-19
G_P     = 1.792847356
R0      = 95.49

# Python params calculation
M2 = 0.938272046
p_magic_base = M2 / np.sqrt(G_P)
E_tot = np.sqrt(p_magic_base**2 + M2**2)
E0_V_m = - (p_magic_base * (p_magic_base / E_tot) / R0) * 1e9

# Python initial p passed to C++
beta0 = p_magic_base / E_tot
gamma0 = 1.0 / np.sqrt(1.0 - beta0**2)
p_mag = gamma0 * M_P * C_LIGHT * beta0

# C++ variables
p = p_mag
p_sq = p**2
gamma = np.sqrt(1.0 + p_sq / (M_P**2 * C_LIGHT**2))
v = p / (gamma * M_P)

# Forces
F_c = p * v / R0
F_E = Q_E * abs(E0_V_m)

print(f"Centripetal Force required: {F_c} N")
print(f"Electric Force provided:    {F_E} N")
print(f"Relative mismatch:          {(F_E - F_c) / F_c}")

# Calculate displacement
# Displacement dx = D_x * dp/p. 
# Or directly from force mismatch: dF/F = dx / R0 ?
# Actually F = m gamma v^2 / R. If F_E is slightly different, equilibrium R is different.
# For cylindrical capacitor, E(R) = E0 * R0 / R.
# F_E(R) = Q_E * E0 * R0 / R.
# F_c(R) = p * v / R.
# Since both scale as 1/R, if they don't match exactly at R0, they don't match ANYWHERE!
# Wait! This means there is NO equilibrium orbit for a pure inverse-R field if the momentum is mismatched!
