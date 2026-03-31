# Quick test: what should dtheta be?
# In global coords: X = R*cos(θ), Y = R*sin(θ), Z = z
# Velocities: dX/dt = -R*sin(θ)*dθ/dt + dR/dt*cos(θ)
#             dY/dt =  R*cos(θ)*dθ/dt + dR/dt*sin(θ)
#             dZ/dt = dzdt
# Momenta: Px ~ dX/dt, Py ~ dY/dt, Pz ~ dZ/dt

# For motion mostly in z direction (longitudinal): 
# dθ/dt = (Px*sin(θ) - Py*cos(θ)) / R = tangential momentum / R

# For a particle going around the ring:
# Longitudinal momentum pz_local drives motion in z_local which wraps around
# Since z_local = R0*θ + small offset, we have:
# dz_local/dt = R0 * dθ/dt
# So: dθ/dt = (1/R0) * dz_local/dt = (1/R0) * pz_local/γ

import numpy as np

# Magic particle parameters
M2 = 0.938272046  # GeV/c^2
AMU = 1.792847356
p_magic = M2 / np.sqrt(AMU)
E_tot = np.sqrt(p_magic**2 + M2**2)
beta = p_magic / E_tot

print(f"p_magic = {p_magic:.6f} GeV/c")
print(f"E_tot = {E_tot:.6f} GeV")
print(f"beta = {beta:.6f}")
print(f"gamma = {1/np.sqrt(1-beta**2):.6f}")

# Circumference
R0 = 95.49
nFODO = 24
quadLen = 0.4
driftLen = 2.0
arc_circ = 2 * np.pi * R0
straight = nFODO * 2 * (quadLen + driftLen)
total_circ = arc_circ + straight

print(f"\nCircumference = {total_circ:.2f} m")

# Time for one revolution
T_rev = total_circ / (beta * 299792458)
print(f"T_rev = {T_rev*1e6:.2f} µs")

# Expected dθ/dt for going around once per revolution
# θ spans 2π in time T_rev
dtheta_dt_expected = 2 * np.pi / T_rev
print(f"Expected dθ/dt = {dtheta_dt_expected:.4f} rad/s")

# So in a timestep dt = 1e-11 s:
dt = 1e-11
dtheta_expected = dtheta_dt_expected * dt
print(f"Expected dθ per dt={dt:.1e}s = {dtheta_expected:.4e} rad")

# This dθ should come from: dθ = (pz_local / R0) * dt (approximately, for nearly on-axis)
# Or: dθ = (pz_global_component / R0 / p_mag) * dt if using global coords
