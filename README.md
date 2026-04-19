# spin-wheel

6D spin dynamics simulation for a proton Electric Dipole Moment (EDM) storage ring. Tracks position, momentum, and spin simultaneously using a symplectic integrator, with a realistic FODO lattice including deflector arcs, straight drift sections, quadrupole/sextupole magnets, and an RF cavity.

## Physics

The experiment aims to measure the proton EDM by storing protons at **magic momentum** in an all-electric ring. At magic momentum, the horizontal spin component is frozen (no precession relative to the momentum), making any vertical spin buildup a signature of a non-zero EDM.

### Magic momentum

```
p_magic = M_p / sqrt(G_p)  ≈  0.7007 GeV/c
```

where `G_p = 1.792847356` is the proton anomalous magnetic moment and `M_p = 0.938272 GeV/c²`.

### Equations of motion

The particle state vector is `[X, Y, Z, Px, Py, Pz, Sx, Sy, Sz]` in the global Cartesian frame (ring plane = XY, vertical = Z).

**Orbital motion** (Lorentz force):
```
dp/dt = q(E + v×B)
dr/dt = v = p / (γ m)
```

**Spin precession** (Thomas-BMT equation):
```
dS/dt = Ω × S

Ω = -(q/m) [ (G + 1/γ)B  -  G·γ/(γ+1) (β·B)β  -  (G + 1/(γ+1))/c · (β×E) ]
```

An optional EDM term proportional to `η` (electric dipole coupling) can be switched on via `EDMSwitch`.

## Lattice

The ring has **nFODO = 24** FODO cells. Each cell contains 8 elements:

```
[ARC] [DRIFT] [QUAD_F] [DRIFT] [ARC] [DRIFT] [QUAD_D] [DRIFT]
```

| Element | Length | Description |
|---------|--------|-------------|
| ARC | π·R0 / nFODO | Deflector arc with radial electric field |
| DRIFT | 2.0833 m | Field-free straight section |
| QUAD_F | 0.4 m | Focusing quadrupole (+K1) |
| QUAD_D | 0.4 m | Defocusing quadrupole (−K1) |

Total circumference: `nFODO × (2·L_arc + 4·L_drift + 2·L_quad) ≈ 800 m`

Ring radius: `R0 = 95.49 m`

An **RF cavity** (thin-lens momentum kick) fires once per turn at the start of cell 0 for longitudinal phase-space confinement.

### Coordinate frame convention

The integrator works in the **global Cartesian frame**. At each arc exit, the coordinate frame is rotated by `Φ_def = π/nFODO` to reset the reference angle to zero, equivalent to a Frenet-Serret frame update. Drift and quad elements are then traversed with the particle moving along the local +Y axis.

## Integrator

**GL4** — 4th-order Gauss-Legendre implicit Runge-Kutta (symplectic):

- 2-stage method with Butcher tableau coefficients derived from 3-point Gauss quadrature
- 4 fixed-point iterations per step to solve the implicit equations
- Preserves the symplectic structure of Hamiltonian dynamics; no artificial energy drift

## Files

| File | Description |
|------|-------------|
| `integrator.cpp` | C++ physics engine (GL4 integrator, fields, lattice loop) |
| `integrator.py` | Python ctypes wrapper; local↔global coordinate conversion |
| `run_simulation.py` | Main script: sets up ICs, calls integrator, writes output |
| `params.json` | Simulation parameters (see below) |
| `plot_results.py` | Plots time-series and Poincaré sections |
| `optimize_sextupole.py` | Scans sextupole strength to minimise chromaticity |

### Build

```bash
g++ -O2 -shared -fPIC -o lib_integrator.so integrator.cpp -lm
```

### Run

```bash
python3 run_simulation.py
```

Output files: `simulation_data.txt`, `poincare_data.txt`, `rf.txt`

```bash
python3 plot_results.py
```

## Parameters (`params.json`)

| Key | Default | Description |
|-----|---------|-------------|
| `R0` | 95.49 | Ring radius (m) |
| `direction` | −1 | Orbit direction: −1 = CW, +1 = CCW |
| `nFODO` | 24 | Number of FODO cells |
| `quadLen` | 0.4 | Quadrupole length (m) |
| `driftLen` | 2.0833 | Drift section length (m) |
| `k1` | 0.21 | Quadrupole gradient (m⁻²) |
| `sextK1` | −0.015 | Sextupole strength (m⁻³) |
| `quadSwitch` | 1 | Enable (1) / disable (0) quadrupoles |
| `sextSwitch` | 0 | Enable sextupoles |
| `E0_power` | 1.0 | Radial E-field fall-off exponent (1 = pure 1/R) |
| `eRatio` | 1.0 | Scale factor on electric field magnitude |
| `rfSwitch` | 0 | Enable RF cavity |
| `rfVoltage` | 1 000 000 | RF peak voltage (V) |
| `h` | 100 | RF harmonic number |
| `t2` | 0.001 | Simulation end time (s) |
| `dt` | 1×10⁻¹¹ | Integration time step (s) |
| `return_steps` | 10 000 | Number of history points saved |
| `dev0` | 0.01 | Initial radial offset (m) |
| `z0` | 0.01 | Initial vertical offset (m) |
| `momError` | 0 | Relative momentum error dp/p |
| `spinHorRotation` | 0 | Initial horizontal spin rotation (rad) |
| `poincare_quad_index` | −1 | Quad index for Poincaré section (−1 = every turn) |
| `B0ver` | 0 | Vertical magnetic field (T) |
| `B0rad` | 0 | Radial magnetic field (T) |
| `B0long` | 0 | Longitudinal magnetic field (T) |
| `EDMSwitch` | 0 | Enable EDM signal |

## Output

`simulation_data.txt` — tab-separated, one row per saved step:

```
Time(s)  Dev_X_m  Y_vert_m  Z_long_m  Px  Py  Pz  S_Rady  S_Dikey  S_Long
```

`poincare_data.txt` — particle state at each Poincaré section crossing.

`rf.txt` — RF phase and dp/p at each cavity crossing (when `rfSwitch=1`).

## Known physics notes

### S_y (vertical spin) oscillation

With a non-zero initial vertical offset (`z0 ≠ 0`), the particle undergoes vertical betatron oscillations. These couple to the spin through two mechanisms in the Thomas-BMT equation:

1. **β×E coupling in the arcs**: vertical velocity `βz` crossed with the radial electric field `E_r` produces a horizontal component of `β×E`, driving `dSz/dt ≠ 0` at rate proportional to `βz · E_r · sin(θ) · (G + 1/(γ+1))/c`.

2. **Quad radial field**: the quadrupole field `Br = K1·Z` (horizontal B proportional to vertical displacement) contributes `Ωx ∝ K1·Z·(G + 1/γ)`, also driving vertical spin change.

Both effects oscillate at the **vertical betatron frequency Qy** and are proportional to the vertical emittance. This is a real spin-orbit coupling effect, not a numerical artifact. Setting `z0 = 0` eliminates the oscillation.

This effect is a potential systematic in the EDM measurement and must be controlled by minimising the vertical beam emittance.
