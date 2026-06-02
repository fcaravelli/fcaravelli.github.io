# Memristive Reservoir Satellite Control

**Author:** generated for Francesco Caravelli (LANL)

## Overview
Three-part simulation of a 6U CubeSat in LEO (400 km) controlled by a
memristive reservoir computer based on the model from
[*The complex dynamics of memristive circuits: analytical results and universal slow relaxation*](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.022140)
(Phys. Rev. E 95, 022140, 2017; [arXiv:1608.08651](https://arxiv.org/abs/1608.08651)).

## Files
| File | Description |
|------|-------------|
| `run_all.py` | **Master script — run this** |
| `satellite_ctdv.py` | Orbital mechanics + memristive reservoir + static plots |
| `make_gif.py` | Animated GIF of the full simulation |
| `index.html` | Browser game version with manual recovery and kick-the-controller modes |
| `game.js` | JavaScript port of the orbit dynamics, CW teacher, and memristive reservoir controller |
| `styles.css` | Browser game layout and visual styling |
| `fig1_orbit.png` | 2D LEO trajectories |
| `fig2_deviation.png` | Orbital deviation vs time |
| `fig3_thrusters.png` | Thruster activation |
| `fig4_reservoir.png` | Reservoir memory state evolution |
| `fig5_dashboard.png` | Summary dashboard |
| `satellite_ctdv.gif` | Animated simulation |

## Requirements
```
pip install numpy scipy matplotlib pillow
```

## Run
```bash
python run_all.py
```

## Browser game
Open `index.html` in a web browser. The app trains the memristive reservoir in the
browser, then exposes two modes:

- **Recover:** the satellite starts after a collision kick and you command the
  four thrusters directly. The `New collision` button applies a harder impact.
- **Kick:** you kick the satellite while the memristive reservoir controller tries to
  recover the reference orbit. The kick slider spans small disturbances through
  severe deorbit-class impulses.

In Kick mode, the controller is considered defeated when the satellite crosses
the game loss boundary: altitude below 120 km, altitude above 1200 km, or
position error above 800 km from the reference orbit.

## Physics
- **Satellite:** 6U CubeSat, 8 kg, 5 mN electrospray thrusters × 4, 20 W budget
- **Orbit:** circular LEO 400 km, T ≈ 92.4 min, V ≈ 7.67 km/s
- **Perturbations:** atmospheric drag (ρ = 2×10⁻¹² kg/m³) + initial 200 m displacement

## Memristive-network evolution law
```
dw/dt = −w/β  +  (α/V₀) Ω (I + ξ W Ω)⁻¹ E(t)
```
- `w ∈ [0,1]^N` — internal memory of each memristor
- `Ω = I − Bᵀ(BBᵀ)⁺B` — cycle-space projector (Kirchhoff topology)
- `ξ = (R_off − R_on)/R_on` — normalised resistance gap
- `E(t) = W_in · δ_norm(t)` — input voltages from satellite state deviation

## Controller
Readout layer `W_out` (2 × N) trained via ridge regression on 25 random
CW-controlled trajectories. Learns to reproduce the Clohessy–Wiltshire
controller from data alone.
