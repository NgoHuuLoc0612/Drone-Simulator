# Drone Simulator v2

A high-fidelity quadrotor flight simulator with a **C++ physics engine** (pybind11), **Python/Ursina** 3-D renderer, and a comprehensive suite of aerodynamic and environmental models.

```
build.bat → cmake → drone_core.pyd → python main.py
```

---

## Features

### Physics Engine (`drone_core`)
- **6-DOF rigid-body dynamics** — full quaternion integration with RK4
- **Drivetrain** — per-rotor DC motor model (back-EMF, winding inductance, thermal drift) + Blade Element Theory thrust
- **Aerodynamics** — body drag tensor, ground effect (Cheeseman-Bennett), vortex ring state detection, rotor-to-rotor interference, added mass
- **Controller** — cascaded PID (pos → vel → attitude → rate) and SO(3) geometric controller (Lee et al. 2010)
- **Terrain** — bilinear DEM heightfield, collision response with Baumgarte stabilisation and Coulomb friction
- **Sensors** — IMU, GPS (HDOP, multipath), barometer, magnetometer, optical flow, lidar altimeter — all with realistic noise models

### Turbulence Models
| Model | Description |
|---|---|
| `dryden` | MIL-SPEC Dryden turbulence channels |
| `vonkarman` | Von Kármán spectral model |
| `kh` | Kelvin–Helmholtz shear instability billows |
| `composite` | Weighted blend of Dryden + VK + KH + thermal convection + domain-warp |

### Parallelism & SIMD
- **AVX2 + FMA** — 4-wide `__m256d` fBm evaluation for noise and turbulence
- **Work-stealing thread pool** — Chase-Lev lock-free deques, NUMA-aware thread affinity, exponential back-off spin
- **`DroneSwarm`** — N drones stepped in parallel via `parallel_for`
- Three turbulence batch paths exposed to Python: serial, parallel (ThreadPool), SIMD Dryden

### Terrain Pipeline (`terrain.py`)
- GeoTIFF ingestion with rasterio (UTM reprojection, nodata fill, Gaussian pre-filter)
- Synthetic procedural DEM fallback (fBm + Worley crags)
- Slope / aspect / Laplacian curvature / Lambertian hillshade
- Quadtree LOD (4 levels, strides 1/2/4/8), T-junction seam stitching
- Elevation × slope vertex colour ramp (7-stop: water → snow)

### Visualisation
- **Ursina** 3-D render with lit shadows
- Quadrotor model with rotor-blur opacity driven by actual RPM
- 1 200-particle turbulence streak field (advected by C++ velocity)
- 3-D arrow grid showing live wind vectors (refreshed at ~5 Hz)
- 2-D noise overlay (6 types: fBm, Perlin, Simplex, Worley, Ridged, Domain-Warp)
- Full telemetry + sensor HUD, 4 camera modes (follow / FPV / top / orbit)

---

## Requirements

| Dependency | Purpose |
|---|---|
| Python ≥ 3.10 | Runtime |
| CMake ≥ 3.18 | Build system |
| GCC/MinGW64 or MSVC 2019+ | C++17 compiler |
| pybind11 | C++/Python bridge |
| ursina | 3-D renderer |
| numpy | Array math |
| rasterio *(optional)* | GeoTIFF DEM loading |
| scipy *(optional)* | Nodata fill, Gaussian filter |
| scikit-image *(optional)* | LOD downscale |

---

## Build & Run

```bat
build.bat
```

The script auto-detects Ninja → Visual Studio → MinGW → NMake, installs Python dependencies, and copies `drone_core.pyd` to the project root.

```bash
python main.py                          # synthetic terrain, default settings
python main.py --dem path/to/dem.tif    # real GeoTIFF DEM
python main.py --turb 1.2               # turbulence intensity (0–2)
python main.py --frame hex              # hexarotor
python main.py --frame octo             # octorotor
python main.py --lod 3                  # terrain LOD stride
python main.py --seed 999               # noise seed
```

### Swarm mode (Python API)
```python
import drone_core as dc

swarm = dc.DroneSwarm()
for _ in range(16):
    swarm.add_drone(dc.DroneConfig(), base_seed=42)
swarm.spawn_grid(spacing=5.0, altitude=100.0)

for tick in range(10_000):
    swarm.step_all(dt=1/500)            # parallel across all drones
```

---

## Controls

| Key | Action |
|---|---|
| `W` / `S` | Forward / Back |
| `A` / `D` | Strafe Left / Right |
| `R` / `F` | Ascend / Descend |
| `Q` / `E` | Yaw Left / Right |
| `1` `2` `3` `4` | Turbulence model |
| `=` / `-` | Turbulence intensity |
| `C` | Cycle camera (follow / FPV / top / orbit) |
| `M` | Toggle control mode (cascaded PID ↔ SO3) |
| `N` | Noise overlay (cycles type each press) |
| `ESC` | Quit |

---

## Project Structure

```
drone-simulator/
├── build.bat                  # one-click Windows build script
├── CMakeLists.txt             # CMake project (AVX2 auto-detect)
├── main.py                    # Ursina app entry point
├── terrain.py                 # DEM pipeline + Ursina mesh builder
└── core/
    ├── bindings.cpp           # pybind11 module definition
    └── include/
        ├── drone_physics.hpp  # DronePhysics, DroneSwarm
        ├── controller.hpp     # CascadedPID, SO3, ControlMixer
        ├── motor_dynamics.hpp # DC motor + BET rotor model
        ├── aerodynamics.hpp   # drag tensor, ground effect, VRS
        ├── turbulence.hpp     # 6 turbulence models + SIMD batch
        ├── noise.hpp          # 8 noise algorithms + AVX2 fBm
        ├── collision.hpp      # TerrainCollider, ImpulseResponse, RayCaster
        ├── sensors.hpp        # IMU, GPS, baro, mag, optical flow, lidar
        ├── atmosphere.hpp     # ISA 1976 + wind shear profiles
        ├── math_types.hpp     # Vec3, Mat3, Quat (header-only)
        ├── simd_math.hpp      # Vec3x4 AVX2 primitives, CPUID detection
        └── thread_pool.hpp    # Work-stealing pool, NUMA affinity, parallel_for
```

---

## Python API Quick Reference

```python
import drone_core as dc
import numpy as np

# ── Single drone ──────────────────────────────────────────────────────────────
cfg            = dc.DroneConfig()
cfg.frame_type = "quad_x"   # quad_x | hex | octo
cfg.mass       = 1.5        # kg
phys = dc.DronePhysics(cfg, seed=42)

phys.set_spawn(0, 0, 100)
phys.set_target(50, 50, 120, yaw_deg=30)

tp = dc.TurbulenceParams()
tp.intensity = 0.8
tp.model     = "composite"
phys.set_turbulence(tp)

for _ in range(500):
    phys.step(1/500)

s = phys.state()
print(s.position(), s.euler_deg())

# ── Batch turbulence ──────────────────────────────────────────────────────────
N  = 2048
xs = np.random.uniform(-500, 500, N)
ys = np.random.uniform(-500, 500, N)
zs = np.full(N, 100.0)

wind_serial   = phys.turbulence_batch(xs, ys, zs)          # (N, 3)
wind_parallel = phys.turbulence_batch_parallel(xs, ys, zs)  # ThreadPool
wind_simd     = phys.turbulence_batch_simd(xs, ys, zs)      # AVX2 Dryden

# ── Thread pool stats ─────────────────────────────────────────────────────────
print(dc.pool_stats())
# {'jobs_submitted': …, 'steals': …, 'n_threads': 16, …}

# ── CPU features ─────────────────────────────────────────────────────────────
print(dc.cpu_info())
# {'avx2': True, 'fma': True, 'sse42': True}
```

---

## License

MIT — see [LICENSE](LICENSE)
