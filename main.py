"""
main.py  --  Drone Simulator v2  |  Ursina render engine
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import traceback
from pathlib import Path

# -- sys.path FIRST so local modules are found before anything else -----------
_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# -- Logging to both console and file (so silent exits are caught) -------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(_HERE / "drone_sim.log"), mode="w", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")
log.info("Python %s  |  cwd: %s", sys.version.split()[0], Path.cwd())

import numpy as np

# -- drone_core ---------------------------------------------------------------
try:
    import drone_core as _core
    PHYSICS_OK = True
    log.info("drone_core loaded OK  (version attrs: %s)", [a for a in dir(_core) if not a.startswith('_')][:6])
except ImportError as exc:
    PHYSICS_OK = False
    log.warning("drone_core not found (%s) -- physics disabled", exc)
except Exception as exc:
    PHYSICS_OK = False
    log.error("drone_core load error: %s", exc, exc_info=True)

# -- Ursina -------------------------------------------------------------------
try:
    from ursina import (
        Ursina, Entity, color, Vec3 as UVec3,
        camera, window, Text, held_keys, application,
        Sky, DirectionalLight, AmbientLight, PointLight,
    )
    from ursina.shaders import unlit_shader, lit_with_shadows_shader
    log.info("ursina imported OK")
except ImportError as exc:
    log.critical("ursina not installed: %s\n  Run: pip install ursina", exc)
    sys.exit(1)
except Exception as exc:
    log.critical("ursina import error: %s", exc, exc_info=True)
    sys.exit(1)

# -- Local modules ------------------------------------------------------------
try:
    from terrain import load_and_build, spawn_terrain_entities
    log.info("terrain.py imported OK")
except Exception as exc:
    log.critical("terrain.py import error: %s", exc, exc_info=True)
    sys.exit(1)

# -- Simulation constants -----------------------------------------------------
PHYS_DT      = 1.0 / 500.0        # physics timestep [s]
SPAWN        = (0.0, 0.0, 150.0)  # world XYZ spawn  [m]
N_PARTICLES  = 1200               # turbulence streak particles
ARROW_GRID   = 12                 # turbulence arrow-field per axis
CAM_LAG      = 7.0                # camera follow smoothing factor
WORLD_SCALE  = 0.01               # Ursina units per metre  (1 u = 100 m)


# =============================================================================
# Drone Visual Entity
# =============================================================================
class DroneEntity(Entity):
    """
    Quadrotor visual: central body + 4 arms + 4 rotor discs.
    Rotor blur opacity and spin rate driven by actual motor RPM from C++.
    """
    _BODY_COLOR  = color.rgb(20, 160, 220)
    _ARM_COLOR   = color.rgb(28, 28, 32)
    _ROTOR_COLOR = color.rgba(200, 200, 255, 80)

    # (local_x, local_y, local_z, rotation_y_deg)
    _ARM_DATA = [
        ( 0.22, 0.0,  0.22,  45.0),
        (-0.22, 0.0,  0.22, -45.0),
        (-0.22, 0.0, -0.22,  45.0),
        ( 0.22, 0.0, -0.22, -45.0),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arms:   list[Entity] = []
        self.rotors: list[Entity] = []
        self._rotor_angle = 0.0
        self._build()

    def _build(self):
        self.body = Entity(
            parent=self, model="cube",
            scale=(0.18, 0.05, 0.18),
            color=self._BODY_COLOR,
            shader=lit_with_shadows_shader,
        )
        self.status_led = Entity(
            parent=self.body, model="sphere",
            scale=0.03, position=UVec3(0, 0.03, 0),
            color=color.lime, shader=unlit_shader,
        )
        for (ax, ay, az, ry) in self._ARM_DATA:
            arm = Entity(
                parent=self, model="cube",
                position=UVec3(ax, ay, az),
                scale=(0.34, 0.012, 0.034),
                rotation_y=ry,
                color=self._ARM_COLOR,
                shader=lit_with_shadows_shader,
            )
            rotor = Entity(
                parent=arm, model="cube",
                scale=(0.18, 0.002, 0.18),
                color=self._ROTOR_COLOR,
                shader=unlit_shader,
            )
            self.arms.append(arm)
            self.rotors.append(rotor)

    def sync(self, state, rpms: list) -> None:
        """Copy C++ DroneState -> Ursina transform.  C++ Z-up -> Ursina Y-up."""
        px, py, pz = state.position()
        self.position = UVec3(px * WORLD_SCALE, pz * WORLD_SCALE, py * WORLD_SCALE)

        roll_d, pitch_d, yaw_d = state.euler_deg()
        self.rotation = UVec3(pitch_d, yaw_d, roll_d)

        mean_rpm = sum(rpms) / max(len(rpms), 1) if rpms else 0.0
        self._rotor_angle += mean_rpm * 0.006 * application.time_step
        for i, rotor in enumerate(self.rotors):
            rotor.rotation_y = self._rotor_angle * (1 if i % 2 == 0 else -1)
            alpha = int(min(255, 50 + mean_rpm * 0.018))
            rotor.color = color.rgba(200, 200, 255, alpha)

        # LED: green = safe, red = near-ground
        agl = state.position()[2] - 5.0
        self.status_led.color = color.red if agl < 12.0 else color.lime


# =============================================================================
# Turbulence VFX
# =============================================================================
class TurbVFX:
    """
    Layer 1: Streak particles  -- advected every frame by C++ velocity field.
    Layer 2: Arrow grid        -- 3-D wind vectors, refreshed at ~5 Hz.
    """
    def __init__(self, phys, n: int = N_PARTICLES,
                 grid: int = ARROW_GRID, radius: float = 700.0):
        self._phys   = phys
        self._radius = radius
        self._grid   = grid

        rng = np.random.default_rng(7)
        self._ppos = rng.uniform(-radius, radius, (n, 3)).astype(np.float64)
        self._ppos[:, 2] += 150.0

        self._streaks: list[Entity] = [
            Entity(model="cube", scale=0.35,
                   color=color.rgba(155, 210, 255, 45), shader=unlit_shader)
            for _ in range(n)
        ]
        n_arrows = grid ** 3
        self._arrows: list[Entity] = [
            Entity(model="cube", scale=(0.08, 0.08, 0.35),
                   color=color.rgba(255, 195, 70, 140), shader=unlit_shader)
            for _ in range(n_arrows)
        ]
        self._arrow_timer = 0.0
        self._ARROW_INTERVAL = 0.18

    def update(self, dt: float, drone_pos: tuple) -> None:
        self._update_streaks(dt, drone_pos)
        self._arrow_timer += dt
        if self._arrow_timer >= self._ARROW_INTERVAL:
            self._arrow_timer = 0.0
            self._update_arrows(drone_pos)

    def _update_streaks(self, dt: float, dp: tuple) -> None:
        xs = np.ascontiguousarray(self._ppos[:, 0])
        ys = np.ascontiguousarray(self._ppos[:, 1])
        zs = np.ascontiguousarray(self._ppos[:, 2])
        vf = self._phys.turbulence_batch(xs, ys, zs)

        self._ppos[:, 0] += (vf[:, 0] * 5.0 + 2.0) * dt
        self._ppos[:, 1] +=  vf[:, 1] * 5.0 * dt
        self._ppos[:, 2] +=  vf[:, 2] * 5.0 * dt

        # Wrap to sphere around drone
        dp_arr = np.array(dp, dtype=np.float64)
        dist   = np.linalg.norm(self._ppos - dp_arr, axis=1)
        wrap   = dist > self._radius
        if wrap.any():
            nw = int(wrap.sum())
            self._ppos[wrap] = dp_arr + np.random.default_rng().uniform(
                -self._radius, self._radius, (nw, 3)
            )

        vmag = np.linalg.norm(vf, axis=1)
        for i, ent in enumerate(self._streaks):
            wx, wy, wz = self._ppos[i]
            ent.position = UVec3(wx * WORLD_SCALE, wz * WORLD_SCALE, wy * WORLD_SCALE)
            alpha = int(np.clip(35 + vmag[i] * 55, 5, 170))
            ent.color = color.rgba(155, 210, 255, alpha)
            ent.scale = UVec3(0.18 + vmag[i] * 0.04, 0.18, 0.35 + vmag[i] * 0.28)

    def _update_arrows(self, dp: tuple) -> None:
        g  = self._grid
        s  = 260.0
        xs = np.linspace(dp[0] - s/2, dp[0] + s/2, g)
        ys = np.linspace(dp[1] - s/2, dp[1] + s/2, g)
        zs = np.linspace(dp[2] - s/3, dp[2] + s/3, g)
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

        vf   = self._phys.turbulence_batch(
            np.ascontiguousarray(pts[:, 0]),
            np.ascontiguousarray(pts[:, 1]),
            np.ascontiguousarray(pts[:, 2]),
        )
        vmag = np.linalg.norm(vf, axis=1)

        for i, ent in enumerate(self._arrows):
            if i >= len(pts):
                break
            wx, wy, wz = pts[i]
            ent.position = UVec3(wx * WORLD_SCALE, wz * WORLD_SCALE, wy * WORLD_SCALE)
            vm = float(vmag[i])
            if vm > 0.04:
                vn    = vf[i] / vm
                yaw_d = math.degrees(math.atan2(float(vn[1]), float(vn[0])))
                pit_d = math.degrees(math.asin(float(np.clip(vn[2], -1.0, 1.0))))
                ent.rotation = UVec3(-pit_d, yaw_d, 0.0)
                ent.scale    = UVec3(0.14, 0.14, 0.22 + vm * 0.5)
                rc = int(np.clip(100 + vm * 80,  0, 255))
                gc = int(np.clip(195 - vm * 50,  0, 255))
                bc = int(np.clip(255 - vm * 120, 0, 255))
                ent.color = color.rgba(rc, gc, bc, int(np.clip(65 + vm * 95, 0, 240)))
            else:
                ent.scale = UVec3(0.01, 0.01, 0.01)


# =============================================================================
# Noise Visualiser  (overhead 2-D slice)
# =============================================================================
class NoiseVisualiser:
    _RES = 32

    def __init__(self, phys, noise_type: str = "warp"):
        self._phys  = phys
        self._type  = noise_type
        self._timer = 0.0

        r  = self._RES
        cw = 0.15
        self._cells: list[Entity] = [
            Entity(model="cube", scale=(cw, 0.01, cw),
                   color=color.white, shader=unlit_shader)
            for _ in range(r * r)
        ]

    def set_type(self, t: str) -> None:
        self._type = t

    def update(self, dt: float, drone_pos: tuple) -> None:
        self._timer += dt
        if self._timer < 0.5:
            return
        self._timer = 0.0

        r   = self._RES
        arr = self._phys.noise_slice_2d(
            drone_pos[0], drone_pos[1], drone_pos[2],
            400.0, r, self._type,
        )
        mn  = float(arr.min())
        rng = max(float(arr.max()) - mn, 1e-6)

        for idx, ent in enumerate(self._cells):
            row = idx // r
            col = idx  % r
            t   = (float(arr[row, col]) - mn) / rng
            nx  = (drone_pos[0] - 200.0 + col * (400.0 / r)) * WORLD_SCALE
            nz  = (drone_pos[1] - 200.0 + row * (400.0 / r)) * WORLD_SCALE
            ny  = (drone_pos[2] - 5.0)  * WORLD_SCALE
            ent.position = UVec3(nx, ny, nz)
            rc = int(np.clip(t * 255,          0, 255))
            gc = int(np.clip(t * 220 + 20,     0, 255))
            bc = int(np.clip((1.0-t)*200 + 55, 0, 255))
            ent.color = color.rgba(rc, gc, bc, 120)


# =============================================================================
# HUD
# =============================================================================
class HUD:
    def __init__(self):
        kw = dict(parent=camera.ui, scale=0.75, background=False)
        self.telem  = Text(text="", position=(-0.88,  0.46), **kw)
        self.sensor = Text(text="", position=(-0.88,  0.06), **kw)
        self.ctrl   = Text(text="", position=( 0.40,  0.46), **kw)
        self.perf   = Text(text="", position=( 0.40, -0.42), **kw)
        self._fps:  list[float] = []

    def update(self, phys, target: list, ctrl_mode: str, turb_model: str,
               turb_intensity: float, dt: float) -> None:
        if not PHYSICS_OK:
            return

        s    = phys.state()
        pos  = s.position()
        vel  = s.velocity()
        eu   = s.euler_deg()
        ow   = s.ang_vel()
        rpms = phys.motor_rpms()
        pwr  = phys.total_power_w()
        agl  = phys.altitude_agl()
        tv   = phys.turbulence_at(*pos)

        self.telem.text = (
            "-- TELEMETRY ------------------\n"
            f"  Pos  X:{pos[0]:+9.1f}  Y:{pos[1]:+9.1f}  Z:{pos[2]:+8.1f} m\n"
            f"  AGL  {agl:6.1f} m\n"
            f"  Vel  X:{vel[0]:+7.2f}  Y:{vel[1]:+7.2f}  Z:{vel[2]:+7.2f} m/s\n"
            f"  Att  R:{eu[0]:+7.2f}  P:{eu[1]:+7.2f}  Y:{eu[2]:+7.2f} deg\n"
            f"  w    R:{ow[0]:+6.3f}  P:{ow[1]:+6.3f}  Y:{ow[2]:+6.3f} r/s\n"
            f"  RPM  {' '.join(f'{r:5.0f}' for r in rpms)}\n"
            f"  Pwr  {pwr:7.1f} W\n"
            f"  t    {phys.sim_time():9.2f} s\n"
            f"  CTRL {ctrl_mode}"
        )

        alt_r, pres_r, temp_r   = phys.read_baro()
        gps_p, gps_v, hdop, fix = phys.read_gps()
        imu_a, imu_g, imu_t     = phys.read_imu()

        self.sensor.text = (
            "-- SENSORS --------------------\n"
            f"  BARO alt:{alt_r:7.1f}m  P:{pres_r:7.0f}Pa  T:{temp_r:4.1f}C\n"
            f"  GPS  X:{gps_p[0]:+7.1f} Y:{gps_p[1]:+7.1f} Z:{gps_p[2]:+7.1f}\n"
            f"       HDOP:{hdop:.2f}  {'FIX' if fix else 'NO-FIX'}\n"
            f"  IMU  a({imu_a[0]:+5.2f},{imu_a[1]:+5.2f},{imu_a[2]:+5.2f}) m/s2\n"
            f"       w({imu_g[0]:+5.3f},{imu_g[1]:+5.3f},{imu_g[2]:+5.3f}) r/s\n"
            f"       T={imu_t:.1f}C\n"
            f"  TURB ({tv[0]:+5.2f},{tv[1]:+5.2f},{tv[2]:+5.2f}) m/s\n"
            f"  Mdl  {turb_model}  I={turb_intensity:.2f}\n"
            f"  Tgt  ({target[0]:+7.1f},{target[1]:+7.1f},{target[2]:+7.1f})"
        )

        self.ctrl.text = (
            "-- CONTROLS -------------------\n"
            "  W/S        Forward / Back\n"
            "  A/D        Strafe Left / Right\n"
            "  R/F        Ascend / Descend\n"
            "  Q/E        Yaw Left / Right\n"
            "  1/2/3/4    Turb model\n"
            "  = / -      Turb intensity\n"
            "  C          Cycle camera\n"
            "  M          Toggle control mode\n"
            "  N          Noise overlay\n"
            "  ESC        Quit"
        )

        self._fps.append(1.0 / max(dt, 1e-6))
        if len(self._fps) > 60:
            self._fps.pop(0)
        fps = sum(self._fps) / len(self._fps)
        self.perf.text = f"FPS  {fps:5.1f}"


# =============================================================================
# Camera Controller
# =============================================================================
class CameraController:
    _MODES = ("follow", "fpv", "top", "orbit")

    def __init__(self):
        self._idx        = 0
        self._orbit_yaw  = 0.0
        self._smooth_pos = UVec3(0, 5, 0)

    @property
    def mode(self) -> str:
        return self._MODES[self._idx % len(self._MODES)]

    def cycle(self) -> None:
        self._idx = (self._idx + 1) % len(self._MODES)
        log.info("Camera: %s", self.mode)

    def update(self, drone: DroneEntity, dt: float) -> None:
        tp = drone.position
        m  = self.mode

        if m == "follow":
            target = tp + UVec3(0, 3, -8)
            k = min(CAM_LAG * dt, 1.0)
            self._smooth_pos = UVec3(
                self._smooth_pos.x + (target.x - self._smooth_pos.x) * k,
                self._smooth_pos.y + (target.y - self._smooth_pos.y) * k,
                self._smooth_pos.z + (target.z - self._smooth_pos.z) * k,
            )
            camera.position = self._smooth_pos
            camera.look_at(tp + UVec3(0, 1, 0))

        elif m == "fpv":
            camera.position = tp + UVec3(0, 0.5, 0.3)
            camera.rotation = drone.rotation

        elif m == "top":
            camera.position = UVec3(tp.x, tp.y + 22, tp.z)
            camera.look_at(tp)

        elif m == "orbit":
            r  = 12.0
            ox = r * math.cos(math.radians(self._orbit_yaw))
            oz = r * math.sin(math.radians(self._orbit_yaw))
            camera.position = tp + UVec3(ox, 5, oz)
            camera.look_at(tp)
            self._orbit_yaw += 18.0 * dt


# =============================================================================
# Single-fire key helper (fires once per keypress, not per-frame while held)
# =============================================================================
class KeyOnce:
    def __init__(self):
        self._prev: set[str] = set()

    def fired(self, key: str) -> bool:
        now = bool(held_keys[key])
        was = key in self._prev
        if now:
            self._prev.add(key)
        else:
            self._prev.discard(key)
        return now and not was


# =============================================================================
# Main Application
# =============================================================================
class DroneSimulator:
    _SUBSTEPS    = max(1, round(1.0 / (PHYS_DT * 60)))
    _CTRL_MODES  = ("cascaded_pid", "so3_geometric")
    _NOISE_TYPES = ("fbm", "perlin", "simplex", "worley", "ridged", "warp")
    _TURB_MODELS = ("dryden", "vonkarman", "kh", "composite")

    def __init__(
        self,
        dem_path:  str   = "",
        turb:      float = 0.6,
        seed:      int   = 42,
        lod:       int   = 2,
        frame:     str   = "quad_x",
    ):
        self._target_pos   = list(SPAWN)
        self._target_yaw   = 0.0
        self._turb_intens  = turb
        self._turb_model   = "composite"
        self._ctrl_idx     = 0
        self._noise_idx    = 0
        self._noise_vis_on = False
        self._key          = KeyOnce()

        log.info("Initialising DroneSimulator  dem=%r  turb=%.2f  frame=%s",
                 dem_path, turb, frame)

        try:
            self._setup(dem_path, turb, seed, lod, frame)
        except Exception:
            log.critical("DroneSimulator setup crashed:\n%s", traceback.format_exc())
            raise

    def _setup(
        self,
        dem_path:  str,
        turb:      float,
        seed:      int,
        lod:       int,
        frame:     str,
    ) -> None:
        self.app = Ursina(
            title="Drone Simulator v2 -- Enterprise Edition",
            development_mode=False,
        )
        window.borderless = False
        window.fullscreen = False
        window.size       = (1280, 720)
        window.position   = (80, 60)
        window.fps_counter.enabled = False
        window.exit_button.visible = False
        camera.fov = 75

        AmbientLight(color=color.rgba(75, 85, 105, 255))
        sun = DirectionalLight(shadows=True)
        sun.look_at(UVec3(1, -2, 1))
        PointLight(
            position=UVec3(0, 40, 0),
            color=color.rgba(220, 200, 180, 160),
        )
        Sky(texture="sky_sunset")

        # -- Terrain ----------------------------------------------------------
        log.info("Loading DEM ...")
        self._dem, mesh_data = load_and_build(
            dem_path    = dem_path,
            downsample  = 2,
            lod_stride  = lod,
            chunk_mode  = False,
        )
        spawn_terrain_entities(mesh_data, world_scale=WORLD_SCALE)
        log.info("Terrain entity spawned")

        # -- C++ Physics ------------------------------------------------------
        if PHYSICS_OK:
            cfg            = _core.DroneConfig()
            cfg.frame_type = frame
            cfg.mass       = 1.5
            cfg.arm_length = 0.225
            cfg.Ixx        = 0.0123
            cfg.Iyy        = 0.0123
            cfg.Izz        = 0.0245

            self._phys = _core.DronePhysics(cfg, seed)
            self._phys.load_terrain(
                self._dem.heights,
                float(self._dem.x_origin),
                float(self._dem.y_origin),
                float(self._dem.cell_size),
            )
            self._phys.set_spawn(*SPAWN)
            self._phys.set_target(*SPAWN, 0.0)
            self._apply_turbulence()
            self._phys.set_mean_wind(3.0, 1.0, 0.0)
            log.info(
                "Physics engine ready  (frame=%s  seed=%d  substeps=%d)",
                frame, seed, self._SUBSTEPS,
            )
        else:
            self._phys = None

        # -- Drone visual -----------------------------------------------------
        self._drone = DroneEntity()
        self._drone.position = UVec3(
            SPAWN[0] * WORLD_SCALE,
            SPAWN[2] * WORLD_SCALE,
            SPAWN[1] * WORLD_SCALE,
        )

        # -- VFX --------------------------------------------------------------
        if self._phys:
            self._turb_vfx  = TurbVFX(self._phys, N_PARTICLES, ARROW_GRID)
            self._noise_vis = NoiseVisualiser(self._phys, "warp")
        else:
            self._turb_vfx  = None
            self._noise_vis = None

        # -- HUD + Camera -----------------------------------------------------
        self._hud = HUD()
        self._cam = CameraController()

        self.app.update = self._update

    # -------------------------------------------------------------------------
    def _apply_turbulence(self) -> None:
        if not (PHYSICS_OK and self._phys):
            return
        tp                 = _core.TurbulenceParams()
        tp.intensity       = self._turb_intens
        tp.model           = self._turb_model
        tp.shear_layer_alt = 300.0
        tp.delta_u         = 8.0
        self._phys.set_turbulence(tp)

    # -------------------------------------------------------------------------
    def _handle_input(self, dt: float) -> None:
        if not self._phys:
            return

        sp = 4.0    # target translate speed [x100 for world units]
        yr = 40.0   # yaw rate [deg/s]

        if held_keys["w"]:  self._target_pos[1] += sp * dt * 100
        if held_keys["s"]:  self._target_pos[1] -= sp * dt * 100
        if held_keys["a"]:  self._target_pos[0] -= sp * dt * 100
        if held_keys["d"]:  self._target_pos[0] += sp * dt * 100
        if held_keys["r"]:  self._target_pos[2] += 3.0 * dt * 100
        if held_keys["f"]:  self._target_pos[2] -= 3.0 * dt * 100
        if held_keys["q"]:  self._target_yaw    -= yr * dt
        if held_keys["e"]:  self._target_yaw    += yr * dt

        # Turbulence model (fire-once)
        for i, key in enumerate(("1", "2", "3", "4")):
            if self._key.fired(key):
                self._turb_model = self._TURB_MODELS[i]
                self._apply_turbulence()
                log.info("Turbulence model: %s", self._turb_model)

        if held_keys["="]:
            self._turb_intens = min(2.0, self._turb_intens + 0.4 * dt)
            self._apply_turbulence()
        if held_keys["-"]:
            self._turb_intens = max(0.0, self._turb_intens - 0.4 * dt)
            self._apply_turbulence()

        if self._key.fired("c"):
            self._cam.cycle()

        if self._key.fired("m"):
            self._ctrl_idx = (self._ctrl_idx + 1) % len(self._CTRL_MODES)
            mode = self._CTRL_MODES[self._ctrl_idx]
            self._phys.set_control_mode(mode)
            log.info("Control mode: %s", mode)

        if self._key.fired("n") and self._noise_vis:
            if not self._noise_vis_on:
                self._noise_vis_on = True
                log.info("Noise overlay ON (%s)", self._NOISE_TYPES[self._noise_idx])
            else:
                self._noise_idx = (self._noise_idx + 1) % len(self._NOISE_TYPES)
                nt = self._NOISE_TYPES[self._noise_idx]
                self._noise_vis.set_type(nt)
                log.info("Noise overlay: %s", nt)

        self._phys.set_target(*self._target_pos, self._target_yaw)

    # -------------------------------------------------------------------------
    def _update(self) -> None:
        dt = application.time_step
        if dt <= 0.0:
            return

        self._handle_input(dt)

        if self._phys:
            # Physics substeps at 500 Hz
            for _ in range(self._SUBSTEPS):
                self._phys.step(PHYS_DT)

            s    = self._phys.state()
            rpms = self._phys.motor_rpms()
            self._drone.sync(s, rpms)
            dp = s.position()

            if self._turb_vfx:
                self._turb_vfx.update(dt, dp)

            if self._noise_vis and self._noise_vis_on:
                self._noise_vis.update(dt, dp)

            self._hud.update(
                self._phys,
                self._target_pos,
                self._CTRL_MODES[self._ctrl_idx],
                self._turb_model,
                self._turb_intens,
                dt,
            )

        self._cam.update(self._drone, dt)

    # -------------------------------------------------------------------------
    def run(self) -> None:
        log.info("Starting render loop ...")
        self.app.run()


# =============================================================================
# Entry point
# =============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description="Drone Simulator v2")
    p.add_argument("--dem",   default="",        help="Path to GeoTIFF DEM")
    p.add_argument("--turb",  default=0.6,       type=float, help="Turbulence intensity [0-2]")
    p.add_argument("--seed",  default=42,        type=int,   help="Noise seed")
    p.add_argument("--lod",   default=2,         type=int,   help="Terrain LOD stride")
    p.add_argument("--frame", default="quad_x",              help="quad_x | hex | octo")
    args = p.parse_args()

    log.info("Args: %s", vars(args))

    try:
        DroneSimulator(
            dem_path = args.dem,
            turb     = args.turb,
            seed     = args.seed,
            lod      = args.lod,
            frame    = args.frame,
        ).run()
    except SystemExit:
        pass  # normal Ursina exit
    except Exception:
        log.critical("CRASHED:\n%s", traceback.format_exc())
        print("\n\n=== CRASH — see drone_sim.log for details ===\n")
        print(traceback.format_exc())
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
