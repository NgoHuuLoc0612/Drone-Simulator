/**
 * bindings.cpp  (parallel edition)
 * Exposes all original APIs + new parallel/SIMD/swarm APIs to Python.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "include/drone_physics.hpp"

namespace py = pybind11;

// ── Helpers ──────────────────────────────────────────────────────────────────
static void copy_dem_to_terrain(
    DronePhysics& dp, py::array_t<float> dem,
    double x_orig,double y_orig,double cell)
{
    auto buf=dem.request();
    if(buf.ndim!=2) throw std::runtime_error("DEM must be 2D array");
    int rows=static_cast<int>(buf.shape[0]);
    int cols=static_cast<int>(buf.shape[1]);
    dp.load_terrain(static_cast<const float*>(buf.ptr),rows,cols,x_orig,y_orig,cell);
}

// Generic batch turbulence helper — dispatches to requested path
static py::array_t<double> _batch_turb(
    const DronePhysics& dp,
    py::array_t<double> xs,py::array_t<double> ys,py::array_t<double> zs,
    const std::string& path)
{
    auto bx=xs.request(),by=ys.request(),bz=zs.request();
    if(bx.size!=by.size||bx.size!=bz.size)
        throw std::runtime_error("xs/ys/zs must have equal length");
    py::ssize_t N=static_cast<py::ssize_t>(bx.size);
    auto out=py::array_t<double>({N,static_cast<py::ssize_t>(3)});
    auto bo=out.mutable_unchecked<2>();

    const double* px=static_cast<const double*>(bx.ptr);
    const double* py_=static_cast<const double*>(by.ptr);
    const double* pz=static_cast<const double*>(bz.ptr);
    double* pout=&bo(0,0);

    if(path=="parallel")
        dp.turbulence_batch_parallel(px,py_,pz,pout,static_cast<int>(N));
    else if(path=="simd_dryden")
        dp.turbulence_batch_dryden_simd(px,py_,pz,pout,static_cast<int>(N));
    else
        dp.turbulence_batch(px,py_,pz,pout,static_cast<int>(N));
    return out;
}

static py::array_t<double> batch_turbulence(
    const DronePhysics& dp,
    py::array_t<double> xs,py::array_t<double> ys,py::array_t<double> zs)
{ return _batch_turb(dp,xs,ys,zs,"serial"); }

static py::array_t<double> batch_turbulence_parallel(
    const DronePhysics& dp,
    py::array_t<double> xs,py::array_t<double> ys,py::array_t<double> zs)
{ return _batch_turb(dp,xs,ys,zs,"parallel"); }

static py::array_t<double> batch_turbulence_simd(
    const DronePhysics& dp,
    py::array_t<double> xs,py::array_t<double> ys,py::array_t<double> zs)
{ return _batch_turb(dp,xs,ys,zs,"simd_dryden"); }

static py::array_t<float> noise_slice_2d(
    const DronePhysics& dp,
    double cx,double cy,double z,
    double span,int res,const std::string& type)
{
    std::vector<py::ssize_t> shape={static_cast<py::ssize_t>(res),
                                    static_cast<py::ssize_t>(res)};
    auto out=py::array_t<float>(shape);
    auto buf=out.mutable_unchecked<2>();
    double step=span/static_cast<double>(res);
    for(int r=0;r<res;++r) for(int c=0;c<res;++c){
        double x=cx-span*0.5+c*step, y=cy-span*0.5+r*step;
        buf(r,c)=dp.noise_at(x,y,z,type);
    }
    return out;
}

static py::tuple read_imu(const DronePhysics& dp){
    const auto& s=dp.state();
    Vec3 accel_approx{0,0,0};
    double motor_temp=50.0;
    auto r=dp.sensors().imu(accel_approx,s.omega,s.att,motor_temp,1.0/500.0);
    return py::make_tuple(r.accel_body.to_array(),r.gyro_body.to_array(),r.temperature);
}
static py::tuple read_gps(const DronePhysics& dp){
    const auto& s=dp.state();
    auto r=dp.sensors().gps(s.pos,s.vel,1.0/500.0);
    return py::make_tuple(r.position.to_array(),r.velocity.to_array(),r.hdop,r.fix);
}
static py::tuple read_baro(const DronePhysics& dp){
    auto r=dp.sensors().barometer(dp.state().pos.z,1.0/500.0);
    return py::make_tuple(r.altitude,r.pressure,r.temperature);
}

// ThreadPool stats helper
static py::dict pool_stats(){
    const auto& s=global_pool().stats();
    py::dict d;
    d["jobs_submitted"]=s.jobs_submitted.load();
    d["jobs_executed"] =s.jobs_executed.load();
    d["steals"]        =s.steals.load();
    d["spin_loops"]    =s.spin_loops.load();
    d["park_waits"]    =s.park_waits.load();
    d["n_threads"]     =global_pool().n_threads();
    return d;
}

// CPU features
static py::dict cpu_info(){
    const auto& f=cpu_features();
    py::dict d;
    d["avx2"] =f.avx2;
    d["fma"]  =f.fma;
    d["sse42"]=f.sse42;
    return d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Module
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(drone_core,m){
    m.doc()="drone_core — 6DOF drone sim (C++ SIMD+ThreadPool engine)";

    // ── TurbulenceParams ─────────────────────────────────────────────────────
    py::class_<TurbulenceParams>(m,"TurbulenceParams")
        .def(py::init<>())
        .def_readwrite("intensity",        &TurbulenceParams::intensity)
        .def_readwrite("shear_layer_alt",  &TurbulenceParams::shear_layer_alt)
        .def_readwrite("delta_u",          &TurbulenceParams::delta_u)
        .def_readwrite("geostrophic_wind", &TurbulenceParams::geostrophic_wind)
        .def_readwrite("surface_roughness",&TurbulenceParams::surface_roughness)
        .def_readwrite("coriolis_f",       &TurbulenceParams::coriolis_f)
        .def_readwrite("K_eddy",           &TurbulenceParams::K_eddy)
        .def_readwrite("model",            &TurbulenceParams::model);

    // ── MotorSpec ────────────────────────────────────────────────────────────
    py::class_<MotorSpec>(m,"MotorSpec")
        .def(py::init<>())
        .def(py::init<double>(),py::arg("kv"))
        .def_readwrite("Kv",        &MotorSpec::Kv)
        .def_readwrite("R_winding", &MotorSpec::R_winding)
        .def_readwrite("L_winding", &MotorSpec::L_winding)
        .def_readwrite("J_rotor",   &MotorSpec::J_rotor)
        .def_readwrite("V_battery", &MotorSpec::V_battery)
        .def_readwrite("eta_mech",  &MotorSpec::eta_mech);

    // ── RotorSpec ────────────────────────────────────────────────────────────
    py::class_<RotorSpec>(m,"RotorSpec")
        .def(py::init<>())
        .def_readwrite("radius",      &RotorSpec::radius)
        .def_readwrite("chord",       &RotorSpec::chord)
        .def_readwrite("n_blades",    &RotorSpec::n_blades)
        .def_readwrite("pitch_angle", &RotorSpec::pitch_angle)
        .def_readwrite("Ct",          &RotorSpec::Ct)
        .def_readwrite("Cq",          &RotorSpec::Cq);

    // ── DroneConfig ──────────────────────────────────────────────────────────
    py::class_<DroneConfig>(m,"DroneConfig")
        .def(py::init<>())
        .def_readwrite("mass",        &DroneConfig::mass)
        .def_readwrite("Ixx",         &DroneConfig::Ixx)
        .def_readwrite("Iyy",         &DroneConfig::Iyy)
        .def_readwrite("Izz",         &DroneConfig::Izz)
        .def_readwrite("Ixy",         &DroneConfig::Ixy)
        .def_readwrite("Ixz",         &DroneConfig::Ixz)
        .def_readwrite("Iyz",         &DroneConfig::Iyz)
        .def_readwrite("arm_length",  &DroneConfig::arm_length)
        .def_readwrite("rotor_radius",&DroneConfig::rotor_radius)
        .def_readwrite("n_rotors",    &DroneConfig::n_rotors)
        .def_readwrite("frame_type",  &DroneConfig::frame_type)
        .def_readwrite("motor",       &DroneConfig::motor)
        .def_readwrite("rotor",       &DroneConfig::rotor);

    // ── DroneState ───────────────────────────────────────────────────────────
    py::class_<DroneState>(m,"DroneState")
        .def(py::init<>())
        .def("position",   &DroneState::position)
        .def("quaternion", &DroneState::quaternion)
        .def("velocity",   &DroneState::velocity)
        .def("ang_vel",    &DroneState::ang_vel)
        .def("euler_deg",  &DroneState::euler_deg)
        .def_readwrite("sim_time",&DroneState::sim_time)
        .def_property("pos_x",
            [](const DroneState& s){return s.pos.x;},
            [](DroneState& s,double v){s.pos.x=v;})
        .def_property("pos_y",
            [](const DroneState& s){return s.pos.y;},
            [](DroneState& s,double v){s.pos.y=v;})
        .def_property("pos_z",
            [](const DroneState& s){return s.pos.z;},
            [](DroneState& s,double v){s.pos.z=v;});

    // ── DronePhysics ─────────────────────────────────────────────────────────
    py::class_<DronePhysics>(m,"DronePhysics")
        .def(py::init<const DroneConfig&,uint64_t>(),
             py::arg("config")=DroneConfig{},py::arg("seed")=42ULL)

        .def("load_terrain",&copy_dem_to_terrain,
             py::arg("dem"),py::arg("x_origin"),py::arg("y_origin"),py::arg("cell_size"))

        .def("set_target",       &DronePhysics::set_target,
             py::arg("x"),py::arg("y"),py::arg("z"),py::arg("yaw_deg")=0.0)
        .def("set_turbulence",   &DronePhysics::set_turbulence_params)
        .def("set_mean_wind",    &DronePhysics::set_mean_wind)
        .def("set_control_mode", &DronePhysics::set_control_mode)
        .def("set_pid_gains",    &DronePhysics::set_pid_gains,
             py::arg("loop"),py::arg("kp"),py::arg("ki"),py::arg("kd"),
             py::arg("imax")=100.0,py::arg("outmax")=200.0)
        .def("set_spawn",        &DronePhysics::set_spawn)

        .def("step",             &DronePhysics::step)
        .def("sim_time",         &DronePhysics::sim_time)

        .def("state",[](DronePhysics& dp)->DroneState&{return dp.state();},
             py::return_value_policy::reference_internal)
        .def("motor_rpms",       &DronePhysics::motor_rpms)
        .def("throttles",        &DronePhysics::throttles)
        .def("total_power_w",    &DronePhysics::total_power_w)

        // ── Turbulence batch paths ────────────────────────────────────────
        .def("turbulence_at",    &DronePhysics::turbulence_at)
        // Serial (original)
        .def("turbulence_batch", &batch_turbulence,
             py::arg("xs"),py::arg("ys"),py::arg("zs"))
        // Parallel via ThreadPool (new)
        .def("turbulence_batch_parallel",&batch_turbulence_parallel,
             py::arg("xs"),py::arg("ys"),py::arg("zs"))
        // SIMD Dryden (new — fastest for VFX)
        .def("turbulence_batch_simd",&batch_turbulence_simd,
             py::arg("xs"),py::arg("ys"),py::arg("zs"))

        .def("noise_slice_2d",   &noise_slice_2d,
             py::arg("cx"),py::arg("cy"),py::arg("z"),
             py::arg("span")=1000.0,py::arg("resolution")=64,py::arg("type")="fbm")

        .def("read_imu",  &read_imu)
        .def("read_gps",  &read_gps)
        .def("read_baro", &read_baro)

        .def("terrain_height_at",[](const DronePhysics& dp,double x,double y){
            return dp.terrain().height_at(x,y);})
        .def("altitude_agl",[](const DronePhysics& dp){
            const auto& s=dp.state();
            return s.pos.z-dp.terrain().height_at(s.pos.x,s.pos.y);});

    // ── DroneSwarm ───────────────────────────────────────────────────────────
    py::class_<DroneSwarm>(m,"DroneSwarm")
        .def(py::init<>())
        .def("add_drone",&DroneSwarm::add_drone,
             py::arg("config")=DroneConfig{},py::arg("base_seed")=42ULL)
        .def("n_drones", &DroneSwarm::n_drones)
        .def("drone",[](DroneSwarm& sw,int idx)->DronePhysics&{return sw.drone(idx);},
             py::arg("idx"),py::return_value_policy::reference_internal)
        .def("step_all",    &DroneSwarm::step_all,   py::arg("dt"))
        .def("spawn_grid",  &DroneSwarm::spawn_grid,
             py::arg("spacing")=5.0,py::arg("altitude")=100.0)
        .def("all_turbulence",&DroneSwarm::all_turbulence,
             "Returns flat [N*3] array of turbulence wind at each drone position");

    // ── Thread pool & CPU utilities ──────────────────────────────────────────
    m.def("pool_stats", &pool_stats,
          "Returns dict with ThreadPool performance counters");
    m.def("cpu_info",   &cpu_info,
          "Returns dict with detected CPU SIMD capabilities");
    m.def("pool_n_threads",[](){return global_pool().n_threads();},
          "Number of worker threads in global pool");
}
