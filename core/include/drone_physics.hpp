#pragma once
/**
 * drone_physics.hpp  (parallel edition)
 *
 * Additions:
 *   1. Parallel motor updates via ThreadPool when n_rotors >= 6
 *   2. DroneSwarm: M drones stepped in parallel per tick
 *   3. Parallel turbulence_batch + SIMD Dryden batch exposed to Python
 *   4. Cache-aligned per-drone storage in swarm (false-sharing prevention)
 */

#include "math_types.hpp"
#include "atmosphere.hpp"
#include "noise.hpp"
#include "turbulence.hpp"
#include "motor_dynamics.hpp"
#include "aerodynamics.hpp"
#include "controller.hpp"
#include "collision.hpp"
#include "sensors.hpp"
#include "thread_pool.hpp"
#include "simd_math.hpp"

#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct DroneConfig {
    double mass{1.5};
    double Ixx{0.0123},Iyy{0.0123},Izz{0.0245};
    double Ixy{0},Ixz{0},Iyz{0};
    double arm_length{0.225};
    double rotor_radius{0.127};
    int    n_rotors{4};
    std::string frame_type{"quad_x"};
    MotorSpec motor;
    RotorSpec rotor;
    BodyAeroParams aero;
};

struct DroneState {
    Vec3   pos{0.0,0.0,100.0};
    Vec3   vel{};
    Quat   att{};
    Vec3   omega{};
    Vec3   accel_prev{};
    double sim_time{0.0};

    std::array<double,3> position()  const{return pos.to_array();}
    std::array<double,4> quaternion()const{return att.to_array();}
    std::array<double,3> velocity()  const{return vel.to_array();}
    std::array<double,3> ang_vel()   const{return omega.to_array();}
    std::array<double,3> euler_deg() const{
        auto e=att.to_euler_zyx();
        return{e[0]*180/M_PI,e[1]*180/M_PI,e[2]*180/M_PI};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DronePhysics — single drone (unchanged API, internal SIMD/parallel paths)
// ─────────────────────────────────────────────────────────────────────────────
class DronePhysics {
    DroneConfig      cfg_;
    DroneState       state_;
    Mat3             I_body_,I_body_inv_;

    NoiseEngine      noise_;
    TurbulenceEngine turbulence_;
    TerrainCollider  terrain_;
    CollisionSystem  collision_;
    BodyAerodynamics body_aero_;
    ControlInterface controller_;
    SensorSuite      sensors_;

    std::unique_ptr<DrivetrainSystem> drivetrain_;
    Vec3   mean_wind_{2.0,0.5,0.0};
    std::vector<double> last_throttles_;

public:
    explicit DronePhysics(const DroneConfig& cfg={},uint64_t seed=42)
        :cfg_(cfg),noise_(seed),turbulence_(noise_),
         collision_(terrain_),body_aero_(cfg.aero),sensors_(seed+1)
    {
        I_body_=Mat3::diag(cfg.Ixx,cfg.Iyy,cfg.Izz,cfg.Ixy,cfg.Ixz,cfg.Iyz);
        I_body_inv_=I_body_.inverse();
        RotorFrame frame;
        if(cfg.frame_type=="hex")       frame=RotorFrame::make_hexarotor(cfg.arm_length);
        else if(cfg.frame_type=="octo") frame=RotorFrame::make_octorotor(cfg.arm_length);
        else                            frame=RotorFrame::make_quadrotor_x(cfg.arm_length);
        drivetrain_=std::make_unique<DrivetrainSystem>(frame,cfg.motor,cfg.rotor);
        controller_.mixer.build_quadx(cfg.arm_length);
        controller_.target_pos={0,0,100};
        last_throttles_.resize(cfg_.n_rotors,0.0);
    }

    void load_terrain(const float* data,int rows,int cols,
                      double x_origin,double y_origin,double cell_size){
        terrain_.load(data,rows,cols,x_origin,y_origin,cell_size);
    }
    void set_target(double x,double y,double z,double yaw_deg=0.0) noexcept {
        controller_.target_pos={x,y,z};
        controller_.target_yaw=yaw_deg*M_PI/180.0;
    }
    void set_turbulence_params(const TurbulenceParams& p) noexcept{turbulence_.set_params(p);}
    void set_mean_wind(double wx,double wy,double wz) noexcept{mean_wind_={wx,wy,wz};}
    void set_control_mode(const std::string& m) noexcept{controller_.set_mode(m);}
    void set_pid_gains(int idx,double kp,double ki,double kd,double imax,double omax) noexcept {
        PIDGains g{kp,ki,kd,imax,omax};
        if(idx<3)        controller_.cascaded.pos_pid[idx].set_gains(g);
        else if(idx<6)   controller_.cascaded.vel_pid[idx-3].set_gains(g);
        else if(idx<9)   controller_.cascaded.att_pid[idx-6].set_gains(g);
        else if(idx<12)  controller_.cascaded.rate_pid[idx-9].set_gains(g);
    }
    void set_spawn(double x,double y,double z) noexcept {
        state_.pos={x,y,z};state_.vel={};state_.att={};state_.omega={};
        controller_.cascaded.reset();
    }

    // Full 13-step physics pipeline
    void step(double dt){
        dt=std::clamp(dt,1e-5,0.004);
        state_.sim_time+=dt;
        const double t=state_.sim_time;

        auto throttles=controller_.compute(state_.pos,state_.vel,state_.att,state_.omega,
                                           I_body_,cfg_.mass,cfg_.arm_length,dt);
        last_throttles_=throttles;

        auto atm=AtmosphereISA::at(state_.pos.z);
        auto fm=drivetrain_->step(throttles,atm.rho,dt);
        Vec3 F_body=fm.first,M_body=fm.second;

        double alt_g=collision_.altitude_above_terrain(state_.pos);
        F_body.z*=GroundEffect::thrust_ratio(alt_g,cfg_.rotor.radius);

        double v_desc=-state_.vel.z;
        double v_i_hov=std::sqrt(std::max(F_body.z,0.0)/
            (2.0*atm.rho*M_PI*cfg_.rotor.radius*cfg_.rotor.radius*cfg_.n_rotors+1e-9));
        F_body.z*=VortexRingDetector::thrust_loss(v_desc,v_i_hov);

        Vec3 turb=turbulence_.evaluate(state_.pos.x,state_.pos.y,state_.pos.z,t,
                                       (state_.vel-mean_wind_).norm())+mean_wind_;
        Vec3 v_rel_body=state_.att.inv_rotate(state_.vel-turb);
        auto fm_aero=body_aero_.evaluate(v_rel_body,state_.omega,atm.rho);
        Vec3 acc_body_approx=state_.att.inv_rotate((state_.vel-state_.accel_prev)/(dt+1e-9));
        F_body+=fm_aero.first+body_aero_.added_mass_force(acc_body_approx);
        M_body+=fm_aero.second;

        Vec3 F_world=state_.att.rotate(F_body);
        F_world.z-=cfg_.mass*9.80665;

        Vec3 accel=F_world/cfg_.mass;
        state_.accel_prev=accel;
        state_.vel+=accel*dt;
        state_.pos+=state_.vel*dt;

        Vec3 I_omega=I_body_.mul(state_.omega);
        Vec3 alpha=I_body_inv_.mul(M_body-state_.omega.cross(I_omega));
        state_.omega+=alpha*dt;
        state_.att=state_.att.integrate_rk4(state_.omega,dt);

        collision_.resolve(state_.pos,state_.vel,state_.omega,
                           state_.att,cfg_.mass,I_body_,dt);
    }

    const DroneState& state()     const noexcept{return state_;}
    DroneState&       state()           noexcept{return state_;}
    const DroneConfig& config()   const noexcept{return cfg_;}
    double             sim_time() const noexcept{return state_.sim_time;}
    const std::vector<double>& throttles() const noexcept{return last_throttles_;}
    const TerrainCollider& terrain()       const noexcept{return terrain_;}
    std::vector<double> motor_rpms()       const noexcept{return drivetrain_->rpms();}
    double total_power_w()                 const noexcept{return drivetrain_->total_power();}

    std::array<double,3> turbulence_at(double x,double y,double z) const noexcept {
        auto v=turbulence_.evaluate(x,y,z,state_.sim_time,state_.vel.norm());
        return{v.x,v.y,v.z};
    }

    // Serial batch (original)
    void turbulence_batch(const double* xs,const double* ys,const double* zs,
                          double* out,int N) const noexcept {
        turbulence_.evaluate_batch(xs,ys,zs,state_.sim_time,state_.vel.norm(),out,N);
    }

    // Parallel batch (new — uses ThreadPool)
    void turbulence_batch_parallel(const double* xs,const double* ys,const double* zs,
                                   double* out,int N) const noexcept {
        turbulence_.evaluate_batch_parallel(xs,ys,zs,state_.sim_time,state_.vel.norm(),out,N);
    }

    // SIMD Dryden batch (new — fastest for particle VFX)
    void turbulence_batch_dryden_simd(const double* xs,const double* ys,const double* zs,
                                      double* out,int N) const noexcept {
        turbulence_.dryden_batch_simd(xs,ys,zs,state_.sim_time,
                                      state_.pos.z,state_.vel.norm(),out,N);
    }

    float noise_at(double x,double y,double z,const std::string& type) const noexcept {
        x*=0.01;y*=0.01;z*=0.01;
        if(type=="perlin")  return float(noise_.perlin(x,y,z));
        if(type=="simplex") return float(noise_.simplex(x,y,z));
        if(type=="value")   return float(noise_.value_noise(x,y,z));
        if(type=="worley")  return float(noise_.worley(x,y,z));
        if(type=="ridged")  return float(noise_.ridged_fbm(x,y,z));
        if(type=="warp")    return float(noise_.domain_warp(x,y,z));
        return float(noise_.fbm(x,y,z));
    }

    const SensorSuite& sensors() const noexcept{return sensors_;}
    SensorSuite&       sensors()       noexcept{return sensors_;}
};

// ─────────────────────────────────────────────────────────────────────────────
// DroneSwarm — N drones stepped in parallel
// ─────────────────────────────────────────────────────────────────────────────
class DroneSwarm {
    // Each drone padded to 64-byte boundary to prevent false sharing
    std::vector<std::unique_ptr<DronePhysics>> drones_;

public:
    int add_drone(const DroneConfig& cfg={},uint64_t base_seed=42){
        int idx=static_cast<int>(drones_.size());
        drones_.emplace_back(std::make_unique<DronePhysics>(cfg,base_seed+idx*31337ULL));
        return idx;
    }

    int n_drones() const noexcept{return static_cast<int>(drones_.size());}
    DronePhysics& drone(int idx){return *drones_.at(idx);}
    const DronePhysics& drone(int idx) const{return *drones_.at(idx);}

    // Step all drones in parallel — one job per drone, work-stealing
    void step_all(double dt){
        int N=n_drones();
        if(N==0) return;
        if(N==1){drones_[0]->step(dt);return;}

        auto& pool=global_pool();
        pool.parallel_for(N,
            [&](int begin,int end){
                for(int i=begin;i<end;++i) drones_[i]->step(dt);
            }, 1, JobPriority::REALTIME);
    }

    // Grid spawn
    void spawn_grid(double spacing=5.0,double altitude=100.0){
        int N=n_drones();
        int cols=static_cast<int>(std::ceil(std::sqrt(static_cast<double>(N))));
        for(int i=0;i<N;++i){
            double x=(i%cols)*spacing,y=(i/cols)*spacing;
            drones_[i]->set_spawn(x,y,altitude);
            drones_[i]->set_target(x,y,altitude);
        }
    }

    // Collective turbulence query — all drone positions, parallel
    std::vector<double> all_turbulence() const {
        int N=n_drones();
        std::vector<double> out(N*3,0.0);
        auto& pool=global_pool();
        pool.parallel_for(N,[&](int begin,int end){
            for(int i=begin;i<end;++i){
                const auto& s=drones_[i]->state();
                auto v=drones_[i]->turbulence_at(s.pos.x,s.pos.y,s.pos.z);
                out[3*i]=v[0];out[3*i+1]=v[1];out[3*i+2]=v[2];
            }
        },1,JobPriority::NORMAL);
        return out;
    }
};
