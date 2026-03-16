#pragma once
/**
 * motor_dynamics.hpp
 * Full motor / rotor pipeline:
 *   PWM command → ESC → motor electrical model → shaft RPM
 *   → aerodynamic thrust (BET) → force body frame → world frame
 *
 * Models:
 *   Motor  : DC motor (Kv, R, L, J_rotor) with back-EMF & saturation
 *   Rotor  : Blade Element Theory (BET) + Momentum Theory correction
 *            + induced velocity (actuator disk, Glauert inflow)
 *   ESC    : 1st-order lag + desaturation anti-windup
 *   Frame  : configurable N-rotor geometry, arbitrary tilt angles
 */

#include "math_types.hpp"
#include "atmosphere.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Motor electrical parameters
// ─────────────────────────────────────────────────────────────────────────────
struct MotorSpec {
    double Kv{920.0};          // RPM/V motor velocity constant
    double Kt;                 // Nm/A  (= 60/(2π·Kv))
    double R_winding{0.12};    // Ω   winding resistance
    double L_winding{30e-6};   // H   winding inductance
    double J_rotor{1.5e-5};    // kg·m²  rotor + prop inertia
    double V_battery{11.1};    // V   supply voltage (3S LiPo default)
    double eta_mech{0.92};     // mechanical efficiency

    MotorSpec() { Kt = 60.0 / (2.0*M_PI*Kv); }
    explicit MotorSpec(double kv_) : Kv(kv_) { Kt = 60.0/(2.0*M_PI*Kv); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Rotor aerodynamic parameters
// ─────────────────────────────────────────────────────────────────────────────
struct RotorSpec {
    double radius{0.127};      // m  tip radius
    double chord{0.013};       // m  mean chord
    int    n_blades{2};
    double pitch_angle{15.0};  // deg collective (fixed pitch)
    double Cd0{0.012};         // profile drag coefficient
    double Cla{5.73};          // lift curve slope dCl/dα [1/rad] (2π for thin aerofoil)
    double solidity;           // σ = n_blades * chord / (π * R)

    RotorSpec() { solidity = n_blades * chord / (M_PI * radius); }

    // Thrust coefficient Ct from Blade Element + Momentum Theory
    // Ct = (σ·Cla/4) * (θ * (2/3) * tip_speed_ratio - λ_i)
    // simplified closed-form for hover:
    double hover_Ct(double lambda_i) const noexcept {
        double theta = pitch_angle * M_PI / 180.0;
        return (solidity * Cla / 4.0) *
               (theta * (2.0/3.0) - lambda_i);
    }

    double Ct{0.0047};   // pre-computed thrust coefficient (dimensionless)
    double Cq{0.00073};  // torque coefficient
};

// ─────────────────────────────────────────────────────────────────────────────
// Single motor state (per-rotor)
// ─────────────────────────────────────────────────────────────────────────────
struct MotorState {
    double rpm{0.0};           // shaft RPM
    double omega{0.0};         // rad/s
    double current{0.0};       // A
    double torque_elec{0.0};   // Nm  electromagnetic torque
    double torque_aero{0.0};   // Nm  aerodynamic reaction torque
    double thrust{0.0};        // N
    double power{0.0};         // W
    double temperature{25.0};  // °C (for thermal model)
};

// ─────────────────────────────────────────────────────────────────────────────
// Single motor model — DC brushless with full electrical dynamics
// ─────────────────────────────────────────────────────────────────────────────
class MotorModel {
    const MotorSpec&  spec_;
    const RotorSpec&  rotor_;
    MotorState        state_{};
    double            i_state_{0.0};  // current integrator state

public:
    MotorModel(const MotorSpec& m, const RotorSpec& r) : spec_(m), rotor_(r) {}

    const MotorState& state() const noexcept { return state_; }
    MotorState&       state()       noexcept { return state_; }

    /**
     * step(): advance motor state by dt seconds given PWM throttle [0,1]
     * and air density rho [kg/m³].
     *
     * Pipeline:
     *   throttle → V_esc → back-EMF → di/dt → I → T_elec
     *   → dω/dt (including aero load) → RPM → thrust
     */
    void step(double throttle, double rho, double dt) noexcept {
        throttle = std::clamp(throttle, 0.0, 1.0);
        double V_in   = throttle * spec_.V_battery;
        double omega  = state_.omega;

        // Back-EMF
        double V_bemf = omega / (spec_.Kv * 2.0*M_PI/60.0);  // Kv in RPM/V → rad/s/V
        double V_bemf2 = omega / (spec_.Kv * (2.0*M_PI/60.0));

        // Electrical: L·di/dt = V - R·i - Ke·ω
        double Ke   = 1.0 / (spec_.Kv * (2.0*M_PI/60.0));    // V·s/rad
        double di_dt = (V_in - spec_.R_winding * i_state_ - Ke * omega) / spec_.L_winding;
        i_state_ += di_dt * dt;
        i_state_  = std::clamp(i_state_, 0.0, spec_.V_battery / spec_.R_winding);

        // Electromagnetic torque  T_e = Kt * I
        double T_elec = spec_.Kt * i_state_ * spec_.eta_mech;

        // Aerodynamic thrust & torque (actuator disk)
        double T_thrust = rotor_.Ct * rho * omega*omega * std::pow(rotor_.radius, 4);
        double T_aero   = rotor_.Cq * rho * omega*omega * std::pow(rotor_.radius, 5);

        // Newton's 2nd for rotor: J·dω/dt = T_elec - T_aero
        double dw_dt = (T_elec - T_aero) / spec_.J_rotor;
        state_.omega = std::max(0.0, omega + dw_dt * dt);
        state_.rpm   = state_.omega * 60.0 / (2.0*M_PI);

        // Glauert inflow correction (hover → forward flight)
        // λ_i = sqrt(Ct/2) for hover; use simplified constant
        double lambda_i = std::sqrt(std::max(rotor_.Ct, 0.0) * 0.5);
        double Ct_actual = rotor_.hover_Ct(lambda_i);
        Ct_actual = std::max(Ct_actual, 0.0);

        state_.thrust     = Ct_actual * rho * state_.omega*state_.omega
                          * std::pow(rotor_.radius, 4);
        state_.torque_elec = T_elec;
        state_.torque_aero = T_aero;
        state_.current    = i_state_;
        state_.power      = V_in * i_state_;

        // Simple thermal model: T += (I²R - k_cool*(T-25)) * dt
        double heat_in   = i_state_*i_state_*spec_.R_winding;
        double k_cool    = 0.05;
        state_.temperature += (heat_in - k_cool*(state_.temperature - 25.0)) * dt;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Rotor descriptor — position, axis, spin direction in body frame
// ─────────────────────────────────────────────────────────────────────────────
struct RotorMount {
    Vec3   position;         // m from CoM in body frame
    Vec3   axis;             // unit vector thrust axis (body frame, usually +Z)
    double spin_dir{1.0};    // +1 CW, -1 CCW viewed from above
    double tilt_x{0.0};      // rad — rotor tilt for hexacopter vectoring
    double tilt_y{0.0};
};

// ─────────────────────────────────────────────────────────────────────────────
// RotorFrame — defines N-rotor geometry with standard configs
// ─────────────────────────────────────────────────────────────────────────────
struct RotorFrame {
    std::vector<RotorMount> rotors;

    static RotorFrame make_quadrotor_x(double arm_len = 0.225) {
        RotorFrame f;
        double a = arm_len * std::cos(M_PI/4.0);
        f.rotors = {
            { { a,  a, 0}, {0,0,1},  1.0},  // FL  CW
            { {-a,  a, 0}, {0,0,1}, -1.0},  // BL  CCW
            { {-a, -a, 0}, {0,0,1},  1.0},  // BR  CW
            { { a, -a, 0}, {0,0,1}, -1.0},  // FR  CCW
        };
        return f;
    }

    static RotorFrame make_hexarotor(double arm_len = 0.25) {
        RotorFrame f;
        for(int i=0;i<6;++i){
            double ang = i * M_PI/3.0;
            f.rotors.push_back({
                {arm_len*std::cos(ang), arm_len*std::sin(ang), 0},
                {0,0,1},
                (i%2==0)?1.0:-1.0
            });
        }
        return f;
    }

    static RotorFrame make_octorotor(double arm_len = 0.28) {
        RotorFrame f;
        for(int i=0;i<8;++i){
            double ang = i * M_PI/4.0;
            f.rotors.push_back({
                {arm_len*std::cos(ang), arm_len*std::sin(ang), 0},
                {0,0,1},
                (i%2==0)?1.0:-1.0
            });
        }
        return f;
    }

    int n() const noexcept { return static_cast<int>(rotors.size()); }
};


// ─────────────────────────────────────────────────────────────────────────────
// DrivetrainSystem — all motors + geometry, computes body-frame forces/torques
// ─────────────────────────────────────────────────────────────────────────────
class DrivetrainSystem {
    RotorFrame             frame_;
    RotorSpec              rotor_spec_;
    MotorSpec              motor_spec_;
    std::vector<MotorModel> motors_;

    double max_throttle_{1.0};

public:
    DrivetrainSystem(RotorFrame frame, MotorSpec ms, RotorSpec rs)
        : frame_(std::move(frame)), rotor_spec_(rs), motor_spec_(ms)
    {
        motors_.reserve(frame_.n());
        for(int i=0;i<frame_.n();++i)
            motors_.emplace_back(motor_spec_, rotor_spec_);
    }

    int n_rotors() const noexcept { return frame_.n(); }

    const MotorState& motor_state(int i) const { return motors_.at(i).state(); }

    std::vector<double> rpms() const noexcept {
        std::vector<double> r; r.reserve(motors_.size());
        for(auto& m : motors_) r.push_back(m.state().rpm);
        return r;
    }

    double total_power() const noexcept {
        double p=0; for(auto& m:motors_) p+=m.state().power; return p;
    }

    /**
     * step()
     * throttles[i] ∈ [0,1] — output of controller mixer
     * rho           — air density [kg/m³]
     * dt            — timestep [s]
     *
     * Computes:
     *   Per-motor: RPM, thrust (body frame)
     *   Returns: F_body [N],  M_body [Nm]
     */
    std::pair<Vec3,Vec3> step(const std::vector<double>& throttles,
                               double rho, double dt) noexcept {
        Vec3 F_body{}, M_body{};

        for(int i=0;i<frame_.n();++i){
            double thr = (i < static_cast<int>(throttles.size()))
                       ? std::clamp(throttles[i], 0.0, max_throttle_)
                       : 0.0;
            motors_[i].step(thr, rho, dt);
            const auto& ms = motors_[i].state();

            // Thrust vector in body frame (along rotor axis)
            const RotorMount& rm = frame_.rotors[i];
            // Apply tilt (vectored thrust for special configs)
            Vec3 thrust_axis = rm.axis;
            if(std::abs(rm.tilt_x) > 1e-6 || std::abs(rm.tilt_y) > 1e-6) {
                Quat tilt = Quat::from_euler_zyx(rm.tilt_x, rm.tilt_y, 0);
                thrust_axis = tilt.rotate(rm.axis);
            }

            Vec3 F_i  = thrust_axis * ms.thrust;
            F_body   += F_i;

            // Moment arm torque: r × F
            M_body   += rm.position.cross(F_i);

            // Reaction torque from rotor spin
            M_body.z += rm.spin_dir * ms.torque_aero;
        }
        return std::make_pair(F_body, M_body);
    }

    // Force body frame → world frame via attitude quaternion
    static std::pair<Vec3,Vec3> to_world(const Vec3& F_body, const Vec3& M_body,
                                          const Quat& att) noexcept {
        return std::make_pair(att.rotate(F_body), att.rotate(M_body));
    }
};
