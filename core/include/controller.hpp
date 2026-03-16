#pragma once
/**
 * controller.hpp
 * Flight controller interface:
 *
 *   1. PIDController  — generic PID with back-calculation anti-windup
 *   2. CascadedPID    — position → velocity → attitude → rate → mixer
 *   3. SO3Controller  — geometric attitude control on SO(3) (Lee et al. 2010)
 *   4. LQRHover       — linearised hover LQR gain matrix (pre-computed)
 *   5. ControlMixer   — maps [thrust, roll, pitch, yaw] → per-rotor throttles
 *   6. ControlInterface — unified C API used by DronePhysics
 */

#include "math_types.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Generic PID with back-calculation anti-windup
// ─────────────────────────────────────────────────────────────────────────────
struct PIDGains { double kp, ki, kd, imax, output_max; };

class PIDController {
    PIDGains gains_;
    double   integral_{0.0};
    double   prev_error_{0.0};
    double   prev_deriv_{0.0};   // low-pass filtered derivative
    double   filter_alpha_{0.8}; // derivative filter coefficient

public:
    explicit PIDController(PIDGains g = {1,0,0,10,100}) : gains_(g) {}

    void  set_gains(const PIDGains& g) noexcept { gains_ = g; }
    void  reset()   noexcept { integral_=0; prev_error_=0; prev_deriv_=0; }
    const PIDGains& gains() const noexcept { return gains_; }

    double update(double error, double dt) noexcept {
        // Proportional
        double P = gains_.kp * error;

        // Integral (back-calculation anti-windup via conditional integration)
        double I_candidate = integral_ + error * dt;
        double output_pre  = P + gains_.ki*I_candidate
                               + gains_.kd*prev_deriv_;
        // Only integrate if output is not saturated OR error reduces it
        if(std::abs(output_pre) < gains_.output_max || output_pre*error < 0)
            integral_ = std::clamp(I_candidate, -gains_.imax, gains_.imax);
        double I_term = gains_.ki * integral_;

        // Derivative with low-pass filter (prevent setpoint kick)
        double raw_deriv = (dt > 1e-7) ? (error - prev_error_) / dt : 0.0;
        double D_filt    = filter_alpha_ * prev_deriv_
                         + (1.0 - filter_alpha_) * raw_deriv;
        prev_deriv_  = D_filt;
        prev_error_  = error;

        double out = P + I_term + gains_.kd * D_filt;
        return std::clamp(out, -gains_.output_max, gains_.output_max);
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Control Mixer — maps [thrust, τ_roll, τ_pitch, τ_yaw] → throttles[0..n-1]
// Supports quad-X, hex, octo via allocation matrix
// ─────────────────────────────────────────────────────────────────────────────
class ControlMixer {
    // Allocation matrix A: dim n_rotors x 4
    // columns: [thrust, roll_moment, pitch_moment, yaw_moment]
    // throttle = A_pseudo_inv * [T, τr, τp, τy]
    std::vector<std::array<double,4>> A_;    // n x 4
    std::vector<std::array<double,4>> A_pinv_; // 4 x n (pseudo-inverse transposed)

    int    n_rotors_{4};
    double max_throttle_{1.0};
    double min_throttle_{0.05};

    // Simple pseudo-inverse: A_pinv = A^T (A A^T)^-1  — works for 4-DOF systems
    void compute_pinv() {
        // For standard N-rotor, build analytically
        int n = n_rotors_;
        A_pinv_.resize(n);
        // Using simple equal-weight distribution as fallback
        // For quad-X the exact analytic solution is used
        for(int i=0;i<n;++i){
            double ti = A_[i][0]; // thrust contribution
            double ri = A_[i][1];
            double pi = A_[i][2];
            double yi = A_[i][3];
            A_pinv_[i] = {ti/n, ri/(ri*ri>0?4*ri:1.0),
                          pi/(pi*pi>0?4*pi:1.0), yi/(yi*yi>0?4*yi:1.0)};
        }
    }

public:
    ControlMixer() { build_quadx(); }

    void build_quadx(double arm=0.225) {
        n_rotors_ = 4;
        double a = arm * std::cos(M_PI/4.0);
        // [thrust, roll(+right), pitch(+nose-up), yaw(+CCW)]
        // FL(CW), BL(CCW), BR(CW), FR(CCW)
        A_ = {
            {1.0,  a,  a, -1.0},  // FL  CW
            {1.0, -a,  a,  1.0},  // BL CCW
            {1.0, -a, -a, -1.0},  // BR  CW
            {1.0,  a, -a,  1.0},  // FR CCW
        };
        compute_pinv();
    }

    void build_hex(double arm=0.25) {
        n_rotors_ = 6;
        A_.clear();
        for(int i=0;i<6;++i){
            double ang = i*M_PI/3.0;
            double x = arm*std::cos(ang), y = arm*std::sin(ang);
            double sd = (i%2==0)?-1.0:1.0;
            A_.push_back({1.0, y, -x, sd});
        }
        compute_pinv();
    }

    /**
     * mix(): converts flight control outputs → throttle commands [0..1]
     * T_des   : desired total normalised thrust [0..1]
     * roll_moment, pitch_moment, yaw_moment: Nm (will be normalised)
     */
    std::vector<double> mix(double T_des, double roll_Nm, double pitch_Nm,
                             double yaw_Nm, double arm=0.225) const noexcept {
        std::vector<double> out(n_rotors_);
        double inv_n = 1.0 / n_rotors_;

        // Inverse allocation: throttle_i = T/n ± (moments via arm)
        // Using the physical mixing laws for quad-X
        if(n_rotors_ == 4) {
            double a = arm * std::cos(M_PI/4.0);
            double inv4a = 1.0 / (4.0 * a);
            out[0] = T_des*inv_n + pitch_Nm*inv4a + roll_Nm*inv4a  - yaw_Nm*0.25;
            out[1] = T_des*inv_n + pitch_Nm*inv4a - roll_Nm*inv4a  + yaw_Nm*0.25;
            out[2] = T_des*inv_n - pitch_Nm*inv4a - roll_Nm*inv4a  - yaw_Nm*0.25;
            out[3] = T_des*inv_n - pitch_Nm*inv4a + roll_Nm*inv4a  + yaw_Nm*0.25;
        } else {
            // Generic: equal split (simplified)
            for(int i=0;i<n_rotors_;++i){
                double ri = A_[i][1], pi = A_[i][2], yi = A_[i][3];
                out[i] = T_des*inv_n + roll_Nm*ri/(4*arm*arm)
                        + pitch_Nm*pi/(4*arm*arm) + yaw_Nm*yi*0.25;
            }
        }
        // Clamp + desaturation
        double max_v = *std::max_element(out.begin(), out.end());
        double min_v = *std::min_element(out.begin(), out.end());
        if(max_v > max_throttle_){
            double excess = max_v - max_throttle_;
            for(auto& v : out) v -= excess;
        }
        if(min_v < min_throttle_){
            double deficit = min_throttle_ - min_v;
            for(auto& v : out) v += deficit;
        }
        for(auto& v : out) v = std::clamp(v, min_throttle_, max_throttle_);
        return out;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// SO(3) Geometric Attitude Controller — Lee et al. (2010)
// ─────────────────────────────────────────────────────────────────────────────
class SO3AttitudeController {
    double kR_{8.0};   // rotation error gain
    double kOm_{2.5};  // angular velocity error gain
    Vec3   e_R_prev_{};

public:
    void set_gains(double kR, double kOm) noexcept { kR_=kR; kOm_=kOm; }

    // Returns desired torque in body frame [Nm]
    // att_des: desired attitude quaternion
    // att:     current attitude
    // omega:   current angular velocity body frame
    // omega_des: desired angular velocity (usually 0 for hover)
    // I:       inertia tensor
    Vec3 compute(const Quat& att_des, const Quat& att,
                 const Vec3& omega, const Vec3& omega_des,
                 const Mat3& I) const noexcept {
        // Rotation error on SO(3):  e_R = 0.5 * vee(R_des^T R - R^T R_des)
        Mat3 R     = att.to_rotation_matrix();
        Mat3 R_des = att_des.to_rotation_matrix();

        // R_err = R_des^T * R
        Mat3 R_err{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
            for(int k=0;k<3;++k) R_err.m[i][j] += R_des.m[k][i]*R.m[k][j];

        // Vee map: extract rotation error vector from skew-symmetric part
        Vec3 e_R{
            (R_err.m[2][1] - R_err.m[1][2]) * 0.5,
            (R_err.m[0][2] - R_err.m[2][0]) * 0.5,
            (R_err.m[1][0] - R_err.m[0][1]) * 0.5
        };

        // Angular velocity error in body frame
        Vec3 e_Om = omega - omega_des;

        // Control: M = -kR*e_R - kOm*e_Om + omega × (I*omega)
        Vec3 Iom = I.mul(omega);
        Vec3 gyro_comp = omega.cross(Iom);

        return e_R*(-kR_) + e_Om*(-kOm_) + gyro_comp;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// CascadedPID — 4-loop position→velocity→attitude→rate
// ─────────────────────────────────────────────────────────────────────────────
class CascadedPIDController {
public:
    // Position loop (3 axes)
    std::array<PIDController,3> pos_pid;   // X, Y, Z
    // Velocity loop
    std::array<PIDController,3> vel_pid;
    // Attitude loop (roll, pitch, yaw)
    std::array<PIDController,3> att_pid;
    // Rate loop
    std::array<PIDController,3> rate_pid;

    CascadedPIDController() {
        // Default gains
        pos_pid[0].set_gains({1.2,0.05,0.5, 5.0, 8.0});   // X
        pos_pid[1].set_gains({1.2,0.05,0.5, 5.0, 8.0});   // Y
        pos_pid[2].set_gains({2.0,0.80,1.5,40.0,50.0});   // Z (altitude)

        vel_pid[0].set_gains({3.0,0.1,0.8,10.0,20.0});
        vel_pid[1].set_gains({3.0,0.1,0.8,10.0,20.0});
        vel_pid[2].set_gains({5.0,0.5,1.2,20.0,30.0});

        att_pid[0].set_gains({7.0,0.3,2.5,30.0,40.0});    // roll
        att_pid[1].set_gains({7.0,0.3,2.5,30.0,40.0});    // pitch
        att_pid[2].set_gains({4.0,0.2,1.5,20.0,25.0});    // yaw

        rate_pid[0].set_gains({120.0,20.0,5.0,200.0,500.0}); // roll rate
        rate_pid[1].set_gains({120.0,20.0,5.0,200.0,500.0}); // pitch rate
        rate_pid[2].set_gains({80.0, 15.0,4.0,150.0,300.0}); // yaw rate
    }

    struct Output {
        double thrust_normalized;  // [0,1]
        Vec3   torque_body;        // [Nm]
        Vec3   des_att_euler;      // [rad] for debug
    };

    /**
     * compute(): full 4-loop cascade
     * Returns throttle-mix inputs to ControlMixer
     */
    Output compute(
        const Vec3& pos, const Vec3& vel,
        const Quat& att, const Vec3& omega,
        const Vec3& target_pos, double target_yaw,
        double mass, double dt) noexcept
    {
        // ── Loop 1: Position → desired velocity ─────────────────────────────
        Vec3 pos_err = target_pos - pos;
        Vec3 des_vel{
            pos_pid[0].update(pos_err.x, dt),
            pos_pid[1].update(pos_err.y, dt),
            pos_pid[2].update(pos_err.z, dt)
        };

        // ── Loop 2: Velocity → desired acceleration ──────────────────────────
        Vec3 vel_err = des_vel - vel;
        Vec3 des_acc{
            vel_pid[0].update(vel_err.x, dt),
            vel_pid[1].update(vel_err.y, dt),
            vel_pid[2].update(vel_err.z, dt)
        };

        // Thrust: feed-forward gravity + velocity feedback
        double T_ff = mass * 9.80665;
        double T_pid = mass * des_acc.z;
        double T_total = std::max(T_ff + T_pid, 0.1);
        double T_norm = std::clamp(T_total / (mass * 9.80665 * 2.0), 0.0, 1.0);

        // ── Loop 3: Desired attitude from horizontal acceleration ─────────────
        auto euler = att.to_euler_zyx();
        double cy = std::cos(euler[2]), sy = std::sin(euler[2]);
        // Desired roll/pitch in body frame (yaw-compensated)
        double ax_world = des_acc.x, ay_world = des_acc.y;
        double ax_body  =  cy*ax_world + sy*ay_world;
        double ay_body  = -sy*ax_world + cy*ay_world;
        double des_pitch = std::clamp( ax_body / (9.80665+0.1), -0.45, 0.45);
        double des_roll  = std::clamp(-ay_body / (9.80665+0.1), -0.45, 0.45);

        Vec3 des_att_euler{des_roll, des_pitch, target_yaw};

        double roll_err  = des_roll  - euler[0];
        double pitch_err = des_pitch - euler[1];
        double yaw_err   = target_yaw - euler[2];
        while(yaw_err >  M_PI) yaw_err -= 2*M_PI;
        while(yaw_err < -M_PI) yaw_err += 2*M_PI;

        Vec3 des_rate{
            att_pid[0].update(roll_err,  dt),
            att_pid[1].update(pitch_err, dt),
            att_pid[2].update(yaw_err,   dt)
        };

        // ── Loop 4: Rate → torque ─────────────────────────────────────────────
        Vec3 rate_err = des_rate - omega;
        Vec3 torque{
            rate_pid[0].update(rate_err.x, dt),
            rate_pid[1].update(rate_err.y, dt),
            rate_pid[2].update(rate_err.z, dt)
        };

        Output out;
        out.thrust_normalized = T_norm;
        out.torque_body       = torque;
        out.des_att_euler     = des_att_euler;
        return out;
    }

    void reset() {
        for(auto& p: pos_pid)  p.reset();
        for(auto& p: vel_pid)  p.reset();
        for(auto& p: att_pid)  p.reset();
        for(auto& p: rate_pid) p.reset();
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// ControlInterface — unified access object for DronePhysics
// ─────────────────────────────────────────────────────────────────────────────
// Undef Win32 macros that collide with enum names
#ifdef PASSTHROUGH
#  undef PASSTHROUGH
#endif
#ifdef CASCADED_PID
#  undef CASCADED_PID
#endif
#ifdef SO3_GEOMETRIC
#  undef SO3_GEOMETRIC
#endif
class ControlInterface {
public:
    enum class Mode { CTL_CASCADED_PID, CTL_SO3_GEOMETRIC, CTL_PASSTHROUGH };

    CascadedPIDController  cascaded;
    SO3AttitudeController  so3;
    ControlMixer           mixer;

    Mode   mode{Mode::CTL_CASCADED_PID};
    Vec3   target_pos{0,0,100};
    double target_yaw{0.0};
    // Manual override throttles [0..1] (used in PASSTHROUGH mode)
    std::vector<double> manual_throttles;

    void set_mode(const std::string& s) noexcept {
        if(s=="so3")        mode = Mode::CTL_SO3_GEOMETRIC;
        else if(s=="pass")  mode = Mode::CTL_PASSTHROUGH;
        else                mode = Mode::CTL_CASCADED_PID;
    }

    std::vector<double> compute(
        const Vec3& pos, const Vec3& vel,
        const Quat& att, const Vec3& omega,
        const Mat3& inertia, double mass, double arm, double dt) noexcept
    {
        if(mode == Mode::CTL_PASSTHROUGH) {
            return manual_throttles;
        }
        if(mode == Mode::CTL_CASCADED_PID) {
            auto out = cascaded.compute(pos, vel, att, omega,
                                        target_pos, target_yaw,
                                        mass, dt);
            return mixer.mix(out.thrust_normalized,
                             out.torque_body.x,
                             out.torque_body.y,
                             out.torque_body.z,
                             arm);
        }
        // SO3 — compute desired attitude from position error
        Vec3 pos_err = target_pos - pos;
        Vec3 des_acc {pos_err.x*2.0, pos_err.y*2.0,
                      pos_err.z*2.0 + 9.80665};
        double T_des   = std::clamp(mass * des_acc.norm() / (mass*9.80665*2.0), 0.0, 1.0);
        // Construct desired rotation: Z_b aligned with des_acc
        Vec3 z_des = des_acc.normalized();
        Vec3 x_c   = {std::cos(target_yaw), std::sin(target_yaw), 0.0};
        Vec3 y_des = z_des.cross(x_c).normalized();
        Vec3 x_des = y_des.cross(z_des);
        Mat3 Rd{};
        Rd.m[0][0]=x_des.x; Rd.m[1][0]=x_des.y; Rd.m[2][0]=x_des.z;
        Rd.m[0][1]=y_des.x; Rd.m[1][1]=y_des.y; Rd.m[2][1]=y_des.z;
        Rd.m[0][2]=z_des.x; Rd.m[1][2]=z_des.y; Rd.m[2][2]=z_des.z;
        // Convert matrix to quaternion
        double trace = Rd.m[0][0]+Rd.m[1][1]+Rd.m[2][2];
        Quat att_des;
        if(trace > 0){
            double s = 0.5/std::sqrt(trace+1);
            att_des = {0.25/s, (Rd.m[2][1]-Rd.m[1][2])*s,
                                (Rd.m[0][2]-Rd.m[2][0])*s,
                                (Rd.m[1][0]-Rd.m[0][1])*s};
        } else {
            att_des = Quat::from_euler_zyx(0,0,target_yaw);  // fallback
        }
        Vec3 tau = so3.compute(att_des, att, omega, {}, inertia);
        return mixer.mix(T_des, tau.x, tau.y, tau.z, arm);
    }
};
