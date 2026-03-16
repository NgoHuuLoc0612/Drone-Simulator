#pragma once
/**
 * sensors.hpp
 * Full sensor suite simulation with realistic noise models:
 *   IMU      : accel + gyro with bias, white noise, random walk, temperature drift
 *   GPS      : position + velocity, HDOP, multi-path, outage simulation
 *   Barometer: altitude + pressure, temperature compensation
 *   Magnetometer: 3-axis with hard/soft iron distortion
 *   Optical Flow: body-rate + translational velocity (for indoor nav)
 *   Lidar Altimeter: range to ground with beam divergence noise
 */

#include "math_types.hpp"
#include "atmosphere.hpp"
#include <random>
#include <cmath>
#include <array>

class SensorSuite {
    mutable std::mt19937_64          rng_;
    mutable std::normal_distribution<double> nd_{0.0, 1.0};

    // IMU bias states (random walk)
    mutable Vec3 accel_bias_{0.01, -0.008, 0.015};
    mutable Vec3 gyro_bias_{0.002, -0.001, 0.0015};
    // GPS position bias (slowly drifting)
    mutable Vec3 gps_bias_{0.0,0.0,0.0};
    // Baro offset
    mutable double baro_offset_{0.5};

    // IMU parameters (MPU-6050-class)
    double accel_noise_density_{0.003};   // m/s²/√Hz
    double accel_bias_instab_  {0.0002};  // m/s²/s random walk
    double gyro_noise_density_ {0.0003};  // rad/s/√Hz
    double gyro_bias_instab_   {1e-5};    // rad/s/s random walk
    double sample_rate_        {500.0};   // Hz

    double randn() const { return nd_(rng_); }

public:
    explicit SensorSuite(uint64_t seed = 99) : rng_(seed) {}

    // ── IMU ──────────────────────────────────────────────────────────────────
    struct IMUReading {
        Vec3   accel_body;   // m/s² in body frame (gravity included!)
        Vec3   gyro_body;    // rad/s
        double temperature;  // °C
    };

    IMUReading imu(const Vec3& accel_world, const Vec3& omega_body,
                   const Quat& att, double motor_temp, double dt) const noexcept {
        // Transform true acceleration to body frame, add gravity
        Vec3 g_world{0, 0, -9.80665};
        Vec3 specific_force_world = accel_world - g_world;   // IMU measures specific force
        Vec3 accel_body = att.inv_rotate(specific_force_world);

        double sqrt_dt  = std::sqrt(std::max(dt, 1e-6));
        double sqrt_sr  = std::sqrt(sample_rate_);
        // White noise
        Vec3 an{ randn()*accel_noise_density_*sqrt_sr,
                 randn()*accel_noise_density_*sqrt_sr,
                 randn()*accel_noise_density_*sqrt_sr };
        // Bias random walk
        accel_bias_.x += randn()*accel_bias_instab_*sqrt_dt;
        accel_bias_.y += randn()*accel_bias_instab_*sqrt_dt;
        accel_bias_.z += randn()*accel_bias_instab_*sqrt_dt;

        Vec3 gn{ randn()*gyro_noise_density_*sqrt_sr,
                 randn()*gyro_noise_density_*sqrt_sr,
                 randn()*gyro_noise_density_*sqrt_sr };
        gyro_bias_.x += randn()*gyro_bias_instab_*sqrt_dt;
        gyro_bias_.y += randn()*gyro_bias_instab_*sqrt_dt;
        gyro_bias_.z += randn()*gyro_bias_instab_*sqrt_dt;

        // Temperature-dependent bias drift
        double T_err   = motor_temp - 25.0;
        Vec3   temp_ab = accel_body * (T_err * 5e-5);

        IMUReading r;
        r.accel_body  = accel_body + accel_bias_ + an + temp_ab;
        r.gyro_body   = omega_body + gyro_bias_  + gn;
        r.temperature = 25.0 + T_err * 0.1;
        return r;
    }

    // ── GPS ──────────────────────────────────────────────────────────────────
    struct GPSReading {
        Vec3   position;
        Vec3   velocity;
        double hdop{0};
        double vdop{0};
        bool   fix{false};
        int    n_sats{0};
    };

    GPSReading gps(const Vec3& true_pos, const Vec3& true_vel,
                   double dt, bool in_shadow=false) const noexcept {
        GPSReading r;
        if(in_shadow){ r.hdop=99.0; r.vdop=99.0; return r; }

        // Slowly drifting bias (ionospheric + multipath)
        gps_bias_.x += randn() * 0.02 * std::sqrt(dt);
        gps_bias_.y += randn() * 0.02 * std::sqrt(dt);
        gps_bias_.z += randn() * 0.04 * std::sqrt(dt);

        double hdop   = 1.1 + std::abs(randn()) * 0.3;
        double vdop   = hdop * 1.5;
        double h_sig  = 2.5 * hdop;
        double v_sig  = 3.5 * vdop;

        Vec3 pos_noise{randn()*h_sig, randn()*h_sig, randn()*v_sig};
        Vec3 vel_noise{randn()*0.05,  randn()*0.05,  randn()*0.08};

        r.position = true_pos + gps_bias_ + pos_noise;
        r.velocity = true_vel + vel_noise;
        r.hdop     = hdop;
        r.vdop     = vdop;
        r.fix      = true;
        r.n_sats   = 10 + static_cast<int>(std::abs(randn())*2);
        return r;
    }

    // ── Barometer ────────────────────────────────────────────────────────────
    struct BaroReading {
        double altitude{0};
        double pressure{0};
        double temperature{0};
    };

    BaroReading barometer(double true_alt, double dt) const noexcept {
        auto atm = AtmosphereISA::at(true_alt);
        baro_offset_ += randn() * 0.02 * std::sqrt(dt);
        BaroReading r;
        r.altitude    = true_alt + baro_offset_ + randn()*0.15;
        r.pressure    = atm.P + randn()*1.5;
        r.temperature = atm.T - 273.15 + randn()*0.5;
        return r;
    }

    // ── Magnetometer ─────────────────────────────────────────────────────────
    struct MagReading {
        Vec3 field_body;
    };

    MagReading magnetometer(const Vec3& earth_field_ned,
                             const Quat& att) const noexcept {
        Vec3 hard_iron{2.1, -1.3, 0.8};
        Vec3 sf{1.02, 0.98, 1.01};
        Vec3 field_ned  = earth_field_ned + hard_iron;
        Vec3 field_body = att.inv_rotate(field_ned);
        field_body.x *= sf.x; field_body.y *= sf.y; field_body.z *= sf.z;
        field_body.x += randn()*0.3;
        field_body.y += randn()*0.3;
        field_body.z += randn()*0.3;
        MagReading r; r.field_body = field_body; return r;
    }

    // ── Optical Flow ─────────────────────────────────────────────────────────
    struct Vec2i { int x{0}, y{0}; };

    struct FlowReading {
        Vec2i  flow_pixels;
        double quality{0.0};
    };

    FlowReading optical_flow(const Vec3& vel_body, double altitude,
                              double focal_length_px=320.0) const noexcept {
        FlowReading r;
        if(altitude < 0.3) return r;
        double scale  = focal_length_px / std::max(altitude, 0.1);
        r.flow_pixels.x = static_cast<int>(-vel_body.x * scale + randn()*1.5);
        r.flow_pixels.y = static_cast<int>(-vel_body.y * scale + randn()*1.5);
        r.quality = std::clamp(200.0 * std::exp(-altitude/10.0), 0.0, 255.0);
        return r;
    }

    // ── Lidar Altimeter ───────────────────────────────────────────────────────
    struct LidarReading {
        double range{0};
        bool   valid{false};
    };

    LidarReading lidar_alt(double true_range, double dt) const noexcept {
        LidarReading r;
        if(true_range > 40.0){ r.range=true_range; return r; }
        double noise = randn()*0.02 + std::abs(randn())*0.005*true_range;
        r.range = std::max(0.0, true_range + noise);
        r.valid = true;
        return r;
    }
};
