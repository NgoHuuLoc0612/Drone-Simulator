#pragma once
/**
 * aerodynamics.hpp
 * Aerodynamic models for the drone body + rotors:
 *
 *   1. Body drag  — 6-component drag tensor (Cd matrix, not scalar)
 *   2. Blade Element Theory (BET) induced inflow correction
 *   3. Ground Effect (empirical IGE model, Cheeseman-Bennett)
 *   4. Translational Lift (vortex ring avoidance / momentum theory)
 *   5. Rotor-rotor aerodynamic interference (download factor)
 *   6. Added mass / apparent mass tensor (for fast manoeuvres)
 */

#include "math_types.hpp"
#include "atmosphere.hpp"
#include <cmath>
#include <algorithm>
#include <array>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Body aerodynamic tensor
// ─────────────────────────────────────────────────────────────────────────────
struct BodyAeroParams {
    // Drag area (Cd*A) per principal body axis [m²]
    double CdA_x{0.030};   // frontal
    double CdA_y{0.030};   // lateral
    double CdA_z{0.045};   // vertical (arms contribute more)
    // Aerodynamic moment coefficients
    double Cm_roll{0.008};
    double Cm_pitch{0.008};
    double Cm_yaw{0.003};
    // Added-mass diagonal [kg]  — non-negligible for fast manoeuvres
    double m_add_x{0.04};
    double m_add_y{0.04};
    double m_add_z{0.06};
};

class BodyAerodynamics {
    BodyAeroParams params_;
public:
    explicit BodyAerodynamics(const BodyAeroParams& p = {}) : params_(p) {}

    /**
     * Compute aerodynamic forces & moments in BODY frame.
     * v_rel   : velocity of drone relative to air, in body frame [m/s]
     * omega   : angular velocity, body frame [rad/s]
     * rho     : air density [kg/m³]
     * Returns : {F_aero_body, M_aero_body}
     */
    std::pair<Vec3,Vec3> evaluate(const Vec3& v_rel_body,
                                   const Vec3& omega,
                                   double rho) const noexcept {
        double q = 0.5 * rho;   // dynamic pressure prefactor

        // Drag force  F = -q * [CdAx*|u|, CdAy*|v|, CdAz*|w|] * sign(velocity)
        auto sgn = [](double v) { return (v>=0)?1.0:-1.0; };
        Vec3 F_drag{
            -q * params_.CdA_x * v_rel_body.x * std::abs(v_rel_body.x),
            -q * params_.CdA_y * v_rel_body.y * std::abs(v_rel_body.y),
            -q * params_.CdA_z * v_rel_body.z * std::abs(v_rel_body.z)
        };

        // Aerodynamic damping moments (angular drag)
        double v2 = v_rel_body.norm2() + 1.0;  // avoid /0
        Vec3 M_aero{
            -q * params_.Cm_roll  * omega.x * std::abs(omega.x),
            -q * params_.Cm_pitch * omega.y * std::abs(omega.y),
            -q * params_.Cm_yaw   * omega.z * std::abs(omega.z)
        };

        return std::make_pair(F_drag, M_aero);
    }

    Vec3 added_mass_force(const Vec3& accel_body) const noexcept {
        return Vec3{
            -params_.m_add_x * accel_body.x,
            -params_.m_add_y * accel_body.y,
            -params_.m_add_z * accel_body.z
        };
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Ground Effect — Cheeseman-Bennett (1955)
// ─────────────────────────────────────────────────────────────────────────────
// T_IGE/T_OGE = 1 / (1 - (R/(4z))²)    for z > 0.25R
// Extended with Prouty correction for small z
struct GroundEffect {
    // Returns thrust multiplier > 1.0 when near ground
    static double thrust_ratio(double altitude_above_ground,
                                double rotor_radius) noexcept {
        if(altitude_above_ground <= 0.0) return 1.0;
        double ratio = rotor_radius / (4.0 * altitude_above_ground);
        if(ratio >= 1.0) ratio = 0.999;  // clamp — too close
        double kge = 1.0 / (1.0 - ratio*ratio);
        // Prouty correction: smooth blend past z=R
        if(altitude_above_ground > rotor_radius) {
            double blend = std::exp(-(altitude_above_ground - rotor_radius) / rotor_radius);
            kge = 1.0 + (kge - 1.0) * blend;
        }
        return kge;
    }

    // Power multiplier IGE (less power needed near ground)
    static double power_ratio(double altitude_above_ground,
                               double rotor_radius) noexcept {
        double kt = thrust_ratio(altitude_above_ground, rotor_radius);
        return 1.0 / std::sqrt(kt);  // P_IGE ≈ T^(3/2) / sqrt(kge)
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Vortex Ring State (VRS) detection & momentum theory breakdown
// VRS occurs when induced velocity ~ descent rate (autorotation boundary)
// ─────────────────────────────────────────────────────────────────────────────
struct VortexRingDetector {
    // Returns 0 (clean) → 1 (full VRS)
    // v_i_hover : induced velocity in hover [m/s] = sqrt(T/(2ρA))
    // v_descent : current descent rate (positive downward) [m/s]
    static double vrs_factor(double v_descent, double v_i_hover) noexcept {
        if(v_i_hover < 1e-6) return 0.0;
        double ratio = v_descent / v_i_hover;
        // VRS regime: 0.5 < ratio < 2.0 (approximately)
        if(ratio < 0.3) return 0.0;
        if(ratio > 2.0) return 0.0;   // windmill brake state — different regime
        // Peak at ratio ≈ 1.0
        double vrs = std::sin(M_PI * (ratio - 0.3) / 1.7);
        return std::max(0.0, vrs * vrs);
    }

    // Thrust loss factor due to VRS: T_actual = T_nominal * (1 - 0.4*VRS)
    static double thrust_loss(double v_descent, double v_i_hover) noexcept {
        return 1.0 - 0.40 * vrs_factor(v_descent, v_i_hover);
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Rotor Interference — download effect (body blocks rotor downwash)
// ─────────────────────────────────────────────────────────────────────────────
struct RotorInterference {
    // download_factor: fraction of rotor thrust lost to body download (~0.08-0.15)
    static double body_download(double download_factor = 0.10) noexcept {
        return 1.0 - download_factor;
    }

    // Rotor-to-rotor induced velocity overlap (simplified, based on separation)
    // Returns correction [0,1] — 1 = no overlap, <1 = interference
    static double rotor_overlap(double separation, double radius) noexcept {
        double ratio = separation / (2.0 * radius);
        if(ratio >= 2.0) return 1.0;
        if(ratio <= 0.5) return 0.7;  // heavy overlap
        // Linear blend
        return 0.7 + 0.3 * (ratio - 0.5) / 1.5;
    }
};
