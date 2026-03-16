#pragma once
/**
 * atmosphere.hpp
 * ISA 1976 Standard Atmosphere + extensions:
 *   - Troposphere / stratosphere layers
 *   - Sutherland dynamic viscosity
 *   - Power-law & log-law wind shear profiles
 *   - Density altitude, Mach number, Reynolds number helpers
 */

#include "math_types.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// ISA atmospheric state at a given geometric altitude
// ─────────────────────────────────────────────────────────────────────────────
struct AtmoState {
    double T;    // temperature [K]
    double P;    // pressure    [Pa]
    double rho;  // density     [kg/m³]
    double mu;   // dynamic viscosity [Pa·s]
    double nu;   // kinematic viscosity [m²/s]
    double a;    // speed of sound [m/s]
    double alt;  // geometric altitude [m]
};

class AtmosphereISA {
public:
    // Physical constants
    static constexpr double G0   = 9.80665;      // m/s²
    static constexpr double R    = 287.05287;     // J/(kg·K) dry air
    static constexpr double GAMMA= 1.4;           // specific heat ratio
    static constexpr double MU_REF= 1.716e-5;    // Pa·s at T_REF
    static constexpr double T_REF= 273.15;        // K (0 °C)
    static constexpr double S_SUTH= 110.4;        // Sutherland const [K]

    // ISA sea-level values
    static constexpr double T0  = 288.15;         // K
    static constexpr double P0  = 101325.0;       // Pa
    static constexpr double RHO0= 1.225;          // kg/m³

    static AtmoState at(double alt_m) noexcept {
        // Layer: troposphere 0-11000 m
        double T, P;
        if(alt_m <= 11000.0) {
            const double L = 0.0065;              // K/m lapse rate
            T = T0 - L * alt_m;
            P = P0 * std::pow(T / T0, G0 / (R * L));
        } else if(alt_m <= 20000.0) {
            // Tropopause: isothermal
            T = 216.65;
            const double P11 = P0 * std::pow(216.65/T0, G0/(R*0.0065));
            P = P11 * std::exp(-G0*(alt_m - 11000.0)/(R*T));
        } else {
            // Lower stratosphere: L = -0.001 K/m
            const double L2 = -0.001;
            T = 216.65 + L2*(alt_m - 20000.0);
            T = std::max(T, 160.0);
            const double P11 = P0 * std::pow(216.65/T0, G0/(R*0.0065));
            const double P20 = P11 * std::exp(-G0*9000.0/(R*216.65));
            P = P20 * std::pow(T/216.65, -G0/(R*L2));
        }
        double rho = P / (R * T);
        double mu  = MU_REF * std::pow(T/T_REF, 1.5) * (T_REF + S_SUTH) / (T + S_SUTH);
        double nu  = mu / rho;
        double a   = std::sqrt(GAMMA * R * T);
        AtmoState st;
        st.T=T; st.P=P; st.rho=rho; st.mu=mu; st.nu=nu; st.a=a; st.alt=alt_m;
        return st;
    }

    // Reynolds number for a characteristic length L and speed V
    static double reynolds(const AtmoState& atm, double speed, double length) noexcept {
        return atm.rho * speed * length / (atm.mu + 1e-20);
    }

    // Density altitude [m] from true altitude and temperature offset
    static double density_altitude(double geom_alt, double T_offset=0.0) noexcept {
        AtmoState s = at(geom_alt);
        double T_act = s.T + T_offset;
        double rho_act = s.P / (R * T_act);
        // Inverse ISA: rho = rho0 * (1 - L*h/T0)^(g/RL - 1)
        // Solve numerically: h_d = T0/L * (1 - (rho_act/rho0)^(RL/g))... approximate:
        double h_d = (T0/0.0065) * (1.0 - std::pow(rho_act/RHO0, (R*0.0065)/G0));
        return h_d;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Wind Profile Models
// ─────────────────────────────────────────────────────────────────────────────
struct WindProfile {
    // Power-law shear: V(z) = V_ref * (z/z_ref)^alpha
    static Vec3 power_law(const Vec3& V_ref_dir, double V_ref_mag,
                          double z, double z_ref, double alpha) noexcept {
        double ratio = (z > 0.1 && z_ref > 0.1)
                     ? std::pow(std::max(z, 0.1) / z_ref, alpha)
                     : 0.0;
        return V_ref_dir.normalized() * (V_ref_mag * ratio);
    }

    // Log-law shear (atmospheric surface layer): V(z) = (u*/κ) * ln(z/z0)
    // u* = friction velocity, κ = 0.41 (von Kármán), z0 = roughness length
    static Vec3 log_law(const Vec3& wind_dir, double u_star,
                        double z, double z0) noexcept {
        const double kappa = 0.41;
        if(z <= z0 || z0 <= 0) return Vec3{};
        double mag = (u_star / kappa) * std::log(z / z0);
        return wind_dir.normalized() * std::max(0.0, mag);
    }

    static Vec3 ekman(const Vec3& geostrophic_wind,
                      double z, double f_coriolis, double K_eddy) noexcept {
        if(f_coriolis <= 0 || K_eddy <= 0) return geostrophic_wind;
        double De    = std::sqrt(2.0 * K_eddy / f_coriolis);
        double decay = std::exp(-z / De);
        double angle = z / De;
        double c = std::cos(angle), s = std::sin(angle);
        double ug = geostrophic_wind.x, vg = geostrophic_wind.y;
        return Vec3{
            ug - decay*(ug*c - vg*s),
            vg - decay*(vg*c + ug*s),
            0.0
        };
    }
};
