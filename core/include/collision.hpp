#pragma once
/**
 * collision.hpp
 * Collision detection & response:
 *   1. TerrainCollider   — bilinear DEM height query, normal estimation
 *   2. SphereCollider    — drone bounding sphere vs terrain
 *   3. ImpulseResponse   — rigid-body impulse at contact point
 *   4. RayCaster         — sphere-march ray vs DEM for line-of-sight / lidar
 *   5. BoundingBox       — AABB of drone vs static obstacles
 */

#include "math_types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
// TerrainCollider  — bilinear interpolated DEM heightmap
// ─────────────────────────────────────────────────────────────────────────────
class TerrainCollider {
    std::vector<float> heights_;
    int    rows_{0}, cols_{0};
    double x_origin_{0.0};
    double y_origin_{0.0};
    double cell_size_{30.0};

    float safe_h(int r, int c) const noexcept {
        r = std::clamp(r, 0, rows_-1);
        c = std::clamp(c, 0, cols_-1);
        return heights_[r*cols_+c];
    }

public:
    TerrainCollider() = default;

    bool loaded() const noexcept { return !heights_.empty(); }

    void load(const float* data, int rows, int cols,
              double x_origin, double y_origin, double cell_size) {
        rows_      = rows;
        cols_      = cols;
        x_origin_  = x_origin;
        y_origin_  = y_origin;
        cell_size_ = cell_size;
        heights_.assign(data, data + rows*cols);
    }

    // Bilinear height at world (x,y)
    double height_at(double x, double y) const noexcept {
        if(heights_.empty()) return 0.0;
        double cf = (x - x_origin_) / cell_size_;
        double rf = (y - y_origin_) / cell_size_;
        int c0 = static_cast<int>(std::floor(cf));
        int r0 = static_cast<int>(std::floor(rf));
        c0 = std::clamp(c0, 0, cols_-2);
        r0 = std::clamp(r0, 0, rows_-2);
        double tx = cf - c0, ty = rf - r0;
        return (1-tx)*(1-ty)*safe_h(r0,c0)
             +    tx *(1-ty)*safe_h(r0,c0+1)
             + (1-tx)*   ty *safe_h(r0+1,c0)
             +    tx *   ty *safe_h(r0+1,c0+1);
    }

    // Surface normal via central differences (for ground friction / bounce)
    Vec3 normal_at(double x, double y) const noexcept {
        double d = cell_size_;
        double hL = height_at(x-d, y),  hR = height_at(x+d, y);
        double hD = height_at(x, y-d),  hU = height_at(x, y+d);
        Vec3 n{hL-hR, hD-hU, 2.0*d};
        return n.normalized();
    }

    double cell_size() const noexcept { return cell_size_; }
};


// ─────────────────────────────────────────────────────────────────────────────
// ImpulseResponse — rigid-body collision impulse (Baumgarte stabilisation)
// ─────────────────────────────────────────────────────────────────────────────
struct ImpulseResponse {
    // Compute velocity correction after ground contact
    // pos     : drone CoM position
    // vel     : drone velocity (world)
    // omega   : drone angular velocity (body)
    // att     : drone attitude quaternion
    // contact_pt : collision point in world frame
    // n       : surface normal (world frame, pointing away from surface)
    // mass    : kg
    // I_body  : inertia tensor (body frame)
    // e       : coefficient of restitution [0=plastic, 1=elastic]
    // mu_f    : coulomb friction coefficient
    struct Result {
        Vec3 delta_vel;    // world frame velocity correction
        Vec3 delta_omega;  // body frame angular velocity correction
    };

    static Result compute(
        const Vec3& pos, const Vec3& vel, const Vec3& omega,
        const Quat& att, const Vec3& contact_pt,
        const Vec3& n, double mass, const Mat3& I_body,
        double e=0.15, double mu_f=0.4) noexcept
    {
        // Relative position: r = contact_pt - CoM
        Vec3 r_world = contact_pt - pos;
        Vec3 r_body  = att.inv_rotate(r_world);

        // Velocity of contact point = v + ω × r
        Vec3 omega_world = att.rotate(omega);
        Vec3 v_contact   = vel + omega_world.cross(r_world);
        double v_n       = v_contact.dot(n);  // normal component

        if(v_n >= 0.0){ Result r{}; return r; }  // separating, no impulse needed

        // Inverse inertia in world frame
        Mat3 I_world = I_body.rotate(att.to_rotation_matrix());
        Mat3 I_inv   = I_world.inverse();

        // Effective mass: m_eff = 1/(1/m + (r×n)·I⁻¹(r×n))
        Vec3 rxn  = r_world.cross(n);
        Vec3 Irxn = I_inv.mul(rxn);
        double inv_meff = 1.0/mass + rxn.dot(Irxn);
        double m_eff = 1.0 / inv_meff;

        // Normal impulse magnitude
        double j_n = -(1.0 + e) * v_n * m_eff;

        // Tangential (friction) impulse
        Vec3 v_t = v_contact - n * v_n;
        double v_t_mag = v_t.norm();
        Vec3   j_impulse = n * j_n;

        if(v_t_mag > 1e-5) {
            Vec3 t_dir = v_t / v_t_mag;
            Vec3 rxt = r_world.cross(t_dir);
            double inv_meff_t = 1.0/mass + rxt.dot(I_inv.mul(rxt));
            double j_t_max    = mu_f * std::abs(j_n);
            double j_t        = std::min(v_t_mag / inv_meff_t, j_t_max);
            j_impulse        += t_dir * (-j_t);
        }

        // Apply impulse
        Vec3 dv   = j_impulse / mass;
        Vec3 dw_w = I_inv.mul(r_world.cross(j_impulse));
        Vec3 dw_b = att.inv_rotate(dw_w);

        Result res;
        res.delta_vel   = dv;
        res.delta_omega = dw_b;
        return res;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// RayCaster — sphere-march against DEM (lidar / terrain avoidance)
// ─────────────────────────────────────────────────────────────────────────────
class RayCaster {
    const TerrainCollider& terrain_;
public:
    explicit RayCaster(const TerrainCollider& t) : terrain_(t) {}

    struct Hit { bool hit; double t; Vec3 point; Vec3 normal; };

    Hit cast(const Vec3& origin, const Vec3& dir,
             double max_dist=10000.0, double step_init=1.0) const noexcept {
        Vec3 d = dir.normalized();
        double t = 0.0;
        for(int i=0;i<4000 && t<max_dist;++i){
            Vec3 p = origin + d*t;
            double g = terrain_.height_at(p.x, p.y);
            double dz = p.z - g;
            if(dz <= 0.05){
                Hit h2; h2.hit=true; h2.t=t; h2.point=p; h2.normal=terrain_.normal_at(p.x,p.y);
                return h2;
            }
            t += std::max(step_init * 0.5, dz * 0.4);
        }
        Hit h2; h2.hit=false; h2.t=max_dist; return h2;
    }

    // Multi-ray lidar scan (azimuth range, elevation range)
    std::vector<Hit> lidar_scan(
        const Vec3& origin, const Quat& att,
        int n_az=36, int n_el=1,
        double az_range=2*M_PI, double el_start=-0.3, double el_step=0.1,
        double max_dist=200.0) const {
        std::vector<Hit> hits;
        hits.reserve(n_az * n_el);
        for(int ie=0;ie<n_el;++ie){
            double el = el_start + ie*el_step;
            for(int ia=0;ia<n_az;++ia){
                double az = ia * az_range / n_az;
                Vec3 d_body{std::cos(el)*std::cos(az),
                            std::cos(el)*std::sin(az),
                            std::sin(el)};
                Vec3 d_world = att.rotate(d_body);
                hits.push_back(cast(origin, d_world, max_dist));
            }
        }
        return hits;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// CollisionSystem — aggregates all detectors for DronePhysics
// ─────────────────────────────────────────────────────────────────────────────
class CollisionSystem {
    TerrainCollider& terrain_;
    RayCaster        ray_;
    double           drone_radius_{0.35};  // m — bounding sphere radius

public:
    explicit CollisionSystem(TerrainCollider& t)
        : terrain_(t), ray_(t) {}

    void set_drone_radius(double r) noexcept { drone_radius_ = r; }
    const TerrainCollider& terrain() const noexcept { return terrain_; }

    /**
     * resolve():
     *   Check drone bounding sphere against terrain.
     *   If penetrating: apply position correction + impulse.
     *   Returns true if collision occurred.
     */
    bool resolve(Vec3& pos, Vec3& vel, Vec3& omega,
                 const Quat& att, double mass,
                 const Mat3& I_body, double dt) noexcept {
        double g   = terrain_.height_at(pos.x, pos.y);
        double alt = pos.z - g;

        if(alt > drone_radius_) return false;

        // Positional correction (Baumgarte)
        Vec3 n     = terrain_.normal_at(pos.x, pos.y);
        double pen = drone_radius_ - alt;
        double beta = 0.3;  // Baumgarte stabilisation factor
        pos       += n * (pen + beta * pen);  // push out of ground

        // Velocity impulse at contact point (bottom of sphere)
        Vec3 contact = pos - n * drone_radius_;
        auto res = ImpulseResponse::compute(
            pos, vel, omega, att, contact, n,
            mass, I_body, 0.10, 0.35);

        vel   += res.delta_vel;
        omega += res.delta_omega;

        // Hard floor: no sinking
        if(vel.dot(n) < 0) vel -= n * vel.dot(n);

        return true;
    }

    // Quick downward ray for HUD altitude display
    double altitude_above_terrain(const Vec3& pos) const noexcept {
        return pos.z - terrain_.height_at(pos.x, pos.y);
    }

    RayCaster& ray_caster() noexcept { return ray_; }
};
