#pragma once
/**
 * math_types.hpp
 * Core math primitives: Vec3, Mat3x3 (full inertia tensor), Quat
 * Used across all subsystems — header-only, no dependencies.
 */

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Vec3
// ─────────────────────────────────────────────────────────────────────────────
struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3  operator+(const Vec3& o) const noexcept { return {x+o.x, y+o.y, z+o.z}; }
    Vec3  operator-(const Vec3& o) const noexcept { return {x-o.x, y-o.y, z-o.z}; }
    Vec3  operator*(double s)      const noexcept { return {x*s,   y*s,   z*s};   }
    Vec3  operator/(double s)      const noexcept { return {x/s,   y/s,   z/s};   }
    Vec3  operator-()              const noexcept { return {-x, -y, -z};           }
    Vec3& operator+=(const Vec3& o) noexcept { x+=o.x; y+=o.y; z+=o.z; return *this; }
    Vec3& operator-=(const Vec3& o) noexcept { x-=o.x; y-=o.y; z-=o.z; return *this; }
    Vec3& operator*=(double s)      noexcept { x*=s;   y*=s;   z*=s;   return *this; }
    bool  operator==(const Vec3& o) const noexcept { return x==o.x && y==o.y && z==o.z; }

    double dot(const Vec3& o)   const noexcept { return x*o.x + y*o.y + z*o.z; }
    Vec3   cross(const Vec3& o) const noexcept {
        return {y*o.z - z*o.y,
                z*o.x - x*o.z,
                x*o.y - y*o.x};
    }
    double norm2()      const noexcept { return dot(*this); }
    double norm()       const noexcept { return std::sqrt(norm2()); }
    Vec3   normalized() const noexcept {
        double n = norm();
        return n > 1e-14 ? *this / n : Vec3{};
    }
    std::array<double,3> to_array() const noexcept { return {x, y, z}; }

    static Vec3 from_array(const std::array<double,3>& a) noexcept {
        return {a[0], a[1], a[2]};
    }
};

inline Vec3 operator*(double s, const Vec3& v) noexcept { return v * s; }


// ─────────────────────────────────────────────────────────────────────────────
// Mat3x3  — full 3×3 inertia tensor (supports off-diagonal products of inertia)
// ─────────────────────────────────────────────────────────────────────────────
struct Mat3 {
    // Row-major storage: m[row][col]
    double m[3][3]{};

    Mat3() = default;

    // Construct diagonal inertia tensor (Ixx, Iyy, Izz, off-diag optional)
    static Mat3 diag(double Ixx, double Iyy, double Izz,
                     double Ixy=0, double Ixz=0, double Iyz=0) noexcept {
        Mat3 r{};
        r.m[0][0]=Ixx; r.m[1][1]=Iyy; r.m[2][2]=Izz;
        r.m[0][1]=r.m[1][0]=-Ixy;
        r.m[0][2]=r.m[2][0]=-Ixz;
        r.m[1][2]=r.m[2][1]=-Iyz;
        return r;
    }

    Vec3 mul(const Vec3& v) const noexcept {
        return {
            m[0][0]*v.x + m[0][1]*v.y + m[0][2]*v.z,
            m[1][0]*v.x + m[1][1]*v.y + m[1][2]*v.z,
            m[2][0]*v.x + m[2][1]*v.y + m[2][2]*v.z,
        };
    }

    // Invert 3×3 (Cramer's rule) — used for I⁻¹
    Mat3 inverse() const {
        double a = m[0][0], b = m[0][1], c = m[0][2];
        double d = m[1][0], e = m[1][1], f = m[1][2];
        double g = m[2][0], h = m[2][1], k = m[2][2];

        double det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
        if(std::abs(det) < 1e-20) throw std::runtime_error("Singular inertia tensor");
        double inv = 1.0 / det;
        Mat3 r{};
        r.m[0][0] = (e*k - f*h)*inv; r.m[0][1] = (c*h - b*k)*inv; r.m[0][2] = (b*f - c*e)*inv;
        r.m[1][0] = (f*g - d*k)*inv; r.m[1][1] = (a*k - c*g)*inv; r.m[1][2] = (c*d - a*f)*inv;
        r.m[2][0] = (d*h - e*g)*inv; r.m[2][1] = (b*g - a*h)*inv; r.m[2][2] = (a*e - b*d)*inv;
        return r;
    }

    // Rotate tensor: I_world = R * I_body * R^T
    Mat3 rotate(const Mat3& R) const noexcept {
        // tmp = R * this
        Mat3 tmp{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
            for(int k=0;k<3;++k) tmp.m[i][j] += R.m[i][k] * m[k][j];
        // out = tmp * R^T
        Mat3 out{};
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
            for(int k=0;k<3;++k) out.m[i][j] += tmp.m[i][k] * R.m[j][k];
        return out;
    }
};


// ─────────────────────────────────────────────────────────────────────────────
// Quat  — Hamilton convention, w scalar first
// ─────────────────────────────────────────────────────────────────────────────
struct Quat {
    double w{1.0}, x{0.0}, y{0.0}, z{0.0};

    constexpr Quat() = default;
    constexpr Quat(double w_, double x_, double y_, double z_)
        : w(w_), x(x_), y(y_), z(z_) {}

    Quat operator*(const Quat& q) const noexcept {
        return {
            w*q.w - x*q.x - y*q.y - z*q.z,
            w*q.x + x*q.w + y*q.z - z*q.y,
            w*q.y - x*q.z + y*q.w + z*q.x,
            w*q.z + x*q.y - y*q.x + z*q.w
        };
    }
    Quat conjugate()   const noexcept { return {w, -x, -y, -z}; }
    double norm2()     const noexcept { return w*w + x*x + y*y + z*z; }
    double norm()      const noexcept { return std::sqrt(norm2()); }

    // Normalise (with 1st-order numerical correction to avoid drift)
    Quat normalized()  const noexcept {
        double n2 = norm2();
        // Fast approximate: q *= (3 - |q|²) / 2  (Newton step)
        double s = 0.5 * (3.0 - n2);
        return {w*s, x*s, y*s, z*s};
    }

    // Rotate a vector: v' = q v q*
    Vec3 rotate(const Vec3& v) const noexcept {
        Vec3 qv{x, y, z};
        Vec3 t = qv.cross(v) * 2.0;
        return v + t*w + qv.cross(t);
    }

    // Inverse rotation (conjugate for unit quat)
    Vec3 inv_rotate(const Vec3& v) const noexcept {
        return conjugate().rotate(v);
    }

    // Rotation matrix (row-major)
    Mat3 to_rotation_matrix() const noexcept {
        Mat3 R{};
        double q0=w,q1=x,q2=y,q3=z;
        R.m[0][0]=1-2*(q2*q2+q3*q3); R.m[0][1]=2*(q1*q2-q0*q3); R.m[0][2]=2*(q1*q3+q0*q2);
        R.m[1][0]=2*(q1*q2+q0*q3);   R.m[1][1]=1-2*(q1*q1+q3*q3); R.m[1][2]=2*(q2*q3-q0*q1);
        R.m[2][0]=2*(q1*q3-q0*q2);   R.m[2][1]=2*(q2*q3+q0*q1); R.m[2][2]=1-2*(q1*q1+q2*q2);
        return R;
    }

    // Euler ZYX (yaw-pitch-roll) in radians
    std::array<double,3> to_euler_zyx() const noexcept {
        double sinr_cosp = 2.0*(w*x + y*z);
        double cosr_cosp = 1.0 - 2.0*(x*x + y*y);
        double roll = std::atan2(sinr_cosp, cosr_cosp);

        double sinp = 2.0*(w*y - z*x);
        double pitch = (std::abs(sinp) >= 1.0)
                     ? std::copysign(M_PI*0.5, sinp)
                     : std::asin(sinp);

        double siny_cosp = 2.0*(w*z + x*y);
        double cosy_cosp = 1.0 - 2.0*(y*y + z*z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        return {roll, pitch, yaw};
    }

    std::array<double,4> to_array() const noexcept { return {w, x, y, z}; }

    // Axis-angle constructor
    static Quat from_axis_angle(Vec3 axis, double angle) noexcept {
        double s = std::sin(angle * 0.5);
        Vec3 n = axis.normalized();
        return {std::cos(angle * 0.5), n.x*s, n.y*s, n.z*s};
    }

    // Euler ZYX constructor
    static Quat from_euler_zyx(double roll, double pitch, double yaw) noexcept {
        double cr=std::cos(roll*0.5),  sr=std::sin(roll*0.5);
        double cp=std::cos(pitch*0.5), sp=std::sin(pitch*0.5);
        double cy=std::cos(yaw*0.5),   sy=std::sin(yaw*0.5);
        return {
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        };
    }

    // Scalar multiply  (needed for RK4)
    Quat operator*(double s) const noexcept { return {w*s, x*s, y*s, z*s}; }
    Quat operator+(const Quat& o) const noexcept { return {w+o.w, x+o.x, y+o.y, z+o.z}; }

    // Integration: q_new = q + 0.5 * q ⊗ [0, ω] * dt  (1st order)
    // Uses RK4 for higher accuracy
    Quat integrate_rk4(const Vec3& omega, double dt) const noexcept {
        auto qdot = [](const Quat& q, const Vec3& w) -> Quat {
            // qdot = 0.5 * Quat(0,wx,wy,wz) * q
            return (Quat{0.0, w.x, w.y, w.z} * q) * 0.5;
        };
        Quat k1 = qdot(*this, omega);
        Quat q2 = *this + k1 * (dt * 0.5);
        Quat k2 = qdot(q2, omega);
        Quat q3 = *this + k2 * (dt * 0.5);
        Quat k3 = qdot(q3, omega);
        Quat q4 = *this + k3 * dt;
        Quat k4 = qdot(q4, omega);
        return (*this + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)).normalized();
    }
};
