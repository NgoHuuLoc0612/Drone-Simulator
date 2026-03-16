#pragma once
/**
 * turbulence.hpp  (parallel edition)
 * All 6 turbulence models + SIMD + ThreadPool batch paths.
 *
 * New APIs:
 *   evaluate_batch_parallel() — splits N points across ThreadPool workers
 *   dryden_batch_simd()       — AVX2-accelerated Dryden channel computation
 */

#include "math_types.hpp"
#include "noise.hpp"
#include "atmosphere.hpp"
#include "thread_pool.hpp"
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct TurbulenceParams {
    double intensity{0.5};
    double shear_layer_alt{300.0};
    double delta_u{8.0};
    double geostrophic_wind{10.0};
    double surface_roughness{0.1};
    double coriolis_f{1e-4};
    double K_eddy{5.0};
    enum class Stability{A,B,C,D,E,F} stability{Stability::D};
    std::string model{"composite"};
};

class DrydenModel {
    const NoiseEngine& noise_;
public:
    explicit DrydenModel(const NoiseEngine& n):noise_(n){}
    Vec3 evaluate(double x,double y,double z,double t,
                  double altitude,double airspeed,const TurbulenceParams& p) const noexcept {
        double h=std::max(altitude,10.0);
        double Lu=h/std::pow(0.177+0.000823*h,1.2),Lv=Lu*0.5,Lw=h;
        double sigma_w=0.1*p.intensity*p.geostrophic_wind*std::pow(0.177+0.000823*h,0.4);
        double sigma_u=sigma_w*1.0,sigma_v=sigma_w*0.75;
        double s=(airspeed>0.1)?airspeed*t:t;
        double gu=noise_.fbm(x/Lu+s/Lu,y*0.01,z*0.01,5,2.0,0.5,NoiseEngine::FbmBase::PERLIN)*sigma_u*2.0;
        double gv=noise_.fbm(x/Lv+s/Lv+31.7,y*0.01,z*0.01,5,2.0,0.5,NoiseEngine::FbmBase::PERLIN)*sigma_v*2.0;
        double gw=noise_.fbm(x/Lw+s/Lw+83.2,y*0.01,z*0.01,5,2.0,0.5,NoiseEngine::FbmBase::PERLIN)*sigma_w*2.0;
        return{gu,gv,gw};
    }
};

class VonKarmanModel {
    const NoiseEngine& noise_;
public:
    explicit VonKarmanModel(const NoiseEngine& n):noise_(n){}
    Vec3 evaluate(double x,double y,double z,double t,
                  double altitude,double airspeed,const TurbulenceParams& p) const noexcept {
        double h=std::max(altitude,10.0);
        double L=1750.0*std::pow(h/(h+200.0),0.35);
        double sigma=p.intensity*0.1*p.geostrophic_wind*std::pow(h/300.0,-0.4)*0.5;
        sigma=std::max(sigma,0.01);
        double s=(airspeed>0.1)?airspeed*t/L:t;
        double u=noise_.fbm(x/L+s,y/(L*2),z/(L*2),8,2.0,0.55,NoiseEngine::FbmBase::SIMPLEX)*sigma;
        double v=noise_.fbm(x/L+s+17.3,y/(L*2),z/(L*2),8,2.0,0.55,NoiseEngine::FbmBase::SIMPLEX)*sigma*0.8;
        double w=noise_.fbm(x/L+s+43.1,y/(L*2),z/(L*2),8,2.0,0.55,NoiseEngine::FbmBase::SIMPLEX)*sigma*0.6;
        return{u,v,w};
    }
};

class KelvinHelmholtzModel {
    const NoiseEngine& noise_;
public:
    explicit KelvinHelmholtzModel(const NoiseEngine& n):noise_(n){}
    Vec3 evaluate(double x,double y,double z,double t,const TurbulenceParams& p) const noexcept {
        double dz=z-p.shear_layer_alt;
        double layer_th=std::max(p.delta_u*5.0,30.0);
        double envelope=std::exp(-0.5*(dz/layer_th)*(dz/layer_th));
        if(envelope<1e-4) return Vec3{};
        double lambda=std::max(200.0,p.delta_u*30.0);
        double cx=p.delta_u*0.5,kx=2.0*M_PI/lambda;
        double phase=kx*(x-cx*t);
        double billow_v=std::sin(phase)*p.delta_u*0.10*envelope;
        double billow_w=std::cos(phase)*p.delta_u*0.08*envelope;
        Vec3 curl_vel=noise_.curl(x*0.008,y*0.008,z*0.008+t*0.02,1e-3,5);
        double kh_amp=p.delta_u*0.12*envelope*p.intensity;
        return{curl_vel.x*kh_amp,curl_vel.y*kh_amp+billow_v,curl_vel.z*kh_amp+billow_w};
    }
};

class ThermalConvectionModel {
    const NoiseEngine& noise_;
    static double sigma_w_conv(double h,double zi,double w_star) noexcept {
        if(zi<=0) return 0;
        double z_over=h/zi;
        return w_star*1.25*std::pow(z_over*(1.0-0.8*z_over),1.0/3.0);
    }
public:
    explicit ThermalConvectionModel(const NoiseEngine& n):noise_(n){}
    Vec3 evaluate(double x,double y,double z,double t,const TurbulenceParams& p) const noexcept {
        static const double zi_table[]={2000,1500,1000,600,300,150};
        static const double ws_table[]={2.0,1.5,1.0,0.4,0.1,0.0};
        int cls=static_cast<int>(p.stability);
        double zi=zi_table[cls],w_star=ws_table[cls]*p.intensity;
        if(w_star<0.01) return Vec3{};
        double sw=sigma_w_conv(z,zi,w_star);
        double thermal=noise_.worley(x*0.002+t*0.001,y*0.002,z*0.002,NoiseEngine::WorleyMode::F2_MINUS_F1);
        thermal=std::clamp(thermal,0.0,1.0);
        double updraft=sw*(2.0*thermal-1.0);
        double mu=noise_.ridged_fbm(x*0.003,y*0.003+t*0.005,z*0.001,5,2.0,0.5,1.0)*sw*0.4;
        double mv=noise_.fbm(x*0.003+99.1,y*0.003+t*0.005,z*0.001,5)*sw*0.4;
        return{mu,mv,updraft};
    }
};

class WakeTurbulenceModel {
    const NoiseEngine& noise_;
public:
    explicit WakeTurbulenceModel(const NoiseEngine& n):noise_(n){}
    Vec3 evaluate(double dx,double dy,double dz,double t_shed,
                  double shed_mass,double shed_speed,double vortex_sep=50.0) const noexcept {
        double rho0=1.225;
        double Gamma=(shed_mass*9.80665)/(rho0*vortex_sep*(shed_speed+1e-6));
        double v_descent=Gamma/(2.0*M_PI*vortex_sep);
        double y_L=-vortex_sep*0.5,y_R=vortex_sep*0.5,z_v=-v_descent*t_shed;
        auto bh=[&](double y_pt,double z_pt,double y_v,double z_vort)->Vec3{
            double rc=5.0,r2=(y_pt-y_v)*(y_pt-y_v)+(z_pt-z_vort)*(z_pt-z_vort);
            double mag=Gamma/(2.0*M_PI)*std::sqrt(r2)/(r2+rc*rc);
            double ang=std::atan2(z_pt-z_vort,y_pt-y_v);
            return{0.0,-mag*std::sin(ang),mag*std::cos(ang)};
        };
        Vec3 total=bh(dy,dz,y_L,z_v)+bh(dy,dz,y_R,z_v);
        return total*std::exp(-t_shed/300.0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// TurbulenceEngine
// ─────────────────────────────────────────────────────────────────────────────
class TurbulenceEngine {
    const NoiseEngine&     noise_;
    SIMDNoiseEngine        simd_noise_;
    DrydenModel            dryden_;
    VonKarmanModel         vonkarman_;
    KelvinHelmholtzModel   kh_;
    ThermalConvectionModel thermal_;
    WakeTurbulenceModel    wake_;
    TurbulenceParams       params_;

public:
    explicit TurbulenceEngine(const NoiseEngine& n)
        :noise_(n),simd_noise_(n),dryden_(n),vonkarman_(n),kh_(n),thermal_(n),wake_(n){}

    void set_params(const TurbulenceParams& p) noexcept{params_=p;}
    const TurbulenceParams& params() const noexcept{return params_;}

    // Single-point — thread-safe (const, no shared write)
    Vec3 evaluate(double x,double y,double z,double t,double airspeed) const noexcept {
        const auto& p=params_;
        if(p.model=="dryden")    return dryden_.evaluate(x,y,z,t,z,airspeed,p);
        if(p.model=="vonkarman") return vonkarman_.evaluate(x,y,z,t,z,airspeed,p);
        if(p.model=="kh")        return kh_.evaluate(x,y,z,t,p);

        Vec3 d=dryden_.evaluate(x,y,z,t,z,airspeed,p);
        Vec3 vk=vonkarman_.evaluate(x,y,z,t,z,airspeed,p);
        Vec3 kh=kh_.evaluate(x,y,z,t,p);
        Vec3 th=thermal_.evaluate(x,y,z,t,p);
        double dw=noise_.domain_warp(x*0.003,y*0.003,z*0.003+t*0.01);
        Vec3 warp{dw*p.intensity*1.5,
                  noise_.ridged_fbm(x*0.004,y*0.004+t*0.02,z*0.002,4)*p.intensity,
                  noise_.worley(x*0.006,y*0.006,z*0.006+t*0.01,
                                NoiseEngine::WorleyMode::F2_MINUS_F1)*p.intensity*0.5};
        double h_norm=std::clamp(z/500.0,0.0,1.0);
        return d*0.30+vk*0.30+kh*(0.20*(1.0-h_norm))+th*(0.15*(1.0-h_norm))+warp*0.10;
    }

    // Serial batch (original interface)
    void evaluate_batch(const double* xs,const double* ys,const double* zs,
                        double t,double airspeed,double* out,int N) const noexcept {
        for(int i=0;i<N;++i){
            Vec3 v=evaluate(xs[i],ys[i],zs[i],t,airspeed);
            out[3*i]=v.x;out[3*i+1]=v.y;out[3*i+2]=v.z;
        }
    }

    // ── Parallel batch via ThreadPool ─────────────────────────────────────────
    // Thread-safe: TurbulenceEngine is const, out[] partitioned by chunk (no conflict)
    void evaluate_batch_parallel(
        const double* xs,const double* ys,const double* zs,
        double t,double airspeed,double* out,int N,
        int parallel_threshold=128) const noexcept
    {
        if(N<=0) return;
        if(N<parallel_threshold){
            evaluate_batch(xs,ys,zs,t,airspeed,out,N);
            return;
        }
        auto& pool=global_pool();
        int nt=pool.n_threads();
        int chunk=std::max(32,(N+nt*2-1)/(nt*2));  // 2 chunks per thread

        pool.parallel_for(N,
            [&](int begin,int end){
                for(int i=begin;i<end;++i){
                    Vec3 v=evaluate(xs[i],ys[i],zs[i],t,airspeed);
                    out[3*i]=v.x;out[3*i+1]=v.y;out[3*i+2]=v.z;
                }
            }, chunk, JobPriority::HIGH);
    }

    // ── SIMD-accelerated Dryden batch (AVX2 fBm for particle systems) ────────
    void dryden_batch_simd(const double* xs,const double* ys,const double* zs,
                           double t,double altitude,double airspeed,
                           double* out,int N) const noexcept {
        if(N<=0) return;
        const auto& p=params_;
        double h=std::max(altitude,10.0);
        double Lu=h/std::pow(0.177+0.000823*h,1.2),Lv=Lu*0.5,Lw=h;
        double sigma_w=0.1*p.intensity*p.geostrophic_wind*std::pow(0.177+0.000823*h,0.4);
        double sigma_u=sigma_w,sigma_v=sigma_w*0.75;
        double s=(airspeed>0.1)?airspeed*t:t;

        std::vector<double> xu(N),xv(N),xw(N),yn(N),zn(N);
        for(int i=0;i<N;++i){
            xu[i]=xs[i]/Lu+s/Lu;
            xv[i]=xs[i]/Lv+s/Lv+31.7;
            xw[i]=xs[i]/Lw+s/Lw+83.2;
            yn[i]=ys[i]*0.01; zn[i]=zs[i]*0.01;
        }
        std::vector<double> u(N),v(N),w(N);
        simd_noise_.fbm_batch(xu.data(),yn.data(),zn.data(),u.data(),N,5,2.0,0.5);
        simd_noise_.fbm_batch(xv.data(),yn.data(),zn.data(),v.data(),N,5,2.0,0.5);
        simd_noise_.fbm_batch(xw.data(),yn.data(),zn.data(),w.data(),N,5,2.0,0.5);
        for(int i=0;i<N;++i){
            out[3*i]  =u[i]*sigma_u*2.0;
            out[3*i+1]=v[i]*sigma_v*2.0;
            out[3*i+2]=w[i]*sigma_w*2.0;
        }
    }
};
