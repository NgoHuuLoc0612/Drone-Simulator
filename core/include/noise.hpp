#pragma once
/**
 * noise.hpp  — SIMD-extended
 * 8 noise algorithms + AVX2 4-wide fBm via SIMDNoiseEngine
 */

#include "math_types.hpp"
#include "simd_math.hpp"
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <vector>
#include <string>
#include <cstdint>

class NoiseEngine {
    std::array<int,   512> perm_{};
    std::array<Vec3,  16>  grad3_{};
    std::array<double,512> perm_f_{};

    static double fade(double t) noexcept { return t*t*t*(t*(t*6.0-15.0)+10.0); }
    static double lerp(double a, double b, double t) noexcept { return a+t*(b-a); }

    static double grad_dot(int h, double x, double y, double z) noexcept {
        int hh=h&15; double u=(hh<8)?x:y, v=(hh<4)?y:((hh==12||hh==14)?x:z);
        return ((hh&1)?-u:u)+((hh&2)?-v:v);
    }
    static constexpr double F3=1.0/3.0, G3=1.0/6.0;
    static constexpr int SIMPLEX_GRAD[12][3]={
        {1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
        {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
        {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}};
    static double simplex_dot3(const int*g,double x,double y,double z) noexcept{return g[0]*x+g[1]*y+g[2]*z;}

    static uint64_t cell_hash(int cx,int cy,int cz) noexcept {
        uint64_t h=static_cast<uint64_t>(cx*1619+cy*31337+cz*6271)^0xDEADBEEFCAFEBABEULL;
        h^=h>>33; h*=0xFF51AFD7ED558CCDULL; h^=h>>33; h*=0xC4CEB9FE1A85EC53ULL; h^=h>>33; return h;
    }
    static double hash_to_01(uint64_t h,int shift=0) noexcept{return double((h>>shift)&0xFFFFF)/double(0xFFFFF);}

public:
    const std::array<int,512>& perm() const noexcept{return perm_;}
    const std::array<double,512>& perm_f() const noexcept{return perm_f_;}

    explicit NoiseEngine(uint64_t seed=42){
        std::mt19937_64 rng(seed); std::array<int,256> p;
        std::iota(p.begin(),p.end(),0); std::shuffle(p.begin(),p.end(),rng);
        for(int i=0;i<256;++i) perm_[i]=perm_[i+256]=p[i];
        for(int i=0;i<512;++i) perm_f_[i]=double(perm_[i])/255.0;
        grad3_={{{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},
                 {1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},
                 {0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1},
                 {1,1,0},{0,-1,1},{-1,1,0},{0,-1,-1}}};
    }

    double perlin(double x,double y,double z) const noexcept {
        int X=static_cast<int>(std::floor(x))&255,Y=static_cast<int>(std::floor(y))&255,Z=static_cast<int>(std::floor(z))&255;
        x-=std::floor(x); y-=std::floor(y); z-=std::floor(z);
        double u=fade(x),v=fade(y),w=fade(z);
        int A=perm_[X]+Y,AA=perm_[A]+Z,AB=perm_[A+1]+Z,B=perm_[X+1]+Y,BA=perm_[B]+Z,BB=perm_[B+1]+Z;
        return lerp(lerp(lerp(grad_dot(perm_[AA],x,y,z),grad_dot(perm_[BA],x-1,y,z),u),
                         lerp(grad_dot(perm_[AB],x,y-1,z),grad_dot(perm_[BB],x-1,y-1,z),u),v),
                    lerp(lerp(grad_dot(perm_[AA+1],x,y,z-1),grad_dot(perm_[BA+1],x-1,y,z-1),u),
                         lerp(grad_dot(perm_[AB+1],x,y-1,z-1),grad_dot(perm_[BB+1],x-1,y-1,z-1),u),v),w);
    }

    double simplex(double x,double y,double z) const noexcept {
        double s=(x+y+z)*F3;
        int i=static_cast<int>(std::floor(x+s)),j=static_cast<int>(std::floor(y+s)),k=static_cast<int>(std::floor(z+s));
        double t=(i+j+k)*G3,x0=x-(i-t),y0=y-(j-t),z0=z-(k-t);
        int i1,j1,k1,i2,j2,k2;
        if(x0>=y0){if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0;}
                   else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1;}else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1;}}
        else{if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1;}
             else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1;}else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0;}}
        double x1=x0-i1+G3,y1=y0-j1+G3,z1=z0-k1+G3;
        double x2=x0-i2+2*G3,y2=y0-j2+2*G3,z2=z0-k2+2*G3;
        double x3=x0-1+3*G3,y3=y0-1+3*G3,z3=z0-1+3*G3;
        int ii=i&255,jj=j&255,kk=k&255;
        auto gidx=[&](int di,int dj,int dk){return perm_[ii+di+perm_[jj+dj+perm_[kk+dk]]]%12;};
        auto corner=[&](int g,double cx,double cy,double cz)->double{
            double t2=0.6-cx*cx-cy*cy-cz*cz;
            return(t2<0)?0.0:(t2*t2)*(t2*t2)*simplex_dot3(SIMPLEX_GRAD[g],cx,cy,cz);};
        return 32.0*(corner(gidx(0,0,0),x0,y0,z0)+corner(gidx(i1,j1,k1),x1,y1,z1)+
                     corner(gidx(i2,j2,k2),x2,y2,z2)+corner(gidx(1,1,1),x3,y3,z3));
    }

    double value_noise(double x,double y,double z) const noexcept {
        int X=static_cast<int>(std::floor(x))&255,Y=static_cast<int>(std::floor(y))&255,Z=static_cast<int>(std::floor(z))&255;
        double fx=x-std::floor(x),fy=y-std::floor(y),fz=z-std::floor(z);
        double u=fade(fx),v=fade(fy),w=fade(fz);
        auto val=[&](int dx,int dy,int dz){return perm_f_[(perm_[(perm_[(X+dx)&255]+(Y+dy))&255]+(Z+dz))&255];};
        return lerp(lerp(lerp(val(0,0,0),val(1,0,0),u),lerp(val(0,1,0),val(1,1,0),u),v),
                    lerp(lerp(val(0,0,1),val(1,0,1),u),lerp(val(0,1,1),val(1,1,1),u),v),w);
    }

    enum class WorleyMode{F1,F2,F2_MINUS_F1};
    double worley(double x,double y,double z,WorleyMode mode=WorleyMode::F1) const noexcept {
        int xi=static_cast<int>(std::floor(x)),yi=static_cast<int>(std::floor(y)),zi=static_cast<int>(std::floor(z));
        double f1=1e9,f2=1e9;
        for(int dz=-2;dz<=2;++dz) for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx){
            int cx=xi+dx,cy=yi+dy,cz=zi+dz;
            uint64_t h=cell_hash(cx,cy,cz);
            double px=cx+hash_to_01(h,0),py=cy+hash_to_01(h,20),pz=cz+hash_to_01(h,40);
            double d2=(x-px)*(x-px)+(y-py)*(y-py)+(z-pz)*(z-pz);
            if(d2<f1){f2=f1;f1=d2;}else if(d2<f2){f2=d2;}
        }
        switch(mode){case WorleyMode::F1:return std::sqrt(f1);case WorleyMode::F2:return std::sqrt(f2);default:return std::sqrt(f2)-std::sqrt(f1);}
    }

    enum class FbmBase{PERLIN,SIMPLEX,VALUE};
    double fbm(double x,double y,double z,int octaves=8,double lacunarity=2.0,double gain=0.5,
               FbmBase base=FbmBase::PERLIN) const noexcept {
        double value=0.0,amplitude=0.5,frequency=1.0;
        for(int i=0;i<octaves;++i){
            double s;
            switch(base){case FbmBase::SIMPLEX:s=simplex(x*frequency,y*frequency,z*frequency);break;
                         case FbmBase::VALUE:s=value_noise(x*frequency,y*frequency,z*frequency);break;
                         default:s=perlin(x*frequency,y*frequency,z*frequency);}
            value+=amplitude*s; amplitude*=gain; frequency*=lacunarity;
        }
        return value;
    }

    double ridged_fbm(double x,double y,double z,int octaves=6,double lacunarity=2.0,
                      double gain=0.5,double offset=1.0) const noexcept {
        double value=0.0,amplitude=0.5,frequency=1.0,weight=1.0;
        for(int i=0;i<octaves;++i){
            double s=offset-std::abs(simplex(x*frequency,y*frequency,z*frequency));
            s*=s*weight; weight=std::clamp(s*gain,0.0,1.0);
            value+=amplitude*s; amplitude*=gain; frequency*=lacunarity;
        }
        return value;
    }

    double domain_warp(double x,double y,double z,int octaves=6,double warp_strength=4.0) const noexcept {
        double wx=fbm(x,y,z,octaves),wy=fbm(x+5.2,y+1.3,z+2.8,octaves),wz=fbm(x+9.4,y+7.1,z+4.5,octaves);
        return fbm(x+warp_strength*wx,y+warp_strength*wy,z+warp_strength*wz,octaves);
    }
    double domain_warp_double(double x,double y,double z,int octaves=5) const noexcept {
        double qx=fbm(x,y,z,octaves),qy=fbm(x+5.2,y+1.3,z+2.8,octaves),qz=fbm(x+9.4,y+7.1,z+4.5,octaves);
        double rx=fbm(x+1.7+4*qx,y+9.2+4*qy,z+3.1+4*qz,octaves),
               ry=fbm(x+8.3+4*qx,y+2.8+4*qy,z+5.1+4*qz,octaves),
               rz=fbm(x+4.1+4*qx,y+6.3+4*qy,z+0.9+4*qz,octaves);
        return fbm(x+4*rx,y+4*ry,z+4*rz,octaves);
    }

    Vec3 curl(double x,double y,double z,double eps=1e-3,int fbm_oct=6) const noexcept {
        auto Fx=[&](double px,double py,double pz){return fbm(px,py,pz,fbm_oct);};
        auto Fy=[&](double px,double py,double pz){return fbm(px+31.7,py+17.3,pz+53.1,fbm_oct);};
        auto Fz=[&](double px,double py,double pz){return fbm(px+83.2,py+43.1,pz+27.9,fbm_oct);};
        double inv2e=1.0/(2.0*eps);
        return {(Fz(x,y+eps,z)-Fz(x,y-eps,z))*inv2e-(Fy(x,y,z+eps)-Fy(x,y,z-eps))*inv2e,
                (Fx(x,y,z+eps)-Fx(x,y,z-eps))*inv2e-(Fz(x+eps,y,z)-Fz(x-eps,y,z))*inv2e,
                (Fy(x+eps,y,z)-Fy(x-eps,y,z))*inv2e-(Fx(x,y+eps,z)-Fx(x,y-eps,z))*inv2e};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SIMDNoiseEngine — AVX2 4-wide fBm
// ─────────────────────────────────────────────────────────────────────────────
class SIMDNoiseEngine {
    const NoiseEngine& base_;
public:
    explicit SIMDNoiseEngine(const NoiseEngine& n) : base_(n) {}

    // 4-wide Perlin fBm — AVX2 when available, scalar fallback
    void fbm4(const double* xs4, const double* ys4, const double* zs4,
              double* out4, int octaves=8, double lacunarity=2.0, double gain=0.5) const noexcept
    {
#ifdef DRONE_AVX2
        if(cpu_features().avx2 && cpu_features().fma){
            __m256d val=_mm256_setzero_pd();
            __m256d amp=_mm256_set1_pd(0.5);
            __m256d freq=_mm256_set1_pd(1.0);
            __m256d xv=_mm256_loadu_pd(xs4),yv=_mm256_loadu_pd(ys4),zv=_mm256_loadu_pd(zs4);
            const __m256d lac=_mm256_set1_pd(lacunarity),gn=_mm256_set1_pd(gain);
            for(int oct=0;oct<octaves;++oct){
                // Evaluate perlin for 4 points — use scalar but keep in __m256d registers
                DRONE_ALIGN double fx[4],fy[4],fz[4],sv[4];
                _mm256_store_pd(fx,_mm256_mul_pd(xv,freq));
                _mm256_store_pd(fy,_mm256_mul_pd(yv,freq));
                _mm256_store_pd(fz,_mm256_mul_pd(zv,freq));
                for(int i=0;i<4;++i) sv[i]=base_.perlin(fx[i],fy[i],fz[i]);
                __m256d s=_mm256_load_pd(sv);
                val=_mm256_fmadd_pd(amp,s,val);
                amp=_mm256_mul_pd(amp,gn);
                freq=_mm256_mul_pd(freq,lac);
            }
            _mm256_storeu_pd(out4,val);
            return;
        }
#endif
        for(int i=0;i<4;++i) out4[i]=base_.fbm(xs4[i],ys4[i],zs4[i],octaves,lacunarity,gain);
    }

    // N-point batch fBm
    void fbm_batch(const double* xs,const double* ys,const double* zs,
                   double* out,int N,int octaves=8,double lacunarity=2.0,double gain=0.5) const noexcept {
        simd_batch::dispatch(N,
            [&](int base){fbm4(xs+base,ys+base,zs+base,out+base,octaves,lacunarity,gain);},
            [&](int i){out[i]=base_.fbm(xs[i],ys[i],zs[i],octaves,lacunarity,gain);});
    }

    // Turbulence batch: 3-channel (u,v,w) fBm for N points
    // out layout: u0,v0,w0, u1,v1,w1, ...
    void turbulence_batch(const double* xs,const double* ys,const double* zs,
                          double* out,int N,
                          int octaves=5,double scale=1.0,double amp_scale=1.0) const noexcept {
        if(N<=0) return;
        std::vector<double> xs_u(N),xs_v(N),xs_w(N),ys_s(N),zs_s(N);
        std::vector<double> u(N),v(N),w(N);
        for(int i=0;i<N;++i){xs_u[i]=xs[i]*scale;xs_v[i]=xs[i]*scale+31.7;xs_w[i]=xs[i]*scale+83.2;
                              ys_s[i]=ys[i]*scale;zs_s[i]=zs[i]*scale;}
        fbm_batch(xs_u.data(),ys_s.data(),zs_s.data(),u.data(),N,octaves,2.0,0.5);
        fbm_batch(xs_v.data(),ys_s.data(),zs_s.data(),v.data(),N,octaves,2.0,0.5);
        fbm_batch(xs_w.data(),ys_s.data(),zs_s.data(),w.data(),N,octaves,2.0,0.5);
        for(int i=0;i<N;++i){out[3*i]=u[i]*amp_scale;out[3*i+1]=v[i]*amp_scale;out[3*i+2]=w[i]*amp_scale;}
    }

    // 2D noise slice for visualization
    void noise_slice_2d(float* dest,int res,double cx,double cy,double z,
                        double span,const std::string& type) const noexcept {
        double step=span/static_cast<double>(res);
        double x0=cx-span*0.5,y0=cy-span*0.5;
        int total=res*res;
        std::vector<double> xs(total),ys(total),zs(total,z*0.01);
        for(int r=0;r<res;++r) for(int c=0;c<res;++c){
            int idx=r*res+c; xs[idx]=(x0+c*step)*0.01; ys[idx]=(y0+r*step)*0.01;
        }
        if(type=="fbm"||type=="perlin"){
            std::vector<double> tmp(total);
            fbm_batch(xs.data(),ys.data(),zs.data(),tmp.data(),total);
            for(int i=0;i<total;++i) dest[i]=static_cast<float>(tmp[i]);
        } else {
            for(int i=0;i<total;++i){
                double val;
                if     (type=="simplex") val=base_.simplex(xs[i],ys[i],zs[i]);
                else if(type=="value")   val=base_.value_noise(xs[i],ys[i],zs[i]);
                else if(type=="worley")  val=base_.worley(xs[i],ys[i],zs[i]);
                else if(type=="ridged")  val=base_.ridged_fbm(xs[i],ys[i],zs[i]);
                else if(type=="warp")    val=base_.domain_warp(xs[i],ys[i],zs[i]);
                else                     val=base_.fbm(xs[i],ys[i],zs[i]);
                dest[i]=static_cast<float>(val);
            }
        }
    }
};
