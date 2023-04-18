// version 2015-07-03--16-30-06
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cuComplex.h>
#include <cufft.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <time.h>

#if __cplusplus >= 201103L
#include <thread>
//#include <mutex>
#include <condition_variable>
#include <mutex>
#define CPP11
#endif

// host code annotation for NVVP. Remember to compile with -lnvToolsExt option.
#if false
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#define NVTOOLS
#endif

using namespace std;

#define CUDA_SAFE_CALL(call)                                                   \
  do {                                                                         \
    cudaError err = call;                                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", __FILE__,  \
              __LINE__, cudaGetErrorString(err));                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define Nx 200 // Grid size, must be even!
#define Ny 200
#define Nz 200
#define PI M_PI

#define defaultIndexing                                                        \
  volatile unsigned int index =                                                \
      Nx * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x

const double Lx = 24, Ly = 24,
             Lz = 24; // Lx=83.6, Ly=83.6, Lz=83.6;//Grid dimensions
const double hx = Lx / (Nx - 1), hy = Ly / (Ny - 1), hz = Lz / (Nz - 1);

/*
   PHYSICAL CONSTANTS
*/
const double omega_r = 200 * 2 * PI, omega_z = 200 * 2 * PI; // shifting the
                                                             // trap
const double gn = 15000.0; // interaction constants
const double gs = 150.0;   // coupling constant c_1
const double ga = 1.5;     // coupling constant c_2
const double alpha = 0.0,
             beta = 0.0; // alpha =198.96, beta=0.00295798; //alpha is the
                         // three-body loss and beta loss from the trap
const int N = Nx * Ny * Nz;
const unsigned int
    Niter = 18000, // how many times iterated when finding the ground state
    monitorfreq = 2000, writefreq = 6000;

const double hbar = 1.05457148e-34;
const double muB = 9.27400968e-24; // bohr magnetonh_norm
const double mag_const = 0.5 * muB / (hbar * omega_r);
const double grad_const =
    0.5 * muB / (hbar * omega_r) * sqrt(hbar / ((1.44316082e-25) * omega_r));
const double a_r = sqrt(hbar / ((1.44316082e-25) * omega_r)); // 87Rb
const double q = 0.0 * hbar; // quadratic zeeman term

// const int T_quad_on=38956, T_ramp=64, T_trap=1558, T_wait=11*487;

/*
   SIMULATION PARAMETERS
*/
// const int shift=0;
const bool resume = true;    // load state from file?
const int resumeInitTag = 0; // tag passed to init when resuming.
const int restartInitTag =
    0; // tag passed to init when restarting from ground state.
const int Max_time = 1000;
const int freq = 1000; // writing frequency
int stepnum = 0; // the timestep we start from when resuming, NOTE: assuming
                 // freq hasn't been changed since.
const double time_step = 1e-4; // 2e-4

// const int period = 130900;

// const int open_trap = 2*T_quad_on + T_ramp/2 + shift + T_wait + T_trap;
const int open_trap = Max_time;
const double GP_tol = 1e-12;
const double lambda_x = 1.0;
const double lambda_y = 1.0;
const double lambda_z = omega_z / omega_r; // 1.32258;
const double relaxation_parameter = 0.5;
const double dd0 = -2.0, dd1 = 1.0;

// s
// const double Bp = (0.25e-2)*mag_const*(7.6256340336e-7), Bz = -2*Bp,
//  B0=(5e-8)*mag_const, Bx=1e-5; //magnetic field values (in Teslas)

// l
const double Bp = (0.005) * mag_const * (7.6256340336e-7), Bz = -2 * Bp,
             B0 = (1e-6) * mag_const,
             Bx = 1e-5; // magnetic field values (in Teslas)

// const double Bp = gf*4.017,Bpp=gf*21.53, Bz = -2*Bp, B0 = gf*(112.87), A=0.0,
// dd=0.4, Bx=0.0;//magnetic fields const double Bx_1=gf*188497.87,
// Bx_2=gf*36119.35,Bx_3=Bx_1-Bx_2, B0_z=gf*5643.65, B0_y=gf*129803.92,
// Bz_2=1128.73;

__constant__ double d_gn;
__constant__ double d_gs;
__constant__ double d_ga;
__constant__ double d_q;
__constant__ double d_Lx, d_Ly, d_Lz;
__constant__ double d_hx, d_hy, d_hz;

/*
  VARIABLES
*/

double
    normi; //, norm1, norm2, norm3, norm4, norm5, cm1, cm2, cm3, cmx, cmy, cmz;
double mu = 10.0;
double E_kin, E_pot, E_nl, E_ss, E_th, E_tot, E_mag, L_z;

/*
    HOST
*/

cuDoubleComplex **h_PSIS, *h_P1, *h_P2, *h_P3, *h_P4, *h_P5, *h_apuC1, *h_apuC2,
    *h_apuC3, *h_apuC4, *h_apuC5;
cuDoubleComplex **h_d_PSIS; // pointer on host pointing to device pointers.
double *h_density;
double *h_Mx, *h_My, *h_Mz, *h_Mu, *h_norm, *h_omega, *h_err, *h_x, *h_y, *h_z,
    *h_trap; //, *h_pol_p, *h_pol_q, *h_pol_bx;

#ifdef CPP11
mutex h_lock;
unique_lock<mutex> h_lock_u(h_lock, defer_lock);
condition_variable h_cv_heavy_write;
#endif

/*
    DEVICE, CUDA STUFF
*/

// note: indexing for psi vectors: d_P5 == psi_(-2), d_P4 == psi_(-1), ...
cuDoubleComplex *d_P1, *d_P2, *d_P3, *d_P4, *d_P5, *d_apuC1, *d_apuC2, *d_apuC3,
    *d_apuC4, *d_apuC5, *swap;
cuDoubleComplex *d_A, *d_B;
// cuDoubleComplex *d_Matrix;
double *d_density, *d_spindensity, *d_thetadensity, *d_TK, *d_fr, *d_fs11,
    *d_Mx, *d_My, *d_Mz, *d_x, *d_y, *d_z,
    *d_trap, //*d_pol_p, *d_pol_q, *d_pol_bx,
    *d_Mu, *d_norm, *d_omega, *d_error_fct, *d_err, *d_apu_real, *d_diagonal,
    *d_apu1, *d_apu2, *d_apu3, *d_apu4, *d_apu5;
thrust::device_ptr<double> dev_dens;
thrust::device_ptr<double> dev_apu_real;

const unsigned int block_size = 256;
const unsigned int n_blocks = Nz * Ny;
const unsigned int n_threads = n_blocks * block_size;
cufftHandle CUDA_plan;
cudaEvent_t start, stop;

/*
  INPUT/OUTPUT
*/
char nimi[20];
const char *std_file = "stdspin2exp.out";
const char *dataFolder = "dataexp/";
FILE *print_file;

inline dim3 make_large_grid(const unsigned int num_threads,
                            const unsigned int blocksize,
                            const unsigned int num_blocks) {
  // const unsigned int num_blocks = DIVIDE_INTO(num_threads, blocksize);
  if (num_blocks <= 65535) {
    // fits in a 1D grid
    return dim3(num_blocks);
  } else {
    // 2D grid is required
    const unsigned int side = (unsigned int)ceil(sqrt((double)num_blocks));
    return dim3(side, side);
  }
}

//-------------------------------------------------------------------------------------------------

// Arithmetic operations for cuda double complex numbers.
#include "headers/cuDoubleComplexOps.h"

//-------------------------------------------------------------------------------------------------
__global__ void normalize_psi(cuDoubleComplex *psi1, cuDoubleComplex *psi2,
                              cuDoubleComplex *psi3, cuDoubleComplex *psi4,
                              cuDoubleComplex *psi5, double *normi) {

  defaultIndexing;
  if (threadIdx.x < Nx) {

    psi1[index] = psi1[index] / sqrt(*normi);
    psi2[index] = psi2[index] / sqrt(*normi);
    psi3[index] = psi3[index] / sqrt(*normi);
    psi4[index] = psi4[index] / sqrt(*normi);
    psi5[index] = psi5[index] / sqrt(*normi);
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void print_vortex(cuDoubleComplex *apu1, cuDoubleComplex *apu2,
                             double *xx, double *yy) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    double phi = atan2(yy[index], xx[index]);

    apu1[index] = sqrt(sqr(cuCabs(apu1[index]))) *
                  cuCexp(-2 * phi * make_cuDoubleComplex(0.0, 1.0));
    apu2[index] = sqrt(sqr(cuCabs(apu2[index]))) *
                  cuCexp(-phi * make_cuDoubleComplex(0.0, 1.0));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void compute_Ks(double *apu) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    unsigned int ix, iy, iz;
    int shift_i, shift_j, shift_k;
    double kx, ky, kz;

    ix = threadIdx.x;
    iy = (blockIdx.x + blockIdx.y * gridDim.x) % Ny;
    iz = (blockIdx.x + blockIdx.y * gridDim.x) / Ny;

    shift_i = (ix + Nx / 2) % Nx - Nx / 2;
    kx = 2.0 * PI * shift_i / d_hx / Nx;
    shift_j = (iy + Ny / 2) % Ny - Ny / 2;
    ky = 2.0 * PI * shift_j / d_hy / Ny;
    shift_k = (iz + Nz / 2) % Nz - Nz / 2;
    kz = 2.0 * PI * shift_k / d_hz / Nz;

    apu[index] = 0.5 * (sqr(kx) + sqr(ky) + sqr(kz));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void polar_complex(cuDoubleComplex *psi, double *apu1,
                              double *apu2) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    psi[index] =
        apu1[index] * cuCexp(apu2[index] * make_cuDoubleComplex(0.0, 1.0));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void arg_psi(cuDoubleComplex *psi, double *apu) {
  defaultIndexing;
  if (threadIdx.x < Nx) {
    apu[index] = atan2(cuCimag(psi[index]), cuCreal(psi[index]));
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void abs_psi(cuDoubleComplex *psi, double *apu) {
  defaultIndexing;
  if (threadIdx.x < Nx) {
    apu[index] = cuCabs(psi[index]);
  }
}
//--------------------------------------------------------------------------------------------------

__global__ void re_psi(cuDoubleComplex *psi, double *apu, int tag) {
  int nn;
  if (tag == 1) {
    nn = Nx;
  } else if (tag == 2) {
    nn = Ny;
  } else {
    nn = Nz;
  }

  unsigned int index = nn * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x;
  if (threadIdx.x < nn) {

    apu[index] = cuCreal(psi[index]);
  }
}
//-------------------------------------------------------------------------------------------------
__global__ void imag_psi(cuDoubleComplex *psi, double *apu, int tag) {
  int nn;
  if (tag == 1) {
    nn = Nx;
  } else if (tag == 2) {
    nn = Ny;
  } else {
    nn = Nz;
  }

  unsigned int index = nn * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x;
  if (threadIdx.x < nn) {

    apu[index] = cuCimag(psi[index]);
  }
}

//--------------------------------------------------------------------------------------------------

__global__ void make_complex(cuDoubleComplex *psi, double *apu, int tag1,
                             int tag2) {

  int nn;
  if (tag2 == 1) {
    nn = Nx;
  } else if (tag2 == 2) {
    nn = Ny;
  } else {
    nn = Nz;
  }

  unsigned int index = nn * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x;

  if (threadIdx.x < nn) {
    if (tag1 == 1)
      psi[index].x = apu[index];
    else if (tag1 == 2)
      psi[index].y = apu[index];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void kinetic_term(cuDoubleComplex *out, const cuDoubleComplex *in,
                             const double *TK, const double t) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = cuCexp(make_cuDoubleComplex(0.0, -t) * TK[index]) * in[index];
  }
}

//--------------------------------------------------------------------------------------------------
// in-place kinetic_term
__global__ void kinetic_term(cuDoubleComplex *inout, const double *TK,
                             const double t) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout[index] =
        cuCexp(make_cuDoubleComplex(0.0, -t) * TK[index]) * inout[index];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void divide_by_N(cuDoubleComplex *inout) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout[index] = inout[index] / (Nx * Ny * Nz);
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_diagonal_term(double *out, const cuDoubleComplex *in1,
                                   const cuDoubleComplex *in2,
                                   const cuDoubleComplex *in3,
                                   const cuDoubleComplex *in4,
                                   const cuDoubleComplex *in5,
                                   const double *trap, const double g) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] =
        trap[index] + g * (sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) +
                           sqr(cuCabs(in3[index])) + sqr(cuCabs(in4[index])) +
                           sqr(cuCabs(in5[index])));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_diagonal_term_three_body(
    cuDoubleComplex *__restrict__ out1, cuDoubleComplex *__restrict__ out2,
    cuDoubleComplex *__restrict__ out3, cuDoubleComplex *__restrict__ out4,
    cuDoubleComplex *__restrict__ out5, const cuDoubleComplex *__restrict__ in1,
    const cuDoubleComplex *__restrict__ in2,
    const cuDoubleComplex *__restrict__ in3,
    const cuDoubleComplex *__restrict__ in4,
    const cuDoubleComplex *__restrict__ in5, const double *__restrict__ trap,
    const double g, const double alpha, const double beta,
    double *__restrict__ Mz) {

  defaultIndexing;

  if (threadIdx.x < Nx) {
    volatile double Fz =
        d_gs * (2.0 * sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) -
                sqr(cuCabs(in4[index])) - 2.0 * sqr(cuCabs(in5[index])));

    const volatile double a =
        trap[index] + g * (sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) +
                           sqr(cuCabs(in3[index])) + sqr(cuCabs(in4[index])) +
                           sqr(cuCabs(in5[index])));

    const volatile double b =
        -beta - alpha * (sqr(sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) +
                             sqr(cuCabs(in3[index])) + sqr(cuCabs(in4[index])) +
                             sqr(cuCabs(in5[index]))));

    out1[index] = make_cuDoubleComplex(
        a + 2.0 * Fz + 0.4 * d_ga * sqr(cuCabs(in5[index])) - 2.0 * Mz[index],
        b);

    out2[index] = make_cuDoubleComplex(
        a + Fz + 0.4 * d_ga * sqr(cuCabs(in4[index])) - Mz[index], b);

    out3[index] =
        make_cuDoubleComplex(a + 0.2 * d_ga * sqr(cuCabs(in3[index])), b);

    out4[index] = make_cuDoubleComplex(
        a - Fz + 0.4 * d_ga * sqr(cuCabs(in2[index])) + Mz[index], b);

    out5[index] = make_cuDoubleComplex(
        a - 2.0 * Fz + 0.4 * d_ga * sqr(cuCabs(in1[index])) + 2.0 * Mz[index],
        b);
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_density(double *out, const cuDoubleComplex *in1,
                             const cuDoubleComplex *in2,
                             const cuDoubleComplex *in3,
                             const cuDoubleComplex *in4,
                             const cuDoubleComplex *in5) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) +
                 sqr(cuCabs(in3[index])) + sqr(cuCabs(in4[index])) +
                 sqr(cuCabs(in5[index]));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_component_density(double *out, const cuDoubleComplex *in) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = sqr(cuCabs(in[index]));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_comp_cm_density(double *out, double *koord,
                                     const cuDoubleComplex *in) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = koord[index] * sqr(cuCabs(in[index]));
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void form_cm_density(double *out, double *koord,
                                const cuDoubleComplex *in1,
                                const cuDoubleComplex *in2,
                                const cuDoubleComplex *in3) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] =
        koord[index] * (sqr(cuCabs(in1[index])) + sqr(cuCabs(in2[index])) +
                        sqr(cuCabs(in3[index])));
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void set_to_zero(double *trap) {
  defaultIndexing;
  if (threadIdx.x < Nx) {
    trap[index] = 100.0;
  }
}

//---------------------------------------------------------------------------------------

__global__ void
update_magneticfields(double *out1, double *out2, double *out3,
                      // double *pol_p, double *pol_q, double *pol_bx,
                      double *xx, double *yy, double *zz, int t) {

  if (threadIdx.x < Nx) {
    defaultIndexing;
    /*
        out1[index] = Bp * xx[index];
        out2[index] = Bp * yy[index];
        out3[index] = -2*Bp * zz[index] + B0;
    */
    /*
        out1[index] = 0.0;
        out2[index] = 0.0;
        out3[index] = 0.0;
    */
    if (t < Max_time / 2) {
      out1[index] = Bp * xx[index] * (t / (0.5 * Max_time));
      out2[index] = Bp * yy[index] * (t / (0.5 * Max_time));
      out3[index] = -2 * Bp * zz[index] * (t / (0.5 * Max_time)) + B0;
    } else {
      out1[index] = Bp * xx[index];
      out2[index] = Bp * yy[index];
      out3[index] =
          -2 * Bp * zz[index] + (Max_time - t) / (0.5 * Max_time) * B0;
    }
  }
}

//---------------------------------------------------------------------------------------

__global__ void compute_Lz(cuDoubleComplex *psi1, cuDoubleComplex *psi2,
                           cuDoubleComplex *psi3, cuDoubleComplex *psi4,
                           cuDoubleComplex *psi5, double *xx, double *yy,
                           double *Lzz) {

  if (threadIdx.x < Nx) {
    defaultIndexing;
    unsigned int indexL = (blockIdx.x + blockIdx.y * gridDim.x + 1) * (Nx) +
                          threadIdx.x; // left point
    unsigned int indexR = (blockIdx.x + blockIdx.y * gridDim.x - 1) * (Nx) +
                          threadIdx.x; // right point

    if (index != 0 && index != (Nx)-1 &&
        (blockIdx.x + blockIdx.y * gridDim.x + 1) % Ny != 0 &&
        (blockIdx.x + blockIdx.y * gridDim.x) % Ny != 0) {
      //(blockIdx.x + blockIdx.y * gridDim.x) > (Ny-1) && (blockIdx.x +
      //blockIdx.y * gridDim.x) < ((Nz-1)*Ny)){

      Lzz[index] = cuCreal(
          cuConj(psi1[index]) * make_cuDoubleComplex(0.0, -1.0) *
              (xx[index] * (psi1[indexL] - psi1[indexR]) / (2 * d_hy) -
               yy[index] * (psi1[index + 1] - psi1[index - 1]) / (2 * d_hx)) +
          cuConj(psi2[index]) * make_cuDoubleComplex(0.0, -1.0) *
              (xx[index] * (psi2[indexL] - psi2[indexR]) / (2 * d_hy) -
               yy[index] * (psi2[index + 1] - psi2[index - 1]) / (2 * d_hx)) +
          cuConj(psi3[index]) * make_cuDoubleComplex(0.0, -1.0) *
              (xx[index] * (psi3[indexL] - psi3[indexR]) / (2 * d_hy) -
               yy[index] * (psi3[index + 1] - psi3[index - 1]) / (2 * d_hx)) +
          cuConj(psi4[index]) * make_cuDoubleComplex(0.0, -1.0) *
              (xx[index] * (psi4[indexL] - psi4[indexR]) / (2 * d_hy) -
               yy[index] * (psi4[index + 1] - psi4[index - 1]) / (2 * d_hx)) +
          cuConj(psi5[index]) * make_cuDoubleComplex(0.0, -1.0) *
              (xx[index] * (psi5[indexL] - psi5[indexR]) / (2 * d_hy) -
               yy[index] * (psi5[index + 1] - psi5[index - 1]) / (2 * d_hx)));

    } else {
      Lzz[index] = 0.0;
    }
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_magnetic(double *out, const cuDoubleComplex *in1,
                              const cuDoubleComplex *in2,
                              const cuDoubleComplex *in3,
                              const cuDoubleComplex *in4,
                              const cuDoubleComplex *in5, double *Mmx,
                              double *Mmy, double *Mmz) {

  defaultIndexing;
  if (threadIdx.x < Nx) {
    out[index] = cuCreal(
        0.5 * make_cuDoubleComplex(0, 1) *
            (Mmy[index] * (2.0 * in2[index] * cuConj(in1[index]) +
                           (-2.0 * in1[index] + sqrt(6.0) * in3[index]) *
                               cuConj(in2[index]) +
                           sqrt(6.0) * (-1.0 * in2[index] + in4[index]) *
                               cuConj(in3[index]) +
                           (-1.0 * sqrt(6.0) * in3[index] + 2.0 * in5[index]) *
                               cuConj(in4[index]) -
                           2 * in4[index] * cuConj(in5[index])))

        + 0.5 * cuCreal(Mmx[index] *
                        (2.0 * in2[index] * cuConj(in1[index]) +
                         (2.0 * in1[index] + sqrt(6.0) * in3[index]) *
                             cuConj(in2[index]) +
                         sqrt(6.0) * (in2[index] + in4[index]) *
                             cuConj(in3[index]) +
                         (sqrt(6.0) * in3[index] + 2.0 * in5[index]) *
                             cuConj(in4[index]) +
                         2 * in4[index] * cuConj(in5[index])))

        + cuCreal(Mmz[index] * (2.0 * in5[index] * cuConj(in5[index]) +
                                in4[index] * cuConj(in4[index]) -
                                in2[index] * cuConj(in2[index]) -
                                2.0 * in1[index] * cuConj(in1[index]))));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_spindensity(double *out, const cuDoubleComplex *in1,
                                 const cuDoubleComplex *in2,
                                 const cuDoubleComplex *in3,
                                 const cuDoubleComplex *in4,
                                 const cuDoubleComplex *in5) {

  defaultIndexing;
  double mag_x, mag_y, mag_z;

  if (threadIdx.x < Nx) {
    mag_x = cuCreal(in1[index] * cuConj(in2[index]) +
                    in4[index] * cuConj(in5[index]) +
                    sqrt(1.5) * (in2[index] * cuConj(in3[index]) +
                                 in3[index] * cuConj(in4[index])));
    mag_y = cuCimag(in1[index] * cuConj(in2[index]) +
                    in4[index] * cuConj(in5[index]) +
                    sqrt(1.5) * (in2[index] * cuConj(in3[index]) +
                                 in3[index] * cuConj(in4[index])));
    mag_z = cuCreal(2.0 * cuConj(in1[index]) * in1[index] +
                    cuConj(in2[index]) * in2[index] -
                    cuConj(in3[index]) * in3[index] -
                    2.0 * cuConj(in5[index]) * in5[index]);
    out[index] = sqrt(sqr(abs(mag_x)) + sqr(abs(mag_y)) + sqr(abs(mag_z)));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void form_thetadensity(double *out, const cuDoubleComplex *in1,
                                  const cuDoubleComplex *in2,
                                  const cuDoubleComplex *in3,
                                  cuDoubleComplex *in4, cuDoubleComplex *in5) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] =
        cuCabs(1.0 / sqrt(5.0) *
               (2.0 * in1[index] * in5[index] - 2.0 * in2[index] * in4[index] +
                in3[index] * in3[index]));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void
form_fr_fs(double *fr, double *fs11, cuDoubleComplex *fs12,
           cuDoubleComplex *fsP1, cuDoubleComplex *fsP2, cuDoubleComplex *fsP3,
           cuDoubleComplex *fsP4, cuDoubleComplex *fsP5,
           const cuDoubleComplex *in1, const cuDoubleComplex *in2,
           const cuDoubleComplex *in3, const cuDoubleComplex *in4,
           const cuDoubleComplex *in5, double *Mmx, double *Mmy, double *Mmz) {

  defaultIndexing;
  double mag_x, mag_y, mag_z;

  if (threadIdx.x < Nx) {
    mag_x = d_gs * cuCreal(in1[index] * cuConj(in2[index]) +
                           in4[index] * cuConj(in5[index]) +
                           sqrt(1.5) * (in2[index] * cuConj(in3[index]) +
                                        in3[index] * cuConj(in4[index]))) -
            Mmx[index];
    mag_y = d_gs * cuCimag(in1[index] * cuConj(in2[index]) +
                           in4[index] * cuConj(in5[index]) +
                           sqrt(1.5) * (in2[index] * cuConj(in3[index]) +
                                        in3[index] * cuConj(in4[index]))) -
            Mmy[index];
    mag_z = d_gs * cuCreal(2.0 * cuConj(in1[index]) * in1[index] +
                           cuConj(in2[index]) * in2[index] -
                           cuConj(in3[index]) * in3[index] -
                           2.0 * cuConj(in5[index]) * in5[index]) -
            Mmz[index];
    fr[index] = sqrt(sqr(abs(mag_x)) + sqr(abs(mag_y)) + sqr(abs(mag_z)));
    if (fr[index] > 1e-15) {
      mag_x = mag_x / fr[index];
      mag_y = mag_y / fr[index];
      mag_z = mag_z / fr[index];
    } else {
      mag_x = 0.0;
      mag_y = 0.0;
      mag_z = 0.0;
    }
    fs11[index] = mag_z;
    fs12[index] = make_cuDoubleComplex(mag_x, -mag_y);

    fsP1[index] = 2.0 * fs11[index] * in1[index] + fs12[index] * in2[index];
    fsP2[index] = fs11[index] * in2[index] + cuConj(fs12[index]) * in1[index] +
                  sqrt(1.5) * fs12[index] * in3[index];
    fsP3[index] = sqrt(1.5) *
                  (cuConj(fs12[index]) * in2[index] + fs12[index] * in4[index]);
    fsP4[index] = -1.0 * fs11[index] * in4[index] + fs12[index] * in5[index] +
                  sqrt(1.5) * cuConj(fs12[index]) * in3[index];
    fsP5[index] =
        -2.0 * fs11[index] * in1[index] + cuConj(fs12[index]) * in2[index];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void diagonal_term(cuDoubleComplex *inout, const double *in,
                              const double t) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout[index] =
        cuCexp(make_cuDoubleComplex(0.0, -t) * in[index]) * inout[index];
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void diagonal_term_three_body(cuDoubleComplex *inout,
                                         const cuDoubleComplex *in,
                                         const double t) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout[index] =
        cuCexp(make_cuDoubleComplex(0.0, -t) * in[index]) * inout[index];
  }
}

//-------------------------------------------------------------------------------------------------

#include "headers/off_diagonal.h"
/*
__global__ void off_diagonal_term(cuDoubleComplex *in1,
                                  cuDoubleComplex *in2,
                                  cuDoubleComplex *in3,
                                  cuDoubleComplex *in4,
                                  cuDoubleComplex *in5,
                                  double *Mx, double *My, const double t) {

    ...

    (included in header file)

*/

//-------------------------------------------------------------------------------------------------

__global__ void spin_term(cuDoubleComplex *inout1, cuDoubleComplex *inout2,
                          cuDoubleComplex *inout3, cuDoubleComplex *inout4,
                          cuDoubleComplex *inout5, const double *fr,
                          const double *fs11, const cuDoubleComplex *fs12,
                          const cuDoubleComplex *fsP1,
                          const cuDoubleComplex *fsP2,
                          const cuDoubleComplex *fsP3,
                          const cuDoubleComplex *fsP4,
                          const cuDoubleComplex *fsP5, const double t) {

  // SPIN-1 !!!
  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout1[index] =
        inout1[index] -
        make_cuDoubleComplex(0.0, 1.0) * sin(t * fr[index]) * fsP1[index] +
        (cos(t * fr[index]) - 1.0) *
            (2.0 * fs11[index] * fsP1[index] + fs12[index] * fsP2[index]);
    inout2[index] =
        inout2[index] -
        make_cuDoubleComplex(0.0, 1.0) * sin(t * fr[index]) * fsP2[index] +
        (cos(t * fr[index]) - 1.0) *
            (fs11[index] * fsP2[index] + cuConj(fs12[index]) * fsP1[index] +
             sqrt(1.5) * fs12[index] * fsP3[index]);
    inout3[index] =
        inout3[index] -
        make_cuDoubleComplex(0.0, 1.0) * sin(t * fr[index]) * fsP3[index] +
        (cos(t * fr[index]) - 1.0) * sqrt(1.5) *
            (cuConj(fs12[index]) * fsP2[index] + fs12[index] * fsP4[index]);
    inout4[index] =
        inout4[index] -
        make_cuDoubleComplex(0.0, 1.0) * sin(t * fr[index]) * fsP4[index] +
        (cos(t * fr[index]) - 1.0) *
            (-1.0 * fs11[index] * fsP4[index] +
             sqrt(1.5) * cuConj(fs12[index]) * fsP3[index] +
             fs12[index] * fsP5[index]);
    inout5[index] =
        inout5[index] -
        make_cuDoubleComplex(0.0, 1.0) * sin(t * fr[index]) * fsP5[index] +
        (cos(t * fr[index]) - 1.0) * (-2.0 * fs11[index] * fsP5[index] +
                                      cuConj(fs12[index]) * fsP4[index]);
  }
}
/*
//--------------------------------------------------------------------------------------------------
__global__ void theta_term(cuDoubleComplex *inout1, cuDoubleComplex *inout2,
cuDoubleComplex *inout3, cuDoubleComplex *inout4, cuDoubleComplex *inout5,
cuDoubleComplex *in1, cuDoubleComplex *in2, cuDoubleComplex *in3,
cuDoubleComplex *in4, cuDoubleComplex *in5,  const double t) {

  defaultIndexing;

  cuDoubleComplex pa1 = make_cuDoubleComplex(0.0,0.0);
  cuDoubleComplex pa2 = make_cuDoubleComplex(0.0,0.0);
  cuDoubleComplex pa3 = make_cuDoubleComplex(0.0,0.0);
  cuDoubleComplex pa4 = make_cuDoubleComplex(0.0,0.0);
  cuDoubleComplex pa5 = make_cuDoubleComplex(0.0,0.0);

  cuDoubleComplex Theta = d_ga/sqrt(5.0)*(2.0*inout1[index]*inout5[index]
- 2.0*inout2[index]*inout4[index] + inout3[index]*inout3[index]); double
fsquared = pow(cuCabs(inout1[index]),2) + pow(cuCabs(inout2[index]),2) +
pow(cuCabs(inout3[index]),2) + pow(cuCabs(inout4[index]),2)
                    + pow(cuCabs(inout5[index]),2);
  if (cuCabs(inout1[index])>1e-50)
    cuDoubleComplex pa1
= 1.0/inout1[index]*sqrt(1.0-(fsquared-pow(cuCabs(inout1[index]),2) )); if
(cuCabs(inout2[index])>1e-50) cuDoubleComplex pa2
= 1.0/inout2[index]*sqrt(-1.0+(fsquared-pow(cuCabs(inout2[index]),2) )); if
(cuCabs(inout3[index])>1e-50) cuDoubleComplex pa3
= 1.0/inout3[index]*sqrt(1.0-(fsquared-pow(cuCabs(inout3[index]),2) )); if
(cuCabs(inout4[index])>1e-50) cuDoubleComplex pa4
= 1.0/inout4[index]*sqrt(-1.0+(fsquared-pow(cuCabs(inout4[index]),2) )); if
(cuCabs(inout5[index])>1e-50) cuDoubleComplex pa5
= 1.0/inout5[index]*sqrt(1.0-(fsquared-pow(cuCabs(inout5[index]),2) ));

  if (threadIdx.x<Nx) {
    if (cuCabs(pa1)>1e-50)
        in1[index] = cuCcos(Theta*pa1*pa5*t)*inout1[index] -
make_cuDoubleComplex(0.0,1.0)*pa5*cuCsin(Theta*pa1*pa5*t)/pa1*inout5[index];
    else
        in1[index] = make_cuDoubleComplex(0.0,-1.0)*pa5*Theta*inout5[index];
    if (cuCabs(pa2)>1e-50)
        in2[index] = cuCcos(Theta*pa2*pa4*t)*inout2[index] -
make_cuDoubleComplex(0.0,1.0)*pa4*cuCsin(Theta*pa2*pa4*t)/pa2*inout4[index];
    else
        in2[index] = make_cuDoubleComplex(0.0,-1.0)*pa4*Theta*t*inout4[index];
    in3[index] =
cuCexp(make_cuDoubleComplex(0.0,-1.0)*Theta*pa3*pa3*t)*inout3[index]; if
(cuCabs(pa4)>1e-50) in4[index] = cuCcos(Theta*pa2*pa4*t)*inout4[index] -
make_cuDoubleComplex(0.0,1.0)*pa2*cuCsin(Theta*pa2*pa4*t)/pa4*inout2[index];
    else
        in4[index] = make_cuDoubleComplex(0.0,-1.0)*pa2*Theta*t*inout2[index];
    if (cuCabs(pa5)>1e-50)
        in5[index] = cuCcos(Theta*pa1*pa5*t)*inout5[index] -
make_cuDoubleComplex(0.0,1.0)*pa1*cuCsin(Theta*pa1*pa5*t)/pa5*inout1[index];
    else
        in5[index] = make_cuDoubleComplex(0.0,-1.0)*pa1*Theta*t*inout1[index];


    inout1[index] = in1[index];
    inout2[index] = in2[index];
    inout3[index] = in3[index];
    inout4[index] = in4[index];
    inout5[index] = in5[index];
  }
}
*/
//--------------------------------------------------------------------------------------------------

__global__ void multiply_CR_C(cuDoubleComplex *inout, const double *in) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    inout[index] = inout[index] * in[index];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void multiply_kinetic(double *out, const cuDoubleComplex *in1,
                                 const cuDoubleComplex *in2) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = cuCreal(cuConj(in1[index]) * in2[index]);
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void multiply_RR_R(double *out, const double *in1,
                              const cuDoubleComplex *in2) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = in1[index] * cuCreal(in2[index]);
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void multiply_RR_R(double *out, const double *in1,
                              const double *in2) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = in1[index] * in2[index];
  }
}
//--------------------------------------------------------------------------------------------------
__global__ void multiply_nonlinear(double *out, const double *in,
                                   const double g) {
  defaultIndexing;

  if (threadIdx.x < Nx) {
    out[index] = g / 2.0 * sqr(abs(in[index]));
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void integrate_z(double *out, const double *Integrand) {

  const int ind = threadIdx.x;
  int avitus;
  __shared__ double integ[block_size];

  integ[ind] = 0.0;

  if (ind < Nz) {
    integ[ind] = Integrand[(blockIdx.x + blockIdx.y * gridDim.x) * Nz + ind];
  }
  __syncthreads();

  avitus = block_size / 2;
  while (avitus > 0) {
    if (ind < avitus) {
      integ[ind] = integ[ind] + integ[ind + avitus];
    }
    __syncthreads();
    avitus = avitus / 2;
  }

  if (ind == 0) {
    out[blockIdx.x + blockIdx.y * gridDim.x] = integ[0];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void integrate_y(double *out, const double *Integrand) {

  const int ind = threadIdx.x;
  int avitus;
  __shared__ double integ[block_size];

  integ[ind] = 0.0;

  if (ind < Ny) {
    integ[ind] = Integrand[(blockIdx.x + blockIdx.y * gridDim.x) * Ny + ind];
  }
  __syncthreads();

  avitus = block_size / 2;
  while (avitus > 0) {
    if (ind < avitus) {
      integ[ind] = integ[ind] + integ[ind + avitus];
    }
    __syncthreads();
    avitus = avitus / 2;
  }

  if (ind == 0) {
    out[blockIdx.x + blockIdx.y * gridDim.x] = integ[0];
  }
}

//--------------------------------------------------------------------------------------------------
__global__ void integrate_x(double *out, const double *Integrand) {

  const int ind = threadIdx.x;
  int avitus;
  __shared__ double integ[block_size];

  integ[ind] = 0.0;

  if (ind < Nx) {
    integ[ind] = Integrand[ind];
  }
  __syncthreads();

  avitus = block_size / 2;
  while (avitus > 0) {
    if (ind < avitus) {
      integ[ind] = integ[ind] + integ[ind + avitus];
    }
    __syncthreads();
    avitus = avitus / 2;
  }

  if (ind == 0) {
    out[0] = integ[0] * d_hx * d_hy * d_hz;
  }
}

//---------------------------------------------------------------------------------------------------
__global__ void spin_rotation_x1(cuDoubleComplex *apu1, cuDoubleComplex *apu2,
                                 cuDoubleComplex *apu3, double theta) {
  defaultIndexing;
  if (threadIdx.x < Nx) {
    cuDoubleComplex P1, P2, P3;

    P1 = 0.5 * (1 + cos(theta)) * apu1[index] -
         make_cuDoubleComplex(0.0, 1.0) * (sin(theta) / sqrt(2.0)) *
             apu2[index] +
         0.5 * (-1 + cos(theta)) * apu3[index];

    P2 =
        cos(theta) * apu2[index] -
        make_cuDoubleComplex(0.0, 1.0) * (sin(theta) / sqrt(2.0)) *
            apu1[index] -
        make_cuDoubleComplex(0.0, 1.0) * (sin(theta) / sqrt(2.0)) * apu3[index];

    P3 = 0.5 * (-1 + cos(theta)) * apu1[index] -
         make_cuDoubleComplex(0.0, 1.0) * (sin(theta) / sqrt(2.0)) *
             apu2[index] +
         0.5 * (1 + cos(theta)) * apu3[index];

    apu1[index] = P1;
    apu2[index] = P2;
    apu3[index] = P3;
  }
}
//----------------------------------------------------------------------------------------------------
__global__ void spin_rotation_x2(cuDoubleComplex *apu1, cuDoubleComplex *apu2,
                                 double theta) {

  defaultIndexing;

  if (threadIdx.x < Nx) {

    cuDoubleComplex P1, P2;
    P1 = cos(theta / 2.0) * apu1[index] -
         make_cuDoubleComplex(0.0, 1.0) * sin(theta / 2.0) * apu2[index];
    P2 = cos(theta / 2.0) * apu2[index] -
         make_cuDoubleComplex(0.0, 1.0) * sin(theta / 2.0) * apu1[index];
    apu1[index] = P1;
    apu2[index] = P2;
  }
}
//---------------------------------------------------------------------------------------------------

__global__ void jacobi(cuDoubleComplex *psi1, cuDoubleComplex *psi2,
                       cuDoubleComplex *psi3, cuDoubleComplex *psi4,
                       cuDoubleComplex *psi5, double *Mx, double *My,
                       double *Mz, double *Potentiaali, double *myy,
                       double *omega_h, cuDoubleComplex *apu1,
                       cuDoubleComplex *apu2, cuDoubleComplex *apu3,
                       cuDoubleComplex *apu4, cuDoubleComplex *apu5, int mo) {

  if (threadIdx.x >= Nx) {
    return;
  }
  volatile const int ind = threadIdx.x;
  volatile const int index =
      (blockIdx.x + blockIdx.y * gridDim.x) * (Nx) + threadIdx.x;
  volatile const int indexL = (blockIdx.x + blockIdx.y * gridDim.x + 1) * (Nx) +
                              threadIdx.x; // index of the point on the left
  volatile const int indexR =
      (blockIdx.x + blockIdx.y * gridDim.x - 1) * (Nx) + threadIdx.x;
  const int layer = Nx * Ny;

  __shared__ cuDoubleComplex fii1[Nx];
  __shared__ cuDoubleComplex fii2[Nx];
  __shared__ cuDoubleComplex fii3[Nx];
  __shared__ cuDoubleComplex fii4[Nx];
  __shared__ cuDoubleComplex fii5[Nx];

  __shared__ double Mmx[Nx];
  __shared__ double Mmy[Nx];
  __shared__ double Mmz[Nx];
  __shared__ double pot[Nx];
  __shared__ double mu;
  __shared__ double omega1;
  __shared__ double hxyz;

  double fiisquared;
  cuDoubleComplex foo1 = make_cuDoubleComplex(0.0, 0.0);
  double foo2 = 0.0;
  double A = 0.0;
  double B = 0.0;
  double C = 0.0;
  cuDoubleComplex Theta = make_cuDoubleComplex(0.0, 0.0);

  if (ind == 0) {
    mu = *myy;
    hxyz = 2.0 / (d_hx * d_hx) + 2.0 / (d_hy * d_hy) + 2.0 / (d_hz * d_hz);
    omega1 = *omega_h;
  }
  pot[ind] = Potentiaali[index];

  Mmx[ind] = Mx[index];
  Mmy[ind] = My[index];
  Mmz[ind] = Mz[index];

  fii1[ind] = psi1[index];
  fii2[ind] = psi2[index];
  fii3[ind] = psi3[index];
  fii4[ind] = psi4[index];
  fii5[ind] = psi5[index];

  __syncthreads();

  fiisquared =
      cuCreal(cuConj(fii1[ind]) * fii1[ind] + cuConj(fii2[ind]) * fii2[ind] +
              cuConj(fii3[ind]) * fii3[ind] + cuConj(fii4[ind]) * fii4[ind] +
              cuConj(fii5[ind]) * fii5[ind]);

  foo1 = fii1[ind] * cuConj(fii2[ind]) + fii4[ind] * cuConj(fii5[ind]) +
         sqrt(1.5) *
             (fii2[ind] * cuConj(fii3[ind]) + fii3[ind] * cuConj(fii4[ind]));

  A = cuCreal(
      foo1); // fii1[ind]*cuConj(fii2[ind]) + fii4[ind]*cuConj(fii5[ind]) +
             // sqrt(1.5)*(fii2[ind]*cuConj(fii3[ind]) +
             // fii3[ind]*cuConj(fii4[ind])));
  B = cuCimag(
      foo1); // fii1[ind]*cuConj(fii2[ind]) + fii4[ind]*cuConj(fii5[ind]) +
             // sqrt(1.5)*(fii2[ind]*cuConj(fii3[ind]) +
             // fii3[ind]*cuConj(fii4[ind])));
  C = 2.0 * sqr(cuCabs(fii1[ind])) + sqr(cuCabs(fii2[ind])) -
      sqr(cuCabs(fii4[ind])) - 2.0 * sqr(cuCabs(fii5[ind]));

  Theta = 1.0 / (sqrt(5.0)) *
          (2.0 * fii1[ind] * fii5[ind] - 2.0 * fii2[ind] * fii4[ind] +
           fii3[ind] * fii3[ind]);

  if (ind != 0 && ind != (Nx)-1 &&
      (blockIdx.x + blockIdx.y * gridDim.x + 1) % Ny != 0 &&
      (blockIdx.x + blockIdx.y * gridDim.x) % Ny != 0 &&
      (blockIdx.x + blockIdx.y * gridDim.x) > (Ny - 1) &&
      (blockIdx.x + blockIdx.y * gridDim.x) < ((Nz - 1) * Ny)) {

    foo1 = -0.5 * (-hxyz * fii1[ind] +
                   (fii1[ind + 1] + fii1[ind - 1]) / sqr(d_hx) +
                   (psi1[indexL] + psi1[indexR]) / sqr(d_hy) +
                   (psi1[index + layer] + psi1[index - layer]) / sqr(d_hz)) +
           (pot[ind] + d_gn * fiisquared - mu) * fii1[ind] +
           d_gs * ((A - make_cuDoubleComplex(0.0, 1.0) * B) * fii2[ind] +
                   2.0 * C * fii1[ind]) -
           (make_cuDoubleComplex(Mmx[ind], 0.0) -
            make_cuDoubleComplex(0.0, Mmy[ind])) *
               fii2[ind] -
           2.0 * Mmz[ind] * fii1[ind] +
           d_ga / sqrt(5.0) * Theta * cuConj(fii5[ind]);

    foo2 = 0.5 * hxyz + pot[ind] + d_gn * fiisquared + 2.0 * d_gs * C +
           0.4 * d_ga * sqr(cuCabs(fii5[ind])); // - mu - 2.0*Mmz[ind];
    apu1[index] = fii1[ind] - omega1 * foo1 / foo2;

    foo1 = -0.5 * (-hxyz * fii2[ind] +
                   (fii2[ind + 1] + fii2[ind - 1]) / sqr(d_hx) +
                   (psi2[indexL] + psi2[indexR]) / sqr(d_hy) +
                   (psi2[index + layer] + psi2[index - layer]) / sqr(d_hz)) +
           (pot[ind] + d_gn * fiisquared - mu) * fii2[ind] +
           d_gs * ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii1[ind] +
                   ((A - make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                       fii3[ind] +
                   C * fii2[ind]) -
           (make_cuDoubleComplex(Mmx[ind], 0.0) +
            make_cuDoubleComplex(0.0, Mmy[ind])) *
               fii1[ind] -
           ((make_cuDoubleComplex(Mmx[ind], 0.0) -
             make_cuDoubleComplex(0.0, Mmy[ind])) *
            sqrt(1.5)) *
               fii3[ind] -
           Mmz[ind] * fii2[ind] - d_ga / sqrt(5.0) * Theta * cuConj(fii4[ind]);

    foo2 = 0.5 * hxyz + pot[ind] + d_gn * fiisquared + d_gs * C +
           0.4 * d_ga * sqr(cuCabs(fii4[ind])); // - mu - Mmz[ind];
    apu2[index] = fii2[ind] - omega1 * foo1 / foo2;

    foo1 = -0.5 * (-hxyz * fii3[ind] +
                   (fii3[ind + 1] + fii3[ind - 1]) / sqr(d_hx) +
                   (psi3[indexL] + psi3[indexR]) / sqr(d_hy) +
                   (psi3[index + layer] + psi3[index - layer]) / sqr(d_hz)) +
           (pot[ind] + d_gn * fiisquared - mu) * fii3[ind] +
           +d_gs * (((A + make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                        fii2[ind] +
                    ((A - make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                        fii4[ind]) -
           ((make_cuDoubleComplex(Mmx[ind], 0.0) +
             make_cuDoubleComplex(0.0, Mmy[ind])) *
            sqrt(1.5)) *
               fii2[ind] -
           ((make_cuDoubleComplex(Mmx[ind], 0.0) -
             make_cuDoubleComplex(0.0, Mmy[ind])) *
            sqrt(1.5)) *
               fii4[ind] +
           d_ga / sqrt(5.0) * Theta * cuConj(fii3[ind]);

    foo2 = 0.5 * hxyz + pot[ind] + d_gn * fiisquared +
           0.2 * d_ga * sqr(cuCabs(fii3[ind])); // - mu;
    apu3[index] = fii3[ind] - omega1 * foo1 / foo2;

    foo1 = -0.5 * (-hxyz * fii4[ind] +
                   (fii4[ind + 1] + fii4[ind - 1]) / sqr(d_hx) +
                   (psi4[indexL] + psi4[indexR]) / sqr(d_hy) +
                   (psi4[index + layer] + psi4[index - layer]) / sqr(d_hz)) +
           (pot[ind] + d_gn * fiisquared - mu) * fii4[ind] +
           +d_gs * ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii3[ind] *
                        sqrt(1.5) +
                    (A - make_cuDoubleComplex(0.0, 1.0) * B) * fii5[ind] -
                    C * fii4[ind]) -
           ((make_cuDoubleComplex(Mmx[ind], 0.0) +
             make_cuDoubleComplex(0.0, Mmy[ind])) *
            sqrt(1.5)) *
               fii3[ind] -
           (make_cuDoubleComplex(Mmx[ind], 0.0) -
            make_cuDoubleComplex(0.0, Mmy[ind])) *
               fii5[ind] +
           Mmz[ind] * fii4[ind] - d_ga / sqrt(5.0) * Theta * cuConj(fii2[ind]);

    foo2 = 0.5 * hxyz + pot[ind] + d_gn * fiisquared - d_gs * C +
           0.4 * d_ga * sqr(cuCabs(fii2[ind])); // - mu + Mmz[ind];
    apu4[index] = fii4[ind] - omega1 * foo1 / foo2;

    foo1 = -0.5 * (-hxyz * fii5[ind] +
                   (fii5[ind + 1] + fii5[ind - 1]) / sqr(d_hx) +
                   (psi5[indexL] + psi5[indexR]) / sqr(d_hy) +
                   (psi5[index + layer] + psi5[index - layer]) / sqr(d_hz)) +
           (pot[ind] + d_gn * fiisquared - mu) * fii5[ind] +
           +d_gs * ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii4[ind] -
                    2.0 * C * fii5[ind]) -
           (make_cuDoubleComplex(Mmx[ind], 0.0) +
            make_cuDoubleComplex(0.0, Mmy[ind])) *
               fii4[ind] +
           2.0 * Mmz[ind] * fii5[ind] +
           d_ga / sqrt(5.0) * Theta * cuConj(fii1[ind]);

    foo2 = 0.5 * hxyz + pot[ind] + d_gn * fiisquared - 2.0 * d_gs * C +
           0.4 * d_ga * sqr(cuCabs(fii1[ind])); // - mu + 2.0*Mmz[ind];
    apu5[index] = fii5[ind] - omega1 * foo1 / foo2;

  } else {

    apu1[index] = make_cuDoubleComplex(0.0, 0.0);
    apu2[index] = make_cuDoubleComplex(0.0, 0.0);
    apu3[index] = make_cuDoubleComplex(0.0, 0.0);
    apu4[index] = make_cuDoubleComplex(0.0, 0.0);
    apu5[index] = make_cuDoubleComplex(0.0, 0.0);
  }
}

//-----------------------------------------------------------------------------------------------------------------
__global__ void psiHpsi(cuDoubleComplex *psi1, cuDoubleComplex *psi2,
                        cuDoubleComplex *psi3, cuDoubleComplex *psi4,
                        cuDoubleComplex *psi5, double *Mx, double *My,
                        double *Mz, double *Potentiaali, double *myy,
                        double *psiHoperpsi, int psi_H_psi) {

  if (threadIdx.x >= (Nx)) {
    return;
  }
  psiHoperpsi[(blockIdx.x + blockIdx.y * gridDim.x) * (Nx) + threadIdx.x] = 0.0;
  // const int idx = large_grid_thread_id();
  const volatile int ind = threadIdx.x;
  const volatile int index =
      (blockIdx.x + blockIdx.y * gridDim.x) * (Nx) + threadIdx.x;
  const volatile int indexL = (blockIdx.x + blockIdx.y * gridDim.x + 1) * (Nx) +
                              threadIdx.x; // left point
  const volatile int indexR = (blockIdx.x + blockIdx.y * gridDim.x - 1) * (Nx) +
                              threadIdx.x; // right point
  const volatile int layer = Ny * Nx;

  __shared__ cuDoubleComplex fii1[Nx];
  __shared__ cuDoubleComplex fii2[Nx];
  __shared__ cuDoubleComplex fii3[Nx];
  __shared__ cuDoubleComplex fii4[Nx];
  __shared__ cuDoubleComplex fii5[Nx];
  __shared__ double Mmx[Nx];
  __shared__ double Mmy[Nx];
  __shared__ double Mmz[Nx];
  __shared__ double pot[Nx];
  __shared__ double mu;
  __shared__ double hxyz;
  double fiisquared;
  double A = 0.0;
  double B = 0.0;
  double C = 0.0;
  cuDoubleComplex Theta = make_cuDoubleComplex(0.0, 0.0);

  cuDoubleComplex Hoperpsi1 = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex Hoperpsi2 = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex Hoperpsi3 = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex Hoperpsi4 = make_cuDoubleComplex(0.0, 0.0);
  cuDoubleComplex Hoperpsi5 = make_cuDoubleComplex(0.0, 0.0);

  if (ind == 0) {
    mu = (*myy);
    hxyz = 2.0 / (d_hx * d_hx) + 2.0 / (d_hy * d_hy) + 2.0 / (d_hz * d_hz);
  }

  pot[ind] =
      Potentiaali[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x];

  Mmx[ind] = Mx[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x];
  Mmy[ind] = My[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x];
  Mmz[ind] = Mz[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x];

  fii1[ind] = psi1[index];
  fii2[ind] = psi2[index];
  fii3[ind] = psi3[index];
  fii4[ind] = psi4[index];
  fii5[ind] = psi5[index];

  __syncthreads();

  fiisquared =
      cuCreal(cuConj(fii1[ind]) * fii1[ind] + cuConj(fii2[ind]) * fii2[ind] +
              cuConj(fii3[ind]) * fii3[ind] + cuConj(fii4[ind]) * fii4[ind] +
              cuConj(fii5[ind]) * fii5[ind]);

  A = cuCreal(fii1[ind] * cuConj(fii2[ind]) + fii4[ind] * cuConj(fii5[ind]) +
              sqrt(1.5) * (fii2[ind] * cuConj(fii3[ind]) +
                           fii3[ind] * cuConj(fii4[ind])));
  B = cuCimag(fii1[ind] * cuConj(fii2[ind]) + fii4[ind] * cuConj(fii5[ind]) +
              sqrt(1.5) * (fii2[ind] * cuConj(fii3[ind]) +
                           fii3[ind] * cuConj(fii4[ind])));
  C = 2.0 * sqr(cuCabs(fii1[ind])) + sqr(cuCabs(fii2[ind])) -
      sqr(cuCabs(fii4[ind])) - 2.0 * sqr(cuCabs(fii5[ind]));

  Theta = 1.0 / (sqrt(5.0)) *
          (2.0 * fii1[ind] * fii5[ind] - 2.0 * fii2[ind] * fii4[ind] +
           fii3[ind] * fii3[ind]);

  if (ind != 0 && ind != (Nx)-1 &&
      (blockIdx.x + blockIdx.y * gridDim.x + 1) % Ny != 0 &&
      (blockIdx.x + blockIdx.y * gridDim.x) % Ny != 0 &&
      (blockIdx.x + blockIdx.y * gridDim.x) > (Ny - 1) &&
      (blockIdx.x + blockIdx.y * gridDim.x) < ((Nz - 1) * Ny)) {

    Hoperpsi1 =
        -0.5 *
            (-hxyz * fii1[ind] + (fii1[ind + 1] + fii1[ind - 1]) / sqr(d_hx) +
             (psi1[indexL] + psi1[indexR]) / sqr(d_hy) +
             (psi1[index + layer] + psi1[index - layer]) / sqr(d_hz)) +
        (pot[ind] + d_gn * fiisquared) * fii1[ind] +
        d_gs * ((A - make_cuDoubleComplex(0.0, 1.0) * B) * fii2[ind] +
                2.0 * C * fii1[ind]) -
        (make_cuDoubleComplex(Mmx[ind], 0.0) -
         make_cuDoubleComplex(0.0, Mmy[ind])) *
            fii2[ind] -
        2.0 * Mmz[ind] * fii1[ind] +
        d_ga / sqrt(5.0) * Theta * cuConj(fii5[ind]);

    Hoperpsi2 =
        -0.5 *
            (-hxyz * fii2[ind] + (fii2[ind + 1] + fii2[ind - 1]) / sqr(d_hx) +
             (psi2[indexL] + psi2[indexR]) / sqr(d_hy) +
             (psi2[index + layer] + psi2[index - layer]) / sqr(d_hz)) +
        (pot[ind] + d_gn * fiisquared) * fii2[ind] +
        d_gs * ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii1[ind] +
                ((A - make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                    fii3[ind] +
                C * fii2[ind]) -
        (make_cuDoubleComplex(Mmx[ind], 0.0) +
         make_cuDoubleComplex(0.0, Mmy[ind])) *
            fii1[ind] -
        ((make_cuDoubleComplex(Mmx[ind], 0.0) -
          make_cuDoubleComplex(0.0, Mmy[ind])) *
         sqrt(1.5)) *
            fii3[ind] -
        Mmz[ind] * fii2[ind] - d_ga / sqrt(5.0) * Theta * cuConj(fii4[ind]);

    Hoperpsi3 =
        -0.5 *
            (-hxyz * fii3[ind] + (fii3[ind + 1] + fii3[ind - 1]) / sqr(d_hx) +
             (psi3[indexL] + psi3[indexR]) / sqr(d_hy) +
             (psi3[index + layer] + psi3[index - layer]) / sqr(d_hz)) +
        (pot[ind] + d_gn * fiisquared) * fii3[ind] +
        d_gs * (((A + make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                    fii2[ind] +
                ((A - make_cuDoubleComplex(0.0, 1.0) * B) * sqrt(1.5)) *
                    fii4[ind]) -
        ((make_cuDoubleComplex(Mmx[ind], 0.0) +
          make_cuDoubleComplex(0.0, Mmy[ind])) *
         sqrt(1.5)) *
            fii2[ind] -
        ((make_cuDoubleComplex(Mmx[ind], 0.0) -
          make_cuDoubleComplex(0.0, Mmy[ind])) *
         sqrt(1.5)) *
            fii4[ind] +
        d_ga / sqrt(5.0) * Theta * cuConj(fii3[ind]);

    Hoperpsi4 =
        -0.5 *
            (-hxyz * fii4[ind] + (fii4[ind + 1] + fii4[ind - 1]) / sqr(d_hx) +
             (psi4[indexL] + psi4[indexR]) / sqr(d_hy) +
             (psi4[index + layer] + psi4[index - layer]) / sqr(d_hz)) +
        (pot[ind] + d_gn * fiisquared) * fii4[ind] +
        d_gs *
            ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii3[ind] * sqrt(1.5) +
             (A - make_cuDoubleComplex(0.0, 1.0) * B) * fii5[ind] -
             C * fii4[ind]) -
        ((make_cuDoubleComplex(Mmx[ind], 0.0) +
          make_cuDoubleComplex(0.0, Mmy[ind])) *
         sqrt(1.5)) *
            fii3[ind] -
        (make_cuDoubleComplex(Mmx[ind], 0.0) -
         make_cuDoubleComplex(0.0, Mmy[ind])) *
            fii5[ind] +
        Mmz[ind] * fii4[ind] - d_ga / sqrt(5.0) * Theta * cuConj(fii2[ind]);

    Hoperpsi5 =
        -0.5 *
            (-hxyz * fii5[ind] + (fii5[ind + 1] + fii5[ind - 1]) / sqr(d_hx) +
             (psi5[indexL] + psi5[indexR]) / sqr(d_hy) +
             (psi5[index + layer] + psi5[index - layer]) / sqr(d_hz)) +
        (pot[ind] + d_gn * fiisquared) * fii5[ind] +
        d_gs * ((A + make_cuDoubleComplex(0.0, 1.0) * B) * fii4[ind] -
                2.0 * C * fii5[ind]) -
        (make_cuDoubleComplex(Mmx[ind], 0.0) +
         make_cuDoubleComplex(0.0, Mmy[ind])) *
            fii4[ind] +
        2.0 * Mmz[ind] * fii5[ind] +
        d_ga / sqrt(5.0) * Theta * cuConj(fii1[ind]);
  }

  if (psi_H_psi == 0) {
    psiHoperpsi[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x] =
        cuCreal(cuConj(fii1[ind]) * Hoperpsi1 + cuConj(fii2[ind]) * Hoperpsi2 +
                cuConj(fii3[ind]) * Hoperpsi3 + cuConj(fii4[ind]) * Hoperpsi4 +
                cuConj(fii5[ind]) * Hoperpsi5);
  } else {

    psiHoperpsi[(blockIdx.x + blockIdx.y * gridDim.x) * Nx + threadIdx.x] =
        sqr(cuCabs(Hoperpsi1 - mu * fii1[ind])) +
        sqr(cuCabs(Hoperpsi2 - mu * fii2[ind])) +
        sqr(cuCabs(Hoperpsi3 - mu * fii3[ind])) +
        sqr(cuCabs(Hoperpsi4 - mu * fii4[ind])) +
        sqr(cuCabs(Hoperpsi5 - mu * fii5[ind]));
  }
}
//-----------------------------------------------------------------------------------------------------------------
void cudaSyncAndErrorCheck() {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaThreadSynchronize();
}

//-------------------------------------------------------------------------------------------------------------
double riemann_sum(double *array) {
  double result;
  int index;
  for (int i = 0; i < Nx; ++i) {
    for (int j = 0; j < Ny; ++j) {
      for (int k = 0; k < Nz; ++k) {
        index = i + Nx * (j + Ny * k);
        result += array[index];
      }
    }
  }
  result = result * hx * hy * hz;
  return result;
}

//-------------------------------------------------------------------------------------------------------------
void calculate_Ks() {
  double *h_TK = new double[N];
  int shift_i, shift_j, shift_k;
  double kx, ky, kz, kr2;
  int index;

  for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
      for (int k = 0; k < Nz; k++) {
        index = i + Nx * (j + Ny * k);

        shift_i = (i + Nx / 2) % Nx - Nx / 2;
        kx = 2.0 * PI * shift_i / hx / Nx;
        shift_j = (j + Ny / 2) % Ny - Ny / 2;
        ky = 2.0 * PI * shift_j / hy / Ny;
        shift_k = (k + Nz / 2) % Nz - Nz / 2;
        kz = 2.0 * PI * shift_k / hz / Nz;
        kr2 = sqr(kx) + sqr(ky) + sqr(kz);
        h_TK[index] = 0.5 * kr2;
      }
    }
  }

  CUDA_SAFE_CALL(
      cudaMemcpy(d_TK, h_TK, N * sizeof(double), cudaMemcpyHostToDevice));
  delete[] h_TK;
}

//-------------------------------------------------------------------------------------------------------------
double random_number() {
  double d = rand() / (RAND_MAX + 1.0);

  return d;
}

//-------------------------------------------------------------------------------------------------------------
void allocate_memory() {
  size_t mem_avail, mem_total, mem_used;

  h_norm = new double[1];
  h_omega = new double[1];
  h_Mu = new double[1];
  h_Mx = new double[N];
  h_My = new double[N];
  h_x = new double[N];
  h_y = new double[N];
  h_z = new double[N];
  h_Mz = new double[N];
  h_trap = new double[N];
  h_density = new double[N];
  h_P1 = new cuDoubleComplex[N];
  h_P2 = new cuDoubleComplex[N];
  h_P3 = new cuDoubleComplex[N];
  h_P4 = new cuDoubleComplex[N];
  h_P5 = new cuDoubleComplex[N];
  h_apuC1 = new cuDoubleComplex[N];
  h_apuC2 = new cuDoubleComplex[N];
  h_apuC3 = new cuDoubleComplex[N];
  h_apuC4 = new cuDoubleComplex[N];
  h_apuC5 = new cuDoubleComplex[N];
  h_err = new double[1];
  /*  h_pol_q=new double[9];
    h_pol_p=new double[9];
    h_pol_bx=new double[9];
  */

  h_PSIS = new cuDoubleComplex *[5];
  h_PSIS[0] = h_P1;
  h_PSIS[1] = h_P2;
  h_PSIS[2] = h_P3;
  h_PSIS[3] = h_P4;
  h_PSIS[4] = h_P5;

  cufftPlan3d(&CUDA_plan, Nz, Ny, Nx, CUFFT_Z2Z);
  CUDA_SAFE_CALL(cudaMemGetInfo(&mem_avail, &mem_total));
  mem_used = mem_total - mem_avail;
  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Total amount of Device memory: %zu \n", mem_total);
  fprintf(print_file, "Used Device memory after FFT:  %zu \n", mem_used);
  fclose(print_file);

  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_Lx, &Lx, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_Ly, &Ly, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_Lz, &Lz, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_hx, &hx, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_hy, &hy, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_hz, &hz, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_gn, &gn, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_gs, &gs, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_ga, &ga, sizeof(double), 0, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpyToSymbol(d_q, &q, sizeof(double), 0, cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu_real, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_err, sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Mu, sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_omega, sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_norm, sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_trap, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_density, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_spindensity, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_thetadensity, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_diagonal, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_TK, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_fr, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Mx, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_My, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Mz, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_y, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_z, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu1, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu2, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu3, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu4, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apu5, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_fs11, sizeof(double) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apuC1, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apuC2, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apuC3, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apuC4, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_apuC5, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P1, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P2, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P3, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P4, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_P5, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A, sizeof(cuDoubleComplex) * N));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_B, sizeof(cuDoubleComplex) * N));
  //  CUDA_SAFE_CALL(cudaMalloc((void**)&d_Matrix, sizeof(cuDoubleComplex)*25));
  /*  CUDA_SAFE_CALL(cudaMalloc((void**)&d_pol_q, sizeof(double)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pol_p, sizeof(double)*9));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_pol_bx, sizeof(double)*9));
  */

  h_d_PSIS = new cuDoubleComplex *[5];
  h_d_PSIS[0] = d_P1;
  h_d_PSIS[1] = d_P2;
  h_d_PSIS[2] = d_P3;
  h_d_PSIS[3] = d_P4;
  h_d_PSIS[4] = d_P5;

  dev_dens = thrust::device_pointer_cast(d_density);
  dev_apu_real = thrust::device_pointer_cast(d_apu_real);

  CUDA_SAFE_CALL(cudaMemGetInfo(&mem_avail, &mem_total));
  mem_used = mem_total - mem_avail;
  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Available Device memory:       %zu \n", mem_avail);
  fprintf(print_file, "Used Device memory:            %zu \n\n", mem_used);
  fclose(print_file);
}

//-------------------------------------------------------------------------------------------------------------

void load_psi(int num) {

#ifdef NVTOOLS
  nvtxRangeId_t loadPsiRange = nvtxRangeStartA("load_psi()"); // nvvp annotation
#endif

  char name[50];
  double r1, r2;
  ifstream inputFile;
  int index;

  sprintf(name, "%sP%d_%s.m", dataFolder, num, nimi);

  inputFile.open(name);

  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {

        index = i + Nx * (j + Ny * k);
        inputFile >> r1 >> r2;

        if (i >= 0 && j >= 0 && k >= 0) {
          h_PSIS[num - 1][index] = make_cuDoubleComplex(r1, r2);
        }
      }
    }
  }

  inputFile.close();

#ifdef NVTOOLS
  nvtxRangeEnd(loadPsiRange);
#endif
}

//-------------------------------------------------------------------------------------------------------------
// INITIALIZATION//
void init(/*int tag*/) {

#ifdef NVTOOLS
  nvtxRangeId_t initRange = nvtxRangeStartA("init()"); // nvvp annotation
#endif

  int tag;
  int index;
  //  double pol;
  double x, y, z;
  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);

  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Initializing.\n");
  fclose(print_file);

  allocate_memory();

  if (resume) {

    sprintf(nimi, "%06d", stepnum / freq); // loading previous file

    print_file = fopen(std_file, "a+");
    fprintf(
        print_file,
        "Loading state from stepnum %d; datafiles ending with \'...%s.m\'.\n\n",
        stepnum, nimi);
    fclose(print_file);

    tag = resumeInitTag;

  } else {

    stepnum = 0;
    sprintf(nimi, "%06d", 0);
    tag = restartInitTag;
  }

  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {

        index = i + Nx * (j + Ny * k);

        x = -Lx / 2.0 + i * hx;
        y = -Ly / 2.0 + j * hy;
        z = -Lz / 2.0 + k * hz;

        h_x[index] = x;
        h_y[index] = y;
        h_z[index] = z;

        h_trap[index] =
            0.5 * (sqr(x * lambda_x) + sqr(y * lambda_y) + sqr(z * lambda_z)) +
            100.0; // + A*( exp(D1) + exp(D2) ) +100.0;

        if (resume == false) {

          if (tag == 0) {

            h_Mx[index] = 0.0;
            h_My[index] = 0.0;
            h_Mz[index] = 0.0;

            // h_Mz[index] = B0;
            // h_Mx[index] = Bp * x;
            // h_My[index] = Bp * y;
            // h_Mz[index] = -2*Bp * z+B0;

          } else if (tag == 1) {

            h_Mx[index] = Bp * x + Bx;
            h_My[index] = Bp * y;
            h_Mz[index] = Bz * z + B0;

          } else if (tag == 2) {

            h_Mx[index] = -(Bp * x + Bx);
            h_My[index] = -Bp * y;
            h_Mz[index] = -(Bz * z + B0);

            // h_Mx[index] = 0.0;
            // h_My[index] = 0.0;
            // h_Mz[index] = B0;

          } else if (tag == 3) {

            h_Mx[index] = Bp * x;
            h_My[index] = Bp * y;
            h_Mz[index] = Bz * z + B0;

          } else if (tag == 4) {

            h_Mx[index] = Bp * x;
            h_My[index] = Bp * y;
            h_Mz[index] = Bz * z;

          } else if (tag == 5) {

            h_Mx[index] = Bp * x;
            h_My[index] = Bp * y;
            h_Mz[index] = Bz * z;
          }

          if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 &&
              k < Nz - 1) {
            /*
                        h_P1[index]=make_cuDoubleComplex(random_number(),random_number());
                        h_P2[index]=make_cuDoubleComplex(random_number(),random_number());
                        h_P3[index]=make_cuDoubleComplex(random_number(),random_number());
                    h_P4[index]=make_cuDoubleComplex(random_number(),random_number());
                    h_P5[index]=make_cuDoubleComplex(random_number(),random_number());
            */

            // cyclic_1 potkasu
            h_P1[index] = make_cuDoubleComplex(0.25, 0.1);
            h_P2[index] = make_cuDoubleComplex(0.1, 0.1);
            h_P3[index] = make_cuDoubleComplex(0.1, 0.5);
            h_P4[index] = make_cuDoubleComplex(0.1, 0.1);
            h_P5[index] = make_cuDoubleComplex(0.25, 0.1);

            /*
                        // cyclic_2 potkasu
                        h_P1[index]=make_cuDoubleComplex(1.0,1.0);
                        h_P2[index]=make_cuDoubleComplex(0.1,0.1);
                        h_P3[index]=make_cuDoubleComplex(0.1,0.1);
                    h_P4[index]=make_cuDoubleComplex(1.0,1.0);
                    h_P5[index]=make_cuDoubleComplex(0.1,0.1);
            */
          } else {

            h_P1[index] = make_cuDoubleComplex(0.0, 0.0);
            h_P2[index] = make_cuDoubleComplex(0.0, 0.0);
            h_P3[index] = make_cuDoubleComplex(0.0, 0.0);
            h_P4[index] = make_cuDoubleComplex(0.0, 0.0);
            h_P5[index] = make_cuDoubleComplex(0.0, 0.0);
          }
        }
      }
    }
  }

  if (resume) {

    char str[30];

    // check that the files which we are resuming from exist.
    for (int i = 1; i <= 5; i++) {
      ifstream f;
      sprintf(str, "%sP%d_%s.m", dataFolder, i, nimi);
      f.open(str);
      if (!f.good()) {
        print_file = fopen(std_file, "a+");
        fprintf(print_file, "Error resuming: Could not find datafile \"%s\".\n",
                str);
        fclose(print_file);
        f.close();
        exit(EXIT_FAILURE);
      }

      f.close();
    }
#ifdef CPP11
    thread t1(&load_psi, 1);
    thread t2(&load_psi, 2);
    thread t3(&load_psi, 3);
    thread t4(&load_psi, 4);
    thread t5(&load_psi, 5);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
#else
    load_psi(1);
    load_psi(2);
    load_psi(3);
    load_psi(4);
    load_psi(5);
#endif
  }

  /*
   //reading the polynomial constants
   strcpy(str,"Polynomialdataexp/polynomial_q.m");
   tiedosto.open (str);
   for (int k=0; k<9; k++){
     tiedosto >> pol;
     h_pol_q[k]=pol;
   }
   tiedosto.close();

   strcpy(str,"Polynomialdataexp/polynomial_p.m");
   tiedosto.open (str);
   for (int k=0; k<9; k++){
     tiedosto >> pol;
     h_pol_p[k]=pol;
   }
   tiedosto.close();

   strcpy(str,"Polynomialdataexp/polynomial_bx.m");
   tiedosto.open (str);
   for (int k=0; k<9; k++){
     tiedosto >> pol;
     h_pol_bx[k]=pol;
   }
   tiedosto.close();
  */
  /*
   CUDA_SAFE_CALL(cudaMemcpy(d_pol_q, h_pol_q, 9*sizeof(double),
   cudaMemcpyHostToDevice)); CUDA_SAFE_CALL(cudaMemcpy(d_pol_p, h_pol_p,
   9*sizeof(double), cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_pol_bx, h_pol_bx, 9*sizeof(double),
   cudaMemcpyHostToDevice));
   */
  CUDA_SAFE_CALL(
      cudaMemcpy(d_trap, h_trap, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_Mx, h_Mx, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_My, h_My, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_Mz, h_Mz, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_y, h_y, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_x, h_x, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_z, h_z, N * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_P1, h_P1, N * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_P2, h_P2, N * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_P3, h_P3, N * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_P4, h_P4, N * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_P5, h_P5, N * sizeof(cuDoubleComplex),
                            cudaMemcpyHostToDevice));

  // some initial computations:

  compute_Ks<<<grid, block_size>>>(d_TK);
  cudaSyncAndErrorCheck();

  form_density<<<grid, block_size>>>(d_density, d_P1, d_P2, d_P3, d_P4, d_P5);
  cudaSyncAndErrorCheck();
  *h_norm = thrust::reduce(dev_dens, dev_dens + N, (double)0.0,
                           thrust::plus<double>()) *
            hx * hy * hz;

  CUDA_SAFE_CALL(
      cudaMemcpy(d_norm, h_norm, sizeof(double), cudaMemcpyHostToDevice));

  if (!resume) {

    normalize_psi<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_norm);
    cudaSyncAndErrorCheck();
  }

#ifdef NVTOOLS
  nvtxRangeEnd(initRange);
#endif
}
//----------------------------------------------------------------------------------------------------

void SOR(const int component) {
  // Serial ground state computation. Not adapted for spin-2.
  int looper = 0;
  int index;
  double x, y, z;
  double d, error = GP_tol + 1.0;
  complex<double> ddx, ddy, ddz, AP;
  double *trap_SOR = new double[N];
  double *error_fct = new double[N];
  complex<double> *P_SOR = new complex<double>[N];
  complex<double> *HP_SOR = new complex<double>[N];

  if (component == 1) {
    for (int i = 0; i < Nx; i++) {
      for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < Nz; k++) {
          index = i + Nx * (j + Ny * k);
          x = (i - Nx / 2 + 0.5) * hx;
          y = (j - Ny / 2 + 0.5) * hy;
          z = (k - Nz / 2 + 0.5) * hz;
          if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 &&
              k < Nz - 1) {
            P_SOR[index] =
                complex<double>(exp(-0.5 * (sqr(x) + sqr(y) + sqr(z))), 0.0);
          } else {
            P_SOR[index] = complex<double>(0.0, 0.0);
          }
          trap_SOR[index] =
              0.5 * (sqr(x * lambda_x) + sqr(y * lambda_y) + sqr(z * lambda_z));
        }
      }
    }
  } else if (component == 2) {
    for (int i = 0; i < Nx; i++) {
      for (int j = 0; j < Ny; j++) {
        for (int k = 0; k < Nz; k++) {
          index = i + Nx * (j + Ny * k);
          x = (i - Nx / 2 + 0.5) * hx;
          y = (j - Ny / 2 + 0.5) * hy;
          z = (k - Nz / 2 + 0.5) * hz;
          if (i > 0 && i < Nx - 1 && j > 0 && j < Ny - 1 && k > 0 &&
              k < Nz - 1) {
            P_SOR[index] =
                complex<double>(exp(-0.5 * (sqr(x) + sqr(y) + sqr(z))), 0.0);
          } else {
            P_SOR[index] = complex<double>(0.0, 0.0);
          }
          trap_SOR[index] =
              0.5 * (sqr(x * lambda_x) + sqr(y * lambda_y) + sqr(z * lambda_z));
        }
      }
    }
  }

  do {
    for (int i = 0; i < N; ++i)
      h_density[i] = sqr(abs(P_SOR[i]));
    normi = sqrt(riemann_sum(h_density));
    mu = 0.9 * mu + 0.1 * mu / normi;

    for (int i = 1; i < Nx - 1; ++i) {
      for (int j = 1; j < Ny - 1; ++j) {
        for (int k = 1; k < Nz - 1; ++k) {
          index = i + Nx * (j + Ny * k);
          d = -0.5 * dd0 * (1.0 / sqr(hx) + 1.0 / sqr(hy) + 1.0 / sqr(hz)) +
              trap_SOR[index] + (gn + gs) * sqr(abs(P_SOR[index]));
          ddz = (dd1 * (P_SOR[i + Nx * (j + Ny * (k + 1))] +
                        P_SOR[i + Nx * (j + Ny * (k - 1))]) +
                 dd0 * P_SOR[index]) /
                sqr(hz);
          ddy = (dd1 * (P_SOR[i + Nx * ((j + 1) + Ny * k)] +
                        P_SOR[i + Nx * ((j - 1) + Ny * k)]) +
                 dd0 * P_SOR[index]) /
                sqr(hy);
          ddx = (dd1 * (P_SOR[(i + 1) + Nx * (j + Ny * k)] +
                        P_SOR[(i - 1) + Nx * (j + Ny * k)]) +
                 dd0 * P_SOR[index]) /
                sqr(hx);
          AP = -0.5 * (ddx + ddy + ddz) +
               (trap_SOR[index] + (gn + gs) * sqr(abs(P_SOR[index])) - mu) *
                   P_SOR[index];
          P_SOR[index] = P_SOR[index] - relaxation_parameter * AP / d;
        }
      }
    }

    if (looper % 100 == 0) {
      // Error check
      for (int i = 0; i < N; ++i)
        h_density[i] = sqr(abs(P_SOR[i]));

      for (int i = 1; i < Nx - 1; i++) {
        for (int j = 1; j < Ny - 1; j++) {
          for (int k = 1; k < Nz - 1; k++) {
            index = i + Nx * (j + Ny * k);
            ddz = (dd1 * (P_SOR[i + Nx * (j + Ny * (k + 1))] +
                          P_SOR[i + Nx * (j + Ny * (k - 1))]) +
                   dd0 * P_SOR[index]) /
                  sqr(hz);
            ddy = (dd1 * (P_SOR[i + Nx * ((j + 1) + Ny * k)] +
                          P_SOR[i + Nx * ((j - 1) + Ny * k)]) +
                   dd0 * P_SOR[index]) /
                  sqr(hy);
            ddx = (dd1 * (P_SOR[(i + 1) + Nx * (j + Ny * k)] +
                          P_SOR[(i - 1) + Nx * (j + Ny * k)]) +
                   dd0 * P_SOR[index]) /
                  sqr(hx);
            HP_SOR[index] = -0.5 * (ddx + ddy + ddz);
          }
        }
      }
      for (int i = 0; i < N; ++i) {
        HP_SOR[i] = HP_SOR[i] +
                    (trap_SOR[i] + (gn + gs) * h_density[i] - mu) * P_SOR[i];
        error_fct[i] = sqr(abs(conj(P_SOR[i]) * HP_SOR[i]));
      }
      error = sqrt(riemann_sum(error_fct));
      print_file = fopen(std_file, "a+");
      fprintf(print_file, "Loopy: %d \n", looper);
      fprintf(print_file, "Error: %e \n\n", error);
      fclose(print_file);
    }
    ++looper;
  } while (error > GP_tol);

  if (component == 1) {
    for (int i = 0; i < N; ++i)
      h_P1[i] = make_cuDoubleComplex(real(P_SOR[i]), imag(P_SOR[i]));
  } else if (component == 2) {
    for (int i = 0; i < N; ++i)
      h_P2[i] = make_cuDoubleComplex(real(P_SOR[i]), imag(P_SOR[i]));
  }

  delete[] trap_SOR;
  delete[] error_fct;
  delete[] P_SOR;
  delete[] HP_SOR;
}

//------------------------------------------------------------------------------------------------------------------------
void one_time_step(int stepNo) {

  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);

  update_magneticfields<<<grid, block_size>>>(d_Mx, d_My, d_Mz, d_x, d_y, d_z,
                                              stepNo);

  // T, t/2
  for (int i = 0; i < 5; i++) {

    cuDoubleComplex *psi = h_d_PSIS[i];

    cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                 (cufftDoubleComplex *)psi, CUFFT_INVERSE);
    cudaSyncAndErrorCheck();
    divide_by_N<<<grid, block_size>>>(psi);
    cudaSyncAndErrorCheck();
  }

  // V_1, t
  form_diagonal_term_three_body<<<grid, block_size>>>(
      d_apuC1, d_apuC2, d_apuC3, d_apuC4, d_apuC5, d_P1, d_P2, d_P3, d_P4, d_P5,
      d_trap, gn, alpha, beta, d_Mz);
  cudaSyncAndErrorCheck();
  diagonal_term_three_body<<<grid, block_size>>>(d_P1, d_apuC1, time_step);
  cudaSyncAndErrorCheck();
  diagonal_term_three_body<<<grid, block_size>>>(d_P2, d_apuC2, time_step);
  cudaSyncAndErrorCheck();
  diagonal_term_three_body<<<grid, block_size>>>(d_P3, d_apuC3, time_step);
  cudaSyncAndErrorCheck();
  diagonal_term_three_body<<<grid, block_size>>>(d_P4, d_apuC4, time_step);
  cudaSyncAndErrorCheck();
  diagonal_term_three_body<<<grid, block_size>>>(d_P5, d_apuC5, time_step);
  cudaSyncAndErrorCheck();

  // V_2, t

  off_diagonal_term<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_Mx,
                                          d_My, time_step * 0.5);
  cudaSyncAndErrorCheck();

  // T, t/2

  for (int i = 0; i < 5; i++) {

    cuDoubleComplex *psi = h_d_PSIS[i];

    cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                 (cufftDoubleComplex *)psi, CUFFT_FORWARD);
    cudaSyncAndErrorCheck();

    if (stepNo % freq == 0) {
      kinetic_term<<<grid, block_size>>>(psi, d_TK, time_step * 0.5);
      cudaSyncAndErrorCheck();
    } else
      kinetic_term<<<grid, block_size>>>(psi, d_TK, time_step);
    cudaSyncAndErrorCheck();
  }
}

//------------------------------------------------------------------------------------------------------------------------
void residual() {

  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);
  psiHpsi<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_Mx, d_My, d_Mz,
                                d_trap, d_Mu, d_apu_real, 1);
  cudaSyncAndErrorCheck();
  *h_err = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                          thrust::plus<double>()) *
           hx * hy * hz;
  //  device_integrate(d_err, d_apu_real);
  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Error: %e \n\n", *h_err);
  fclose(print_file);
}
void chemPot() {
  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);
  psiHpsi<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_Mx, d_My, d_Mz,
                                d_trap, d_Mu, d_apu_real, 0);
  cudaSyncAndErrorCheck();
  *h_Mu = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                         thrust::plus<double>()) *
          hx * hy * hz;
  //  device_integrate(d_Mu, d_apu_real);
  CUDA_SAFE_CALL(
      cudaMemcpy(d_Mu, h_Mu, sizeof(double), cudaMemcpyHostToDevice));
  // print_file = fopen("stdKnot_n11.out","a+");
  // fprintf(print_file, "Myy: %e \n\n", *h_Mu);
  // fclose(print_file);
}

void compute_L_z() {

  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);
  compute_Lz<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_x, d_y,
                                   d_apu_real);
  cudaSyncAndErrorCheck();
  L_z = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                       thrust::plus<double>()) *
        hx * hy * hz;
}

void suffle(cuDoubleComplex *p1, cuDoubleComplex *p2, cuDoubleComplex *p3,
            cuDoubleComplex *apu1, cuDoubleComplex *apu2, cuDoubleComplex *apu3,
            int tag) {

  unsigned int index1, index2, index3;

  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {
        index1 = i + Nx * (j + Ny * k);
        index2 = j + Ny * (i + Nx * k);
        index3 = k + Nz * (j + Ny * i);

        if (tag == 2) {
          apu1[index2] = p1[index1];
          apu2[index2] = p2[index1];
          apu3[index2] = p3[index1];
        } else if (tag == 3) {
          apu1[index3] = p1[index2];
          apu2[index3] = p2[index2];
          apu3[index3] = p3[index2];
        } else if (tag == 1) {
          apu1[index1] = p1[index3];
          apu2[index1] = p2[index3];
          apu3[index1] = p3[index3];
        }
      }
    }
  }
}

void light_writefile() {
  ofstream tiedosto;
  char str[25];

  sprintf(str, "%sInfo_%s.m", dataFolder, nimi);

  tiedosto.open(str);
  tiedosto << Nx << "\n";
  tiedosto << Ny << "\n";
  tiedosto << Nz << "\n";
  tiedosto << Lx << "\n";
  tiedosto << Ly << "\n";
  tiedosto << Lz << "\n";
  tiedosto << gn << "\n";
  tiedosto << gs << "\n";
  tiedosto << ga << "\n";
  tiedosto << setprecision(15) << E_kin << "\n";
  tiedosto << setprecision(15) << E_pot - 100.0 << "\n";
  tiedosto << setprecision(15) << E_nl << "\n";
  tiedosto << setprecision(15) << E_ss << "\n";
  tiedosto << setprecision(15) << E_th << "\n";
  tiedosto << setprecision(15) << E_mag << "\n";
  tiedosto << setprecision(15) << E_tot - 100.0 << "\n";
  tiedosto << setprecision(15) << L_z << "\n";
  // tiedosto << setprecision(15) << cmz << "\n";
  tiedosto << setprecision(15) << B0 << "\n";
  tiedosto << setprecision(15) << Bp << "\n";
  tiedosto << setprecision(15) << Bx << "\n";

  tiedosto.close();
}

//-------------------------------------------------------------------------------------------------
void heavy_writefile() {

#ifdef NVTOOLS
  nvtxRangeId_t hwRange =
      nvtxRangeStartA("heavy_writefile()"); // nvvp annotation
#endif

  const int iofield_width = 20;
  ofstream tiedosto1, tiedosto2, tiedosto3, tiedosto4, tiedosto5;
  char str1[30], str2[30], str3[30], str4[30], str5[30];
  int index;
#ifdef CPP11
  h_lock.lock();
#endif

  sprintf(str1, "%sP1_%s.m", dataFolder, nimi);
  sprintf(str2, "%sP2_%s.m", dataFolder, nimi);
  sprintf(str3, "%sP3_%s.m", dataFolder, nimi);
  sprintf(str4, "%sP4_%s.m", dataFolder, nimi);
  sprintf(str5, "%sP5_%s.m", dataFolder, nimi);

  tiedosto1.open(str1);
  tiedosto2.open(str2);
  tiedosto3.open(str3);
  tiedosto4.open(str4);
  tiedosto5.open(str5);

  for (int k = 0; k < Nz; k++) {
    for (int j = 0; j < Ny; j++) {
      for (int i = 0; i < Nx; i++) {
        index = i + Nx * (j + Ny * k);

        tiedosto1 << setw(iofield_width) << setprecision(10) << h_P1[index].x
                  << " ";
        tiedosto1 << setw(iofield_width) << setprecision(10) << h_P1[index].y
                  << "\n";

        tiedosto2 << setw(iofield_width) << setprecision(10) << h_P2[index].x
                  << " ";
        tiedosto2 << setw(iofield_width) << setprecision(10) << h_P2[index].y
                  << "\n";

        tiedosto3 << setw(iofield_width) << setprecision(10) << h_P3[index].x
                  << " ";
        tiedosto3 << setw(iofield_width) << setprecision(10) << h_P3[index].y
                  << "\n";

        tiedosto4 << setw(iofield_width) << setprecision(10) << h_P4[index].x
                  << " ";
        tiedosto4 << setw(iofield_width) << setprecision(10) << h_P4[index].y
                  << "\n";

        tiedosto5 << setw(iofield_width) << setprecision(10) << h_P5[index].x
                  << " ";
        tiedosto5 << setw(iofield_width) << setprecision(10) << h_P5[index].y
                  << "\n";
      }
    }
  }

  tiedosto1.close();
  tiedosto2.close();
  tiedosto3.close();
  tiedosto4.close();
  tiedosto5.close();

#ifdef CPP11
  h_lock.unlock();
  h_cv_heavy_write.notify_all();
#endif
#ifdef NVTOOLS
  nvtxRangeEnd(hwRange);
#endif
}

//-------------------------------------------------------------------------------------------------
void calculate_energies(int aika) {

  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);

  form_density<<<grid, block_size>>>(d_density, d_P1, d_P2, d_P3, d_P4, d_P5);
  cudaSyncAndErrorCheck();
  form_spindensity<<<grid, block_size>>>(d_spindensity, d_P1, d_P2, d_P3, d_P4,
                                         d_P5);
  cudaSyncAndErrorCheck();
  form_thetadensity<<<grid, block_size>>>(d_thetadensity, d_P1, d_P2, d_P3,
                                          d_P4, d_P5);
  cudaSyncAndErrorCheck();

  E_kin = 0;

  for (int i = 0; i < 5; i++) {

    cuDoubleComplex *psi = h_d_PSIS[i];

    cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                 (cufftDoubleComplex *)d_A, CUFFT_FORWARD);
    cudaSyncAndErrorCheck();

    multiply_CR_C<<<grid, block_size>>>(d_A, d_TK);
    cudaSyncAndErrorCheck();

    cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)d_A,
                 (cufftDoubleComplex *)d_B, CUFFT_INVERSE);
    cudaSyncAndErrorCheck();
    divide_by_N<<<grid, block_size>>>(d_B);
    cudaSyncAndErrorCheck();

    multiply_kinetic<<<grid, block_size>>>(d_apu_real, psi, d_B);
    cudaSyncAndErrorCheck();

    E_kin += thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                            thrust::plus<double>()) *
             hx * hy * hz;
  }

  multiply_RR_R<<<grid, block_size>>>(d_apu_real, d_density, d_trap);
  cudaSyncAndErrorCheck();

  E_pot = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                         thrust::plus<double>()) *
          hx * hy * hz;

  multiply_nonlinear<<<grid, block_size>>>(d_apu_real, d_density, gn);
  cudaSyncAndErrorCheck();
  E_nl = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                        thrust::plus<double>()) *
         hx * hy * hz;

  multiply_nonlinear<<<grid, block_size>>>(d_apu_real, d_spindensity, gs);
  cudaSyncAndErrorCheck();
  E_ss = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                        thrust::plus<double>()) *
         hx * hy * hz;

  multiply_nonlinear<<<grid, block_size>>>(d_apu_real, d_thetadensity, ga);
  cudaSyncAndErrorCheck();
  E_th = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                        thrust::plus<double>()) *
         hx * hy * hz;

  form_magnetic<<<grid, block_size>>>(d_apu_real, d_P1, d_P2, d_P3, d_P4, d_P5,
                                      d_Mx, d_My, d_Mz);
  cudaSyncAndErrorCheck();
  E_mag = thrust::reduce(dev_apu_real, dev_apu_real + N, (double)0.0,
                         thrust::plus<double>()) *
          hx * hy * hz;

  E_tot = E_kin + E_pot + E_nl + E_ss + E_mag + E_th;

  form_density<<<grid, block_size>>>(d_density, d_P1, d_P2, d_P3, d_P4, d_P5);
  cudaSyncAndErrorCheck();
  *h_norm = thrust::reduce(dev_dens, dev_dens + N, (double)0.0,
                           thrust::plus<double>()) *
            hx * hy * hz;
  CUDA_SAFE_CALL(
      cudaMemcpy(d_norm, h_norm, sizeof(double), cudaMemcpyHostToDevice));
}

//-------------------------------------------------------------------------------------------------
void write_energies(int tag) {

  compute_L_z();

  print_file = fopen(std_file, "a+");
  if (tag >= 0)
    fprintf(print_file, "Time steps: %d \n", tag);
  fprintf(print_file, "Normi:      %15.12f \n", *h_norm);
  if (tag == -1)
    fprintf(print_file, "Chem_pot:   %15.12f \n", *h_Mu - 100.0);
  fprintf(print_file, "Lz:         %15.12f \n", L_z);
  fprintf(print_file, "E_kin:      %15.12f \n", E_kin);
  fprintf(print_file, "E_mag:      %15.12f \n", E_mag);
  fprintf(print_file, "E_pot:      %15.12f \n", E_pot - 100.0);
  fprintf(print_file, "E_nl:       %15.12f \n", E_nl);
  fprintf(print_file, "E_ss:       %15.12f \n", E_ss);
  fprintf(print_file, "E_th:       %15.12f \n", E_th);
  fprintf(print_file, "E_tot:      %15.12f \n\n", E_tot - 100.0);

  fclose(print_file);
}

/////////////////////////////////////////////////////////////////////////
void ground_state() {

  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Calculating ground state.\n");
  fclose(print_file);

  int m = 0;
  float gputime;
  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);

  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));
  CUDA_SAFE_CALL(cudaEventRecord(start, 0));

  //*h_omega = 0.9;
  *h_omega = 1.0;
  *h_norm = 1.0;

  CUDA_SAFE_CALL(
      cudaMemcpy(d_omega, h_omega, sizeof(double), cudaMemcpyHostToDevice));

  chemPot();

  for (int p = 1; p < Niter + 1; p++) {
    m++;

    ////JACOBI/////
    jacobi<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_Mx, d_My, d_Mz,
                                 d_trap, d_Mu, d_omega, d_apuC1, d_apuC2,
                                 d_apuC3, d_apuC4, d_apuC5, p);

    cudaSyncAndErrorCheck();

    if (p == Niter) { // spin flip (the wanted initial state |1,0> is not ground
                      // state)

      CUDA_SAFE_CALL(cudaMemcpy(d_P1, d_apuC1, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P2, d_apuC2, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P3, d_apuC3, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P4, d_apuC4, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P5, d_apuC5, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));

    } else {

      CUDA_SAFE_CALL(cudaMemcpy(d_P1, d_apuC1, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P2, d_apuC2, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P3, d_apuC3, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P4, d_apuC4, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
      CUDA_SAFE_CALL(cudaMemcpy(d_P5, d_apuC5, N * sizeof(cuDoubleComplex),
                                cudaMemcpyDeviceToDevice));
    }

    form_density<<<grid, block_size>>>(d_density, d_P1, d_P2, d_P3, d_P4, d_P5);
    cudaSyncAndErrorCheck();
    *h_norm = thrust::reduce(dev_dens, dev_dens + N, (double)0.0,
                             thrust::plus<double>()) *
              hx * hy * hz;
    CUDA_SAFE_CALL(
        cudaMemcpy(d_norm, h_norm, sizeof(double), cudaMemcpyHostToDevice));
    normalize_psi<<<grid, block_size>>>(d_P1, d_P2, d_P3, d_P4, d_P5, d_norm);
    cudaSyncAndErrorCheck();
    *h_Mu = 0.9 * (*h_Mu) + 0.1 * (*h_Mu) / sqrt(*h_norm);
    CUDA_SAFE_CALL(
        cudaMemcpy(d_Mu, h_Mu, sizeof(double), cudaMemcpyHostToDevice));

    if (p % monitorfreq == 0 || p == Niter) {
      print_file = fopen(std_file, "a+");
      fprintf(print_file, "Round: %d\n", p);
      fclose(print_file);

      CUDA_SAFE_CALL(
          cudaMemcpy(h_norm, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(
          cudaMemcpy(h_Mu, d_Mu, sizeof(double), cudaMemcpyDeviceToHost));

      /// ENERGY//////////
      calculate_energies(0);
      write_energies(-1);

      /// RESIDUAL
      residual();
    }

    if (p % writefreq == 0 || p == Niter) {

      for (int i = 0; i < 5; i++) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(h_PSIS[i], h_d_PSIS[i],
                                       N * sizeof(cuDoubleComplex),
                                       cudaMemcpyDeviceToHost));
      }
      cudaSyncAndErrorCheck();

      heavy_writefile();
      light_writefile();
    }
  }

  CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  CUDA_SAFE_CALL(cudaEventElapsedTime(&gputime, start, stop));
  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));
  print_file = fopen(std_file, "a+");
  fprintf(print_file, "JACOBI:         %g seconds, %d rounds \n",
          gputime / 1.0e3, m);
  fclose(print_file);
}
//----------------------------------------------------------------------------------------------------
void time_propagator() {
  int time_counter = stepnum / freq;

  dim3 grid = make_large_grid(n_threads, block_size, n_blocks);

  if (resume) {
    // time_counter=stepnum/freq;

    if (((stepnum / freq) * freq + 1) > open_trap) {
      // remove optical trapping
      set_to_zero<<<grid, block_size>>>(d_trap);
      cudaSyncAndErrorCheck();
      CUDA_SAFE_CALL(cudaMemcpy(h_trap, d_trap, N * sizeof(double),
                                cudaMemcpyDeviceToHost));
    }
  }

  // initial forward fft and kin term
  for (int i = 0; i < 5; i++) {
    cuDoubleComplex *psi = h_d_PSIS[i];
    cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                 (cufftDoubleComplex *)psi, CUFFT_FORWARD);
    cudaSyncAndErrorCheck();
    kinetic_term<<<grid, block_size>>>(psi, d_TK, time_step * 0.5);
    cudaSyncAndErrorCheck();
  }

#ifdef CPP11
  bool heavy_write_called = false;
#endif

  for (int time = (stepnum / freq) * freq + 1; time < Max_time + 1; ++time) {

    if (time == open_trap) {
      // remove optical trapping
      set_to_zero<<<grid, block_size>>>(d_trap);
      cudaSyncAndErrorCheck();
      CUDA_SAFE_CALL(cudaMemcpy(h_trap, d_trap, N * sizeof(double),
                                cudaMemcpyDeviceToHost));
    }

    one_time_step(time);

    if (time % freq == 0) {

      for (int i = 0; i < 5; i++) {

        cuDoubleComplex *psi = h_d_PSIS[i];
        cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                     (cufftDoubleComplex *)psi, CUFFT_INVERSE);
        divide_by_N<<<grid, block_size>>>(psi);
      }

      cudaSyncAndErrorCheck();

#ifdef CPP11
      h_lock.lock();
#endif
      ++time_counter;
      sprintf(nimi, "%06d", time_counter);

      calculate_energies(time);
      write_energies(time);

      light_writefile();

      for (int i = 0; i < 5; i++) {
        CUDA_SAFE_CALL(cudaMemcpy(h_PSIS[i], h_d_PSIS[i],
                                  N * sizeof(cuDoubleComplex),
                                  cudaMemcpyDeviceToHost));
      }

      cudaSyncAndErrorCheck();
#ifdef CPP11
      h_lock.unlock();
      thread heavy_write_thread(&heavy_writefile);
      heavy_write_thread.detach();
      heavy_write_called = true;

#else
      heavy_writefile();
#endif

      for (int i = 0; i < 5; i++) {
        cuDoubleComplex *psi = h_d_PSIS[i];
        cufftExecZ2Z(CUDA_plan, (cufftDoubleComplex *)psi,
                     (cufftDoubleComplex *)psi, CUFFT_FORWARD);
        kinetic_term<<<grid, block_size>>>(psi, d_TK, time_step * 0.5);
        cudaSyncAndErrorCheck();
      }

      cudaSyncAndErrorCheck();
    }
  }

#ifdef CPP11
  if (heavy_write_called) {
    h_cv_heavy_write.wait(h_lock_u);
  }
#endif
}

int main(void) {

  clock_t start_time, end_time;
  double elapsed_time;

  //  CUDA_SAFE_CALL(cudaSetDevice(1));

  remove(std_file);

  if (block_size < Nx) {
    print_file = fopen(std_file, "a+");
    fprintf(print_file, "Block size too small.\n");
    fclose(print_file);
    exit(EXIT_FAILURE);
  }

  init();

  if (resume) {

    light_writefile();

  } else {

    ground_state();
  }

  // Time propagation

  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Start norm: %f\n", *h_norm);
  fclose(print_file);

  start_time = clock();
  time_propagator();
  end_time = clock();

  elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
  print_file = fopen(std_file, "a+");
  fprintf(print_file, "Time spent on %d steps was %f\n", Max_time - (stepnum),
          elapsed_time);
  fclose(print_file);

  // clear mem

  delete[] h_PSIS;
  delete[] h_d_PSIS;
  delete[] h_P1;
  delete[] h_P2;
  delete[] h_P3;
  delete[] h_P4;
  delete[] h_P5;
  delete[] h_trap;
  delete[] h_Mx;
  delete[] h_My;
  delete[] h_Mz;
  delete[] h_density;
  delete[] h_apuC1;
  delete[] h_apuC2;
  delete[] h_apuC3;
  delete[] h_apuC4;
  delete[] h_apuC5;

  CUDA_SAFE_CALL(cudaFree(d_P1));
  CUDA_SAFE_CALL(cudaFree(d_P2));
  CUDA_SAFE_CALL(cudaFree(d_P3));
  CUDA_SAFE_CALL(cudaFree(d_P4));
  CUDA_SAFE_CALL(cudaFree(d_P5));

  CUDA_SAFE_CALL(cudaFree(d_Mx));
  CUDA_SAFE_CALL(cudaFree(d_My));
  CUDA_SAFE_CALL(cudaFree(d_Mz));

  CUDA_SAFE_CALL(cudaFree(d_density));
  CUDA_SAFE_CALL(cudaFree(d_spindensity));
  CUDA_SAFE_CALL(cudaFree(d_thetadensity));

  CUDA_SAFE_CALL(cudaFree(d_diagonal));
  CUDA_SAFE_CALL(cudaFree(d_trap));

  CUDA_SAFE_CALL(cudaFree(d_fr));
  CUDA_SAFE_CALL(cudaFree(d_fs11));

  CUDA_SAFE_CALL(cudaFree(d_A));
  CUDA_SAFE_CALL(cudaFree(d_B));

  CUDA_SAFE_CALL(cudaFree(d_x));
  CUDA_SAFE_CALL(cudaFree(d_y));
  CUDA_SAFE_CALL(cudaFree(d_z));

  CUDA_SAFE_CALL(cudaFree(d_apu_real));

  CUDA_SAFE_CALL(cudaFree(d_apu1));
  CUDA_SAFE_CALL(cudaFree(d_apu2));
  CUDA_SAFE_CALL(cudaFree(d_apu3));
  CUDA_SAFE_CALL(cudaFree(d_apu4));
  CUDA_SAFE_CALL(cudaFree(d_apu5));

  CUDA_SAFE_CALL(cudaFree(d_apuC1));
  CUDA_SAFE_CALL(cudaFree(d_apuC2));
  CUDA_SAFE_CALL(cudaFree(d_apuC3));
  CUDA_SAFE_CALL(cudaFree(d_apuC4));
  CUDA_SAFE_CALL(cudaFree(d_apuC5));

  return 0;
}
