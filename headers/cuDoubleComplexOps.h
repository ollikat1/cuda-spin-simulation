// cuda double complex arithmetic operations.

/*
__host__ __device__ __inline__ double pow(double p, int q) {
    if(q == 2) {
      return p*p;
    } else {
      return pow(p, (double) q);
    }
//    return pow(p, 2);
}
*/

__host__ __device__ static __inline__ cuDoubleComplex
operator+(const cuDoubleComplex &a, const cuDoubleComplex &b) {
  //  return make_cuDoubleComplex(a.x + b.x, a.y+ b.y);
  return cuCadd(a, b);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator+(const double &a, const cuDoubleComplex &b) {
  return make_cuDoubleComplex(a + b.x, b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator+(const cuDoubleComplex &a, const double &b) {
  return make_cuDoubleComplex(a.x + b, a.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const cuDoubleComplex &a) {
  return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const cuDoubleComplex &a, const cuDoubleComplex &b) {
  //  return make_cuDoubleComplex(a.x - b.x, a.y- b.y);
  return cuCsub(a, b);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const double &a, const cuDoubleComplex &b) {
  return make_cuDoubleComplex(a - b.x, b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator-(const cuDoubleComplex &a, const double &b) {
  return make_cuDoubleComplex(a.x - b, a.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const cuDoubleComplex &a, const cuDoubleComplex &b) {
  return cuCmul(a, b);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const double &a, const cuDoubleComplex &b) {
  return make_cuDoubleComplex(a * b.x, a * b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator*(const cuDoubleComplex &b, const double &a) {
  return make_cuDoubleComplex(a * b.x, a * b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex
operator/(const cuDoubleComplex &b, const double &a) {
  return make_cuDoubleComplex(b.x / a, b.y / a);
}

__host__ __device__ static __inline__ double sqr(double p) { return p * p; }

__host__ __device__ static __inline__ cuDoubleComplex sqr(cuDoubleComplex p) {
  return p * p;
}

__host__ __device__ static __inline__ cuDoubleComplex
operator/(const double &b, const cuDoubleComplex &a) {
  return make_cuDoubleComplex((b * a.x) / (sqr(a.x) + sqr(a.y)),
                              -(b * a.y) / (sqr(a.x) + sqr(a.y)));
}

__host__ __device__ static __inline__ cuDoubleComplex
operator/(const cuDoubleComplex &a, const cuDoubleComplex &b) {
  //  return make_cuDoubleComplex((a.x*b.x+a.y*b.y)/(sqr(b.x) + sqr(b.y))
  //  ,(-a.x*b.y+a.y*b.x)/(sqr(b.x) + sqr(b.y)));
  return cuCdiv(a, b);
}

//-------------------------------------------------------------------------------------------------
__host__ __device__ static __inline__ cuDoubleComplex
cuCexp(const cuDoubleComplex z) {
  cuDoubleComplex res;
  res.x = exp(z.x) * cos(z.y);
  res.y = exp(z.x) * sin(z.y);
  return res;
}
//-------------------------------------------------------------------------------------------------
__host__ __device__ static __inline__ cuDoubleComplex
cuCsin(const cuDoubleComplex z) {
  return 0.5 * make_cuDoubleComplex(0.0, 1.0) *
         (cuCexp(make_cuDoubleComplex(0.0, -1.0) * z) -
          cuCexp(make_cuDoubleComplex(0.0, 1.0) * z));
}
//-------------------------------------------------------------------------------------------------
__host__ __device__ static __inline__ cuDoubleComplex
cuCcos(const cuDoubleComplex z) {
  return 0.5 * (cuCexp(make_cuDoubleComplex(0.0, -1.0) * z) +
                cuCexp(make_cuDoubleComplex(0.0, 1.0) * z));
}
//-------------------------------------------------------------------------------------------------
__host__ __device__ static __inline__ double dexp(const double z) {
  double res;
  res = exp(z);
  return res;
}
//-------------------------------------------------------------------------------------------------
__host__ __device__ static __inline__ double cuCarg(const cuDoubleComplex z) {
  double res;
  res = atan2(cuCimag(z), cuCreal(z));
  return res;
}
