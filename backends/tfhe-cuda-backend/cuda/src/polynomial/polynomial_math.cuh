#ifndef CUDA_POLYNOMIAL_MATH_CUH
#define CUDA_POLYNOMIAL_MATH_CUH

#include "crypto/torus.cuh"
#include "parameters.cuh"

template <typename T>
__device__ T *get_chunk(T *data, int chunk_num, int chunk_size) {
  int pos = chunk_num * chunk_size;
  T *ptr = &data[pos];
  return ptr;
}

template <typename FT, class params>
__device__ void sub_polynomial(FT *result, FT *first, FT *second) {
  int tid = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    result[tid] = first[tid] - second[tid];
    tid += params::degree / params::opt;
  }
}

template <class params, typename T>
__device__ void polynomial_product_in_fourier_domain(T *result, T *first,
                                                     T *second) {
  int tid = threadIdx.x;
  for (int i = 0; i < params::opt / 2; i++) {
    result[tid] = first[tid] * second[tid];
    tid += params::degree / params::opt;
  }

  if (threadIdx.x == 0) {
    result[params::degree / 2] =
        first[params::degree / 2] * second[params::degree / 2];
  }
}

// Computes result += first * second
// If init_accumulator is set, assumes that result was not initialized and does
// that with the outcome of first * second
template <class params, typename T>
__device__ void polynomial_product_accumulate_in_fourier_domain(
    T *result, T *first, const T *second, bool init_accumulator = false) {
  int tid = threadIdx.x;
  if (init_accumulator) {
    for (int i = 0; i < params::opt / 2; i++) {
      result[tid] = first[tid] * second[tid];
      tid += params::degree / params::opt;
    }
  } else {
    for (int i = 0; i < params::opt / 2; i++) {
      result[tid] += first[tid] * second[tid];
      tid += params::degree / params::opt;
    }
  }
}

// This method expects to work with polynomial_size / params::opt threads in the
// x-block If init_accumulator is set, assumes that result was not initialized
// and does that with the outcome of first * second
template <typename T, class params>
__device__ void
polynomial_accumulate_monic_monomial_mul(T *result, const T *__restrict__ poly,
                                         uint64_t monomial_degree,
                                         bool init_accumulator = false) {
  // monomial_degree \in [0, 2 * params::degree)
  int full_cycles_count = monomial_degree / params::degree;
  int remainder_degrees = monomial_degree % params::degree;

  int pos = threadIdx.x;
  for (int i = 0; i < params::opt; i++) {
    T element = poly[pos];
    int new_pos = (pos + monomial_degree) % params::degree;

    T x = SEL(element, -element, full_cycles_count % 2); // monomial coefficient
    x = SEL(-x, x, new_pos >= remainder_degrees);

    if (init_accumulator)
      result[new_pos] = x;
    else
      result[new_pos] += x;
    pos += params::degree / params::opt;
  }
}

// This method expects to work with num_poly * polynomial_size threads in the
// grid
template <typename T>
__device__ void polynomial_accumulate_monic_monomial_mul_batch(
    T *result_array, T *poly_array, uint64_t monomial_degree,
    uint32_t polynomial_size, uint32_t num_poly,
    bool init_accumulator = false) {
  // monomial_degree \in [0, 2 * params::degree)
  int full_cycles_count = monomial_degree / polynomial_size;
  int remainder_degrees = monomial_degree % polynomial_size;

  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  int pos = tid % polynomial_size;

  // Select a input
  auto poly = poly_array + (tid / polynomial_size) * polynomial_size;
  auto result = result_array + (tid / polynomial_size) * polynomial_size;

  // Calculate the rotation
  T element = poly[pos];
  int new_pos = (pos + monomial_degree) % polynomial_size;

  // Calculate the new coefficient
  T x = SEL(element, -element, full_cycles_count % 2); // monomial coefficient
  x = SEL(-x, x, new_pos >= remainder_degrees);

  // Write result
  if (init_accumulator)
    result[new_pos] = x;
  else
    result[new_pos] += x;
}

#endif // CNCRT_POLYNOMIAL_MATH_H
