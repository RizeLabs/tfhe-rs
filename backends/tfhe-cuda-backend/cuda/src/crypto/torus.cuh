#ifndef CNCRT_TORUS_CUH
#define CNCRT_TORUS_CUH

#include "device.h"
#include "types/int128.cuh"
#include "utils/kernel_dimensions.cuh"
#include <limits>

template <typename T>
__device__ inline void typecast_double_to_torus(double x, T &r) {
  r = T(x);
}

template <>
__device__ inline void typecast_double_to_torus<uint32_t>(double x,
                                                          uint32_t &r) {
  r = __double2uint_rn(x);
}

template <>
__device__ inline void typecast_double_to_torus<uint64_t>(double x,
                                                          uint64_t &r) {
  // The ull intrinsic does not behave in the same way on all architectures and
  // on some platforms this causes the cmux tree test to fail
  // Hence the intrinsic is not used here
  uint128 nnnn = make_uint128_from_float(x);
  uint64_t lll = nnnn.lo_;
  r = lll;
}

template <typename T>
__device__ inline T round_to_closest_multiple(T x, uint32_t base_log,
                                              uint32_t level_count) {
  T shift = sizeof(T) * 8 - level_count * base_log;
  T mask = 1ll << (shift - 1);
  T b = (x & mask) >> (shift - 1);
  T res = x >> shift;
  res += b;
  res <<= shift;
  return res;
}

template <typename T>
__device__ __forceinline__ void apply_modulus_switch(T input, T &output,
                                                     uint32_t log_modulus) {
  constexpr uint32_t BITS = sizeof(T) * 8;

  output = input + (((T)1) << (BITS - log_modulus - 1));
  output >>= (BITS - log_modulus);
}

template <typename T>
__device__ __forceinline__ T apply_modulus_switch(T input,
                                                  uint32_t log_modulus) {
  T output;
  apply_modulus_switch(input, output, log_modulus);
  return output;
}

template <typename Torus>
__global__ void apply_modulus_switch_inplace(Torus *array, int size,
                                             uint32_t log_modulus) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    array[tid] = apply_modulus_switch(array[tid], log_modulus);
  }
}

template <typename Torus>
__host__ void modulus_switch(cudaStream_t stream, uint32_t gpu_index,
                             Torus *array, int size, uint32_t log_modulus) {

  int num_threads = 0, num_blocks = 0;
  getNumBlocksAndThreads(size, 1024, num_blocks, num_threads);

  apply_modulus_switch_inplace<<<num_blocks, num_threads, 0, stream>>>(
      array, size, log_modulus);
  check_cuda_error(cudaGetLastError());
}

#endif // CNCRT_TORUS_H
