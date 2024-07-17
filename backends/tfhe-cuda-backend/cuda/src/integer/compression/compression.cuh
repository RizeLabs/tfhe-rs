#ifndef CUDA_INTEGER_COMPRESSION_CUH
#define CUDA_INTEGER_COMPRESSION_CUH

#include "crypto/keyswitch.cuh"
#include "device.h"
#include "integer.h"
#include "integer/integer.cuh"
#include "linearalgebra/multiplication.cuh"
#include "polynomial/functions.cuh"
#include "utils/kernel_dimensions.cuh"

template <typename Torus>
__host__ void host_integer_compression_compress(
    cudaStream_t *streams, uint32_t *gpu_indexes, uint32_t gpu_count,
    Torus *glwe_array_out, Torus *lwe_array_in, Torus **fp_ksk,
    uint32_t num_lwes, int_compression<Torus> *mem_ptr) {
  auto params = mem_ptr->params;

  // Shift
  auto lwe_shifted = mem_ptr->tmp_lwe_shifted;
  host_cleartext_multiplication(streams[0], gpu_indexes[0], lwe_shifted,
                                lwe_array_in, (uint64_t)params.message_modulus,
                                params.big_lwe_dimension, num_lwes);

  uint32_t lwe_in_size = params.big_lwe_dimension + 1;
  uint32_t glwe_out_size = (params.glwe_dimension + 1) * params.polynomial_size;
  uint32_t num_glwes = num_lwes / mem_ptr->lwe_per_glwe;

  // Keyswitch LWEs to GLWE
  for (int i = 0; i < num_glwes; i++) {
    auto lwe_subset = lwe_shifted + i * lwe_in_size;
    auto glwe_out = glwe_array_out + i * glwe_out_size;

    host_fp_keyswitch_lwe_list_to_glwe(
        streams[0], gpu_indexes[0], glwe_out, lwe_subset, fp_ksk[0],
        params.big_lwe_dimension, params.glwe_dimension, params.polynomial_size,
        params.ks_base_log, params.ks_level, mem_ptr->lwe_per_glwe);
  }

  // Modulus switch
  int num_blocks = 0, num_threads = 0;
  getNumBlocksAndThreads(glwe_out_size, 512, num_blocks, num_threads);
  apply_modulus_switch_inplace<<<num_blocks, num_threads, 0, streams[0]>>>(
      glwe_array_out, num_glwes * glwe_out_size, mem_ptr->storage_log_modulus);
}

template <typename Torus>
__host__ void host_integer_compression_decompress(
    cudaStream_t *streams, uint32_t *gpu_indexes, uint32_t gpu_count,
    Torus *lwe_out, Torus *glwe_array_in, void **bsks,
    int_compression<Torus> *mem_ptr) {}

template <typename Torus>
__host__ void scratch_cuda_compression_integer_radix_ciphertext_64(
    cudaStream_t *streams, uint32_t *gpu_indexes, uint32_t gpu_count,
    int_compression<Torus> **mem_ptr, uint32_t num_lwes,
    int_radix_params params, uint32_t lwe_per_glwe,
    uint32_t storage_log_modulus, COMPRESSION_MODE mode,
    bool allocate_gpu_memory) {

  *mem_ptr = new int_compression<Torus>(
      streams, gpu_indexes, gpu_count, params, num_lwes, lwe_per_glwe,
      storage_log_modulus, mode, allocate_gpu_memory);
}
#endif
