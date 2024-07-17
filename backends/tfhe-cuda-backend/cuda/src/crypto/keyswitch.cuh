#ifndef CNCRT_KS_CUH
#define CNCRT_KS_CUH

#include "device.h"
#include "gadget.cuh"
#include "helper_multi_gpu.h"
#include "polynomial/functions.cuh"
#include "polynomial/polynomial_math.cuh"
#include "torus.cuh"
#include "utils/helper.cuh"
#include "utils/kernel_dimensions.cuh"
#include <thread>
#include <vector>

template <typename Torus>
__device__ Torus *get_ith_block(Torus *ksk, int i, int level,
                                uint32_t lwe_dimension_out,
                                uint32_t level_count) {
  int pos = i * level_count * (lwe_dimension_out + 1) +
            level * (lwe_dimension_out + 1);
  Torus *ptr = &ksk[pos];
  return ptr;
}

/*
 * keyswitch kernel
 * Each thread handles a piece of the following equation:
 * $$GLWE_s2(\Delta.m+e) = (0,0,..,0,b) - \sum_{i=0,k-1} <Dec(a_i),
 * (GLWE_s2(s1_i q/beta),..,GLWE(s1_i q/beta^l)>$$ where k is the dimension of
 * the GLWE ciphertext. If the polynomial dimension in GLWE is > 1, this
 * equation is solved for each polynomial coefficient. where Dec denotes the
 * decomposition with base beta and l levels and the inner product is done
 * between the decomposition of a_i and l GLWE encryptions of s1_i q/\beta^j,
 * with j in [1,l] We obtain a GLWE encryption of Delta.m (with Delta the
 * scaling factor) under key s2 instead of s1, with an increased noise
 *
 */
// Each thread in x are used to calculate one output.
// threads in y are used to paralelize the lwe_dimension_in loop.
// shared memory is used to store intermediate results of the reduction.
template <typename Torus>
__global__ void
keyswitch(Torus *lwe_array_out, const Torus *__restrict__ lwe_output_indexes,
          const Torus *__restrict__ lwe_array_in,
          const Torus *__restrict__ lwe_input_indexes,
          const Torus *__restrict__ ksk, uint32_t lwe_dimension_in,
          uint32_t lwe_dimension_out, uint32_t base_log, uint32_t level_count) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int shmem_index = threadIdx.x + threadIdx.y * blockDim.x;

  extern __shared__ int8_t sharedmem[];
  Torus *lwe_acc_out = (Torus *)sharedmem;
  auto block_lwe_array_out = get_chunk(
      lwe_array_out, lwe_output_indexes[blockIdx.y], lwe_dimension_out + 1);

  if (tid <= lwe_dimension_out) {

    Torus local_lwe_out = 0;
    auto block_lwe_array_in = get_chunk(
        lwe_array_in, lwe_input_indexes[blockIdx.y], lwe_dimension_in + 1);

    if (tid == lwe_dimension_out && threadIdx.y == 0) {
      local_lwe_out = block_lwe_array_in[lwe_dimension_in];
    }
    const Torus mask_mod_b = (1ll << base_log) - 1ll;

    const int pack_size = (lwe_dimension_in + blockDim.y - 1) / blockDim.y;
    const int start_i = pack_size * threadIdx.y;
    const int end_i = SEL(lwe_dimension_in, pack_size * (threadIdx.y + 1),
                          pack_size * (threadIdx.y + 1) <= lwe_dimension_in);

    // This loop distribution seems to benefit the global mem reads
    for (int i = start_i; i < end_i; i++) {
      Torus a_i = round_to_closest_multiple(block_lwe_array_in[i], base_log,
                                            level_count);
      Torus state = a_i >> (sizeof(Torus) * 8 - base_log * level_count);

      for (int j = 0; j < level_count; j++) {
        auto ksk_block =
            get_ith_block(ksk, i, j, lwe_dimension_out, level_count);
        Torus decomposed = decompose_one<Torus>(state, mask_mod_b, base_log);
        local_lwe_out -= (Torus)ksk_block[tid] * decomposed;
      }
    }

    lwe_acc_out[shmem_index] = local_lwe_out;
  }

  if (tid <= lwe_dimension_out) {
    for (int offset = blockDim.y / 2; offset > 0 && threadIdx.y < offset;
         offset /= 2) {
      __syncthreads();
      lwe_acc_out[shmem_index] +=
          lwe_acc_out[shmem_index + offset * blockDim.x];
    }
    if (threadIdx.y == 0)
      block_lwe_array_out[tid] = lwe_acc_out[shmem_index];
  }
}

template <typename Torus>
__host__ void host_keyswitch_lwe_ciphertext_vector(
    cudaStream_t stream, uint32_t gpu_index, Torus *lwe_array_out,
    Torus *lwe_output_indexes, Torus *lwe_array_in, Torus *lwe_input_indexes,
    Torus *ksk, uint32_t lwe_dimension_in, uint32_t lwe_dimension_out,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples) {

  cudaSetDevice(gpu_index);

  constexpr int num_threads_y = 32;
  int num_blocks, num_threads_x;

  getNumBlocksAndThreads2D(lwe_dimension_out + 1, 512, num_threads_y,
                           num_blocks, num_threads_x);

  int shared_mem = sizeof(Torus) * num_threads_y * num_threads_x;
  dim3 grid(num_blocks, num_samples, 1);
  dim3 threads(num_threads_x, num_threads_y, 1);

  keyswitch<Torus><<<grid, threads, shared_mem, stream>>>(
      lwe_array_out, lwe_output_indexes, lwe_array_in, lwe_input_indexes, ksk,
      lwe_dimension_in, lwe_dimension_out, base_log, level_count);
  check_cuda_error(cudaGetLastError());
}

template <typename Torus>
void execute_keyswitch_async(cudaStream_t *streams, uint32_t *gpu_indexes,
                             uint32_t gpu_count,
                             const LweArrayVariant<Torus> &lwe_array_out,
                             const LweArrayVariant<Torus> &lwe_output_indexes,
                             const LweArrayVariant<Torus> &lwe_array_in,
                             const LweArrayVariant<Torus> &lwe_input_indexes,
                             Torus **ksks, uint32_t lwe_dimension_in,
                             uint32_t lwe_dimension_out, uint32_t base_log,
                             uint32_t level_count, uint32_t num_samples) {

  /// If the number of radix blocks is lower than the number of GPUs, not all
  /// GPUs will be active and there will be 1 input per GPU
  for (uint i = 0; i < gpu_count; i++) {
    int num_samples_on_gpu = get_num_inputs_on_gpu(num_samples, i, gpu_count);

    Torus *current_lwe_array_out = GET_VARIANT_ELEMENT(lwe_array_out, i);
    Torus *current_lwe_output_indexes =
        GET_VARIANT_ELEMENT(lwe_output_indexes, i);
    Torus *current_lwe_array_in = GET_VARIANT_ELEMENT(lwe_array_in, i);
    Torus *current_lwe_input_indexes =
        GET_VARIANT_ELEMENT(lwe_input_indexes, i);

    // Compute Keyswitch
    host_keyswitch_lwe_ciphertext_vector<Torus>(
        streams[i], gpu_indexes[i], current_lwe_array_out,
        current_lwe_output_indexes, current_lwe_array_in,
        current_lwe_input_indexes, ksks[i], lwe_dimension_in, lwe_dimension_out,
        base_log, level_count, num_samples_on_gpu);
  }
}

// chunk_count = glwe_size * polynomial_size / threads.
// each threads will responsible to process only lwe_size times multiplication
template <typename Torus>
__device__ void keyswitch_lwe_ciphertext_into_glwe_ciphertext(
    Torus *glwe_out, Torus *lwe_in, Torus *fp_ksk, uint32_t lwe_dimension_in,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count) {

  extern __shared__ int8_t sharedmem[];

  // result accumulator, shared memory is used because of frequent access
  Torus *local_glwe_chunk = (Torus *)sharedmem;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int shmem_index = threadIdx.x;
  // the output_glwe is split in chunks and each x-block takes one of them
  size_t chunk_id = blockIdx.x;
  size_t coef_per_block = blockDim.x;

  // dimensions
  size_t glwe_size = (glwe_dimension + 1);
  size_t lwe_size = (lwe_dimension_in + 1);
  size_t ksk_size = lwe_size * level_count * glwe_size * polynomial_size;
  // number of coefficients inside fp-ksk block for each lwe_input coefficient
  size_t ksk_block_size = glwe_size * polynomial_size * level_count;

  // initialize accumulator to 0
  local_glwe_chunk[shmem_index] =
      SEL(0, lwe_in[lwe_dimension_in], tid == glwe_dimension * polynomial_size);

  // Iterate through all lwe elements
  for (int i = 0; i < lwe_dimension_in; i++) {
    // Round and prepare decomposition
    Torus a_i = round_to_closest_multiple(lwe_in[i], base_log, level_count);

    Torus state = a_i >> (sizeof(Torus) * 8 - base_log * level_count);
    Torus mod_b_mask = (1ll << base_log) - 1ll;

    // block of key for current lwe coefficient (cur_input_lwe[i])
    auto ksk_block = &fp_ksk[i * ksk_block_size];
    for (int j = 0; j < level_count; j++) {
      auto ksk_glwe = &ksk_block[j * glwe_size * polynomial_size];
      // Iterate through each level and multiply by the ksk piece
      auto ksk_glwe_chunk = &ksk_glwe[chunk_id * coef_per_block];
      Torus decomposed = decompose_one<Torus>(state, mod_b_mask, base_log);
      local_glwe_chunk[shmem_index] -= decomposed * ksk_glwe_chunk[shmem_index];
    }
  }

  // Persist
  glwe_out[tid] = local_glwe_chunk[shmem_index];
}

// public functional packing keyswitch
//
// chunk_count = glwe_size * polynomial_size / threads.
template <typename Torus>
__global__ void
fp_keyswitch_lwe_to_glwe(Torus *glwe_out, Torus *lwe_in, Torus *fp_ksk,
                         uint32_t lwe_dimension_in, uint32_t glwe_dimension,
                         uint32_t polynomial_size, uint32_t base_log,
                         uint32_t level_count) {

  keyswitch_lwe_ciphertext_into_glwe_ciphertext(
      glwe_out, lwe_in, fp_ksk, lwe_dimension_in, glwe_dimension,
      polynomial_size, base_log, level_count);
}

template <typename Torus>
__host__ void
host_fp_keyswitch_lwe_to_glwe(cudaStream_t stream, uint32_t gpu_index,
                              Torus *glwe_array_out, Torus *lwe_array_in,
                              Torus *fp_ksk_array, uint32_t lwe_dimension_in,
                              uint32_t glwe_dimension, uint32_t polynomial_size,
                              uint32_t base_log, uint32_t level_count) {
  cudaSetDevice(gpu_index);
  int glwe_accumulator_size = (glwe_dimension + 1) * polynomial_size;

  int num_threads = 0, num_blocks = 0;
  getNumBlocksAndThreads(glwe_accumulator_size, 512, num_blocks, num_threads);
  int shared_mem = sizeof(Torus) * num_threads;
  fp_keyswitch_lwe_to_glwe<<<num_blocks, num_threads, shared_mem, stream>>>(
      glwe_array_out, lwe_array_in, fp_ksk_array, lwe_dimension_in,
      glwe_dimension, polynomial_size, base_log, level_count);
  check_cuda_error(cudaGetLastError());
}

// public functional packing keyswitch
//
// blockIdx.y - input
// chunk_count = glwe_size * polynomial_size / threads.
template <typename Torus>
__global__ void
fp_keyswitch_lwe_list_to_glwe(Torus *glwe_array_out, Torus *lwe_array_in,
                              Torus *fp_ksk, uint32_t lwe_dimension_in,
                              uint32_t glwe_dimension, uint32_t polynomial_size,
                              uint32_t base_log, uint32_t level_count,
                              uint32_t num_samples, Torus *d_mem) {
  int glwe_accumulator_size = (glwe_dimension + 1) * polynomial_size;

  const int input_id = threadIdx.y + blockIdx.y * blockDim.y;
  const int degree = input_id;

  // Select a input
  auto input_d_mem = d_mem + input_id * glwe_accumulator_size;

  // KS LWE to GLWE
  auto ks_glwe_out = input_d_mem;
  auto lwe_in = lwe_array_in + (lwe_dimension_in + 1) * input_id;
  keyswitch_lwe_ciphertext_into_glwe_ciphertext(
      ks_glwe_out, lwe_in, fp_ksk, lwe_dimension_in, glwe_dimension,
      polynomial_size, base_log, level_count);

  // P * x ^degree
  auto glwe_out =
      glwe_array_out + (glwe_dimension + 1) * polynomial_size * input_id;
  polynomial_accumulate_monic_monomial_mul_batch(
      glwe_out, ks_glwe_out, degree, polynomial_size, glwe_dimension + 1, true);
}

template <typename Torus>
__global__ void
accumulate_glwes(Torus *glwe_out, Torus *glwe_array_in, uint32_t glwe_dimension,
                 uint32_t polynomial_size, uint32_t num_samples) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ int8_t sharedmem[];
  const int shmem_index = threadIdx.x;

  Torus *glwe_acc = (Torus *)sharedmem;
  glwe_acc[shmem_index] = glwe_array_in[tid];

  // Accumulate
  for (int i = 1; i < num_samples; i++) {
    auto glwe_in = glwe_array_in + i * (glwe_dimension + 1) * polynomial_size;
    glwe_acc[shmem_index] += glwe_in[tid];
  }

  glwe_out[tid] = glwe_acc[shmem_index];
}

template <typename Torus>
__host__ void host_fp_keyswitch_lwe_list_to_glwe(
    cudaStream_t stream, uint32_t gpu_index, Torus *glwe_out,
    Torus *lwe_array_in, Torus *fp_ksk_array, uint32_t lwe_dimension_in,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count, uint32_t num_samples) {
  cudaSetDevice(gpu_index);
  int glwe_accumulator_size = (glwe_dimension + 1) * polynomial_size;

  int num_blocks = 0, num_threads_x = 0;
  getNumBlocksAndThreads(glwe_accumulator_size, 512, num_blocks, num_threads_x);

  auto shared_mem = sizeof(Torus) * num_threads_x;
  dim3 grid(num_blocks, num_samples);
  dim3 threads(num_threads_x);

  auto d_mem = (Torus *)cuda_malloc_async(
      num_samples * glwe_accumulator_size * sizeof(Torus), stream, gpu_index);
  auto d_tmp_glwe_array_out = (Torus *)cuda_malloc_async(
      num_samples * glwe_accumulator_size * sizeof(Torus), stream, gpu_index);

  // individually keyswitch each lwe
  fp_keyswitch_lwe_list_to_glwe<<<grid, threads, shared_mem, stream>>>(
      d_tmp_glwe_array_out, lwe_array_in, fp_ksk_array, lwe_dimension_in,
      glwe_dimension, polynomial_size, base_log, level_count, num_samples,
      d_mem);
  check_cuda_error(cudaGetLastError());
  // accumulate to a single glwe
  accumulate_glwes<<<num_blocks, num_threads_x, shared_mem, stream>>>(
      glwe_out, d_tmp_glwe_array_out, glwe_dimension, polynomial_size,
      num_samples);
  check_cuda_error(cudaGetLastError());

  cuda_drop_async(d_mem, stream, gpu_index);
  cuda_drop_async(d_tmp_glwe_array_out, stream, gpu_index);
}

#endif
