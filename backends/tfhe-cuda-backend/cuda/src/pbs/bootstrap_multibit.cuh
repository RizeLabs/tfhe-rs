#ifndef CUDA_MULTIBIT_PBS_CUH
#define CUDA_MULTIBIT_PBS_CUH

#include "bootstrap.h"
#include "bootstrap_fast_low_latency.cuh"
#include "bootstrap_multibit.h"
#include "cooperative_groups.h"
#include "crypto/gadget.cuh"
#include "crypto/ggsw.cuh"
#include "crypto/torus.cuh"
#include "device.h"
#include "fft/bnsmfft.cuh"
#include "fft/twiddles.cuh"
#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include "polynomial/polynomial_math.cuh"
#include "types/complex/operations.cuh"
#include <vector>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include <utils/helper.cuh>

template <typename Torus, class params>
__device__ Torus calculates_monomial_degree(Torus *lwe_array_group,
                                            uint32_t ggsw_idx,
                                            uint32_t grouping_factor) {
  Torus x = 0;
  for (int i = 0; i < grouping_factor; i++) {
    uint32_t mask_position = grouping_factor - (i + 1);
    int selection_bit = (ggsw_idx >> mask_position) & 1;
    x += selection_bit * lwe_array_group[i];
  }

  return rescale_torus_element(
      x, 2 * params::degree); // 2 * params::log2_degree + 1);
}

// Computes `lwe_chunk_size` keybundles in the standard domain but packed as
// complex polynomials
template <typename Torus, class params>
__global__ void device_multi_bit_bootstrap_keybundle(
    Torus *lwe_array_in, Torus *lwe_input_indexes, double2 *keybundle_array,
    Torus *bootstrapping_key, uint32_t lwe_dimension, uint32_t glwe_dimension,
    uint32_t polynomial_size, uint32_t grouping_factor, uint32_t base_log,
    uint32_t level_count, uint32_t lwe_offset, uint32_t lwe_chunk_size,
    uint32_t keybundle_size_per_input) {

  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory = sharedmem;

  // Ids
  uint32_t level_id = blockIdx.z;
  uint32_t glwe_id = blockIdx.y / (glwe_dimension + 1);
  uint32_t poly_id = blockIdx.y % (glwe_dimension + 1);
  uint32_t lwe_iteration = (blockIdx.x % lwe_chunk_size + lwe_offset);
  uint32_t input_idx = blockIdx.x / lwe_chunk_size;
  uint32_t lwe_chunk_id = blockIdx.x % lwe_chunk_size;

  if (lwe_iteration < (lwe_dimension / grouping_factor)) {
    //
    Torus *accumulator = (Torus *)selected_memory;

    Torus *block_lwe_array_in =
        &lwe_array_in[lwe_input_indexes[input_idx] * (lwe_dimension + 1)];

    double2 *keybundle = keybundle_array +
                         // select the input
                         input_idx * keybundle_size_per_input;

    ////////////////////////////////////////////////////////////
    // Computes all keybundles
    uint32_t rev_lwe_iteration =
        ((lwe_dimension / grouping_factor) - lwe_iteration - 1);

    // ////////////////////////////////
    // Keygen guarantees the first term is a constant term of the polynomial, no
    // polynomial multiplication required
    Torus *bsk_slice = get_multi_bit_ith_lwe_gth_group_kth_block(
        bootstrapping_key, 0, rev_lwe_iteration, glwe_id, level_id,
        grouping_factor, 2 * polynomial_size, glwe_dimension, level_count);
    Torus *bsk_poly = bsk_slice + poly_id * params::degree;

    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        bsk_poly, accumulator);

    // Accumulate the other terms
    for (int g = 1; g < (1 << grouping_factor); g++) {

      Torus *bsk_slice = get_multi_bit_ith_lwe_gth_group_kth_block(
          bootstrapping_key, g, rev_lwe_iteration, glwe_id, level_id,
          grouping_factor, 2 * polynomial_size, glwe_dimension, level_count);
      Torus *bsk_poly = bsk_slice + poly_id * params::degree;

      // Calculates the monomial degree
      Torus *lwe_array_group =
          block_lwe_array_in + rev_lwe_iteration * grouping_factor;
      uint32_t monomial_degree = calculates_monomial_degree<Torus, params>(
          lwe_array_group, g, grouping_factor);

      synchronize_threads_in_block();
      // Multiply by the bsk element
      polynomial_product_accumulate_by_monomial<Torus, params>(
          accumulator, bsk_poly, monomial_degree, false);
    }

    synchronize_threads_in_block();
    // lwe iteration
    auto keybundle_out =
        get_ith_mask_kth_block(keybundle, lwe_chunk_id, glwe_id, level_id,
                               polynomial_size, glwe_dimension, level_count);
    auto keybundle_poly = keybundle_out + poly_id * params::degree / 2;

    // Move from local memory back to shared memory but as complex
    int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < params::opt / 2; i++) {
      double2 complex_acc = make_double2(
          __ll2double_rn((int64_t)accumulator[tid]),
          __ll2double_rn((int64_t)accumulator[tid + params::degree / 2]));
      complex_acc /= (double)std::numeric_limits<Torus>::max();
      keybundle_poly[tid] = complex_acc;
      tid += params::degree / params::opt;
    }
  }
}

template <class params>
__global__ void device_apply_nsmfft_inplace(double2 *poly_array) {

  extern __shared__ int8_t sharedmem[];
  double2 *fft = (double2 *)sharedmem;

  // Load a complex polynomial in the standard domain
  double2 *poly = poly_array + blockIdx.x * (params::degree / 2);

  copy_polynomial<double2, params::opt / 2, params::degree / params::opt>(poly,
                                                                          fft);

  // Apply the FFT
  synchronize_threads_in_block();
  NSMFFT_direct<HalfDegree<params>>(fft);

  synchronize_threads_in_block();

  // Write to the right place
  copy_polynomial<double2, params::opt / 2, params::degree / params::opt>(fft,
                                                                          poly);
}

template <typename Torus, class params>
__global__ void device_multi_bit_bootstrap_accumulate_step_one(
    Torus *lwe_array_in, Torus *lwe_input_indexes, Torus *lut_vector,
    Torus *lut_vector_indexes, Torus *global_accumulator,
    double2 *global_accumulator_fft, uint32_t lwe_dimension,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t base_log,
    uint32_t level_count, uint32_t lwe_iteration) {

  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;

  selected_memory = sharedmem;

  Torus *accumulator = (Torus *)selected_memory;
  double2 *accumulator_fft =
      (double2 *)accumulator +
      (ptrdiff_t)(sizeof(Torus) * polynomial_size / sizeof(double2));

  Torus *block_lwe_array_in =
      &lwe_array_in[lwe_input_indexes[blockIdx.z] * (lwe_dimension + 1)];

  Torus *block_lut_vector = &lut_vector[lut_vector_indexes[blockIdx.z] *
                                        params::degree * (glwe_dimension + 1)];

  Torus *global_slice =
      global_accumulator +
      (blockIdx.y + blockIdx.z * (glwe_dimension + 1)) * params::degree;

  double2 *global_fft_slice =
      global_accumulator_fft +
      (blockIdx.y + blockIdx.x * (glwe_dimension + 1) +
       blockIdx.z * level_count * (glwe_dimension + 1)) *
          (polynomial_size / 2);

  if (lwe_iteration == 0) {
    // First iteration
    ////////////////////////////////////////////////////////////
    // Initializes the accumulator with the body of LWE
    // Put "b" in [0, 2N[
    Torus b_hat = 0;
    rescale_torus_element(block_lwe_array_in[lwe_dimension], b_hat,
                          2 * params::degree);

    divide_by_monomial_negacyclic_inplace<Torus, params::opt,
                                          params::degree / params::opt>(
        accumulator, &block_lut_vector[blockIdx.y * params::degree], b_hat,
        false);

    // Persist
    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        accumulator, global_slice);
  } else {
    // Load the accumulator calculated in previous iterations
    copy_polynomial<Torus, params::opt, params::degree / params::opt>(
        global_slice, accumulator);
  }

  // Perform a rounding to increase the accuracy of the
  // bootstrapped ciphertext
  round_to_closest_multiple_inplace<Torus, params::opt,
                                    params::degree / params::opt>(
      accumulator, base_log, level_count);

  // Decompose the accumulator. Each block gets one level of the
  // decomposition, for the mask and the body (so block 0 will have the
  // accumulator decomposed at level 0, 1 at 1, etc.)
  GadgetMatrix<Torus, params> gadget_acc(base_log, level_count, accumulator);
  gadget_acc.decompose_and_compress_next_polynomial(accumulator_fft,
                                                    blockIdx.x);

  // We are using the same memory space for accumulator_fft and
  // accumulator_rotated, so we need to synchronize here to make sure they
  // don't modify the same memory space at the same time
  // Switch to the FFT space
  NSMFFT_direct<HalfDegree<params>>(accumulator_fft);

  copy_polynomial<double2, params::opt / 2, params::degree / params::opt>(
      accumulator_fft, global_fft_slice);
}

template <typename Torus, class params>
__global__ void device_multi_bit_bootstrap_accumulate_step_two(
    Torus *lwe_array_out, Torus *lwe_output_indexes, double2 *keybundle_array,
    Torus *global_accumulator, double2 *global_accumulator_fft,
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t grouping_factor, uint32_t iteration,
    uint32_t lwe_offset, uint32_t lwe_chunk_size) {
  // We use shared memory for the polynomials that are used often during the
  // bootstrap, since shared memory is kept in L1 cache and accessing it is
  // much faster than global memory
  extern __shared__ int8_t sharedmem[];
  int8_t *selected_memory;

  selected_memory = sharedmem;
  double2 *accumulator_fft = (double2 *)selected_memory;

  double2 *keybundle = keybundle_array +
                       // select the input
                       blockIdx.x * lwe_chunk_size * level_count *
                           (glwe_dimension + 1) * (glwe_dimension + 1) *
                           (polynomial_size / 2);

  double2 *global_accumulator_fft_input =
      global_accumulator_fft +
      blockIdx.x * level_count * (glwe_dimension + 1) * (polynomial_size / 2);

  for (int level = 0; level < level_count; level++) {
    double2 *global_fft_slice =
        global_accumulator_fft_input +
        level * (glwe_dimension + 1) * (polynomial_size / 2);

    for (int j = 0; j < (glwe_dimension + 1); j++) {
      double2 *fft = global_fft_slice + j * params::degree / 2;

      // Get the bootstrapping key piece necessary for the multiplication
      // It is already in the Fourier domain
      auto bsk_slice =
          get_ith_mask_kth_block(keybundle, iteration, j, level,
                                 polynomial_size, glwe_dimension, level_count);
      auto bsk_poly = bsk_slice + blockIdx.y * params::degree / 2;

      polynomial_product_accumulate_in_fourier_domain<params, double2>(
          accumulator_fft, fft, bsk_poly, !level && !j);
    }
  }

  // Perform the inverse FFT on the result of the GGSW x GLWE and add to the
  // accumulator
  NSMFFT_inverse<HalfDegree<params>>(accumulator_fft);
  Torus *global_slice =
      global_accumulator +
      (blockIdx.y + blockIdx.x * (glwe_dimension + 1)) * params::degree;

  add_to_torus<Torus, params>(accumulator_fft, global_slice, true);
  synchronize_threads_in_block();

  uint32_t lwe_iteration = iteration + lwe_offset;
  if (lwe_iteration + 1 == (lwe_dimension / grouping_factor)) {
    // Last iteration
    auto block_lwe_array_out =
        &lwe_array_out[lwe_output_indexes[blockIdx.x] *
                           (glwe_dimension * polynomial_size + 1) +
                       blockIdx.y * polynomial_size];

    if (blockIdx.y < glwe_dimension) {
      // Perform a sample extract. At this point, all blocks have the result,
      // but we do the computation at block 0 to avoid waiting for extra blocks,
      // in case they're not synchronized
      sample_extract_mask<Torus, params>(block_lwe_array_out, global_slice);
    } else if (blockIdx.y == glwe_dimension) {
      sample_extract_body<Torus, params>(block_lwe_array_out, global_slice, 0);
    }
  }
}
template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_full_sm_multibit_bootstrap_keybundle(uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size; // accumulator
}

template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_full_sm_multibit_bootstrap_step_one(uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size * 2; // accumulator
}
template <typename Torus>
__host__ __device__ uint64_t
get_buffer_size_full_sm_multibit_bootstrap_step_two(uint32_t polynomial_size) {
  return sizeof(Torus) * polynomial_size; // accumulator
}

template <typename Torus>
__host__ __device__ uint64_t get_buffer_size_multibit_bootstrap(
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t level_count,
    uint32_t input_lwe_ciphertext_count, uint32_t lwe_chunk_size) {

  uint64_t buffer_size = 0;
  buffer_size += input_lwe_ciphertext_count * lwe_chunk_size * level_count *
                 (glwe_dimension + 1) * (glwe_dimension + 1) *
                 (polynomial_size / 2) * sizeof(double2); // keybundle fft
  buffer_size += input_lwe_ciphertext_count * (glwe_dimension + 1) *
                 level_count * (polynomial_size / 2) *
                 sizeof(double2); // global_accumulator_fft
  buffer_size += input_lwe_ciphertext_count * (glwe_dimension + 1) *
                 polynomial_size * sizeof(Torus); // global_accumulator

  return buffer_size + buffer_size % sizeof(double2);
}

template <typename Torus, typename STorus, typename params>
__host__ void scratch_multi_bit_pbs(
    cuda_stream_t *stream, pbs_multibit_buffer<uint64_t> **pbs_buffer,
    uint32_t lwe_dimension, uint32_t glwe_dimension, uint32_t polynomial_size,
    uint32_t level_count, uint32_t input_lwe_ciphertext_count,
    uint32_t grouping_factor, bool allocate_gpu_memory) {
  cudaSetDevice(stream->gpu_index);

  uint64_t full_sm_keybundle =
      get_buffer_size_full_sm_multibit_bootstrap_keybundle<Torus>(
          polynomial_size);
  uint64_t full_sm_accumulate_step_one =
      get_buffer_size_full_sm_multibit_bootstrap_step_one<Torus>(
          polynomial_size);
  uint64_t full_sm_accumulate_step_two =
      get_buffer_size_full_sm_multibit_bootstrap_step_two<Torus>(
          polynomial_size);

  check_cuda_error(cudaFuncSetAttribute(
      device_multi_bit_bootstrap_keybundle<Torus, params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, full_sm_keybundle));
  cudaFuncSetCacheConfig(device_multi_bit_bootstrap_keybundle<Torus, params>,
                         cudaFuncCachePreferShared);
  check_cuda_error(cudaGetLastError());

  check_cuda_error(cudaFuncSetAttribute(
      device_multi_bit_bootstrap_accumulate_step_one<Torus, params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      full_sm_accumulate_step_one));
  cudaFuncSetCacheConfig(
      device_multi_bit_bootstrap_accumulate_step_one<Torus, params>,
      cudaFuncCachePreferShared);
  check_cuda_error(cudaGetLastError());

  check_cuda_error(cudaFuncSetAttribute(
      device_multi_bit_bootstrap_accumulate_step_two<Torus, params>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      full_sm_accumulate_step_two));
  cudaFuncSetCacheConfig(
      device_multi_bit_bootstrap_accumulate_step_two<Torus, params>,
      cudaFuncCachePreferShared);
  check_cuda_error(cudaGetLastError());

  (*pbs_buffer) = new pbs_multibit_buffer<Torus>(
      stream, glwe_dimension, polynomial_size, level_count,
      input_lwe_ciphertext_count, allocate_gpu_memory);
}

template <typename Torus, class params>
void producer_thread(cuda_stream_t *producer_stream, int producer_id,
                     int num_producers, Torus *lwe_array_in,
                     Torus *lwe_input_indexes, Torus *bootstrapping_key,
                     pbs_multibit_buffer<Torus> *pbs_buffer,
                     uint32_t glwe_dimension, uint32_t lwe_dimension,
                     uint32_t polynomial_size, uint32_t grouping_factor,
                     uint32_t base_log, uint32_t level_count,
                     uint32_t num_samples, std::condition_variable &cv_producer,
                     std::condition_variable &cv_consumer, std::mutex &mtx,
                     std::queue<std::pair<uint32_t, double2 *>> &queue,
                     int max_pool_size) {

  uint32_t lwe_chunk_size = pbs_buffer->lwe_chunk_size;

  uint64_t full_sm_keybundle =
      get_buffer_size_full_sm_multibit_bootstrap_keybundle<Torus>(
          polynomial_size);

  uint32_t keybundle_size_per_input =
      lwe_chunk_size * level_count * (glwe_dimension + 1) *
      (glwe_dimension + 1) * (polynomial_size / 2);

  dim3 thds(polynomial_size / params::opt, 1, 1);

  std::queue<double2 *> keybundle_buffer;

  for (uint32_t lwe_offset = producer_id * lwe_chunk_size;
       lwe_offset < (lwe_dimension / grouping_factor);
       lwe_offset += num_producers * lwe_chunk_size) {

    uint32_t chunk_size = std::min(
        lwe_chunk_size, (lwe_dimension / grouping_factor) - lwe_offset);

    auto keybundle_array = (double2 *)cuda_malloc_async(
        num_samples * lwe_chunk_size * level_count * (glwe_dimension + 1) *
            (glwe_dimension + 1) * (polynomial_size / 2) * sizeof(double2),
        producer_stream);

    // Compute a keybundle
    dim3 grid_keybundle(num_samples * chunk_size,
                        (glwe_dimension + 1) * (glwe_dimension + 1),
                        level_count);

    device_multi_bit_bootstrap_keybundle<Torus, params>
        <<<grid_keybundle, thds, full_sm_keybundle, producer_stream->stream>>>(
            lwe_array_in, lwe_input_indexes, keybundle_array, bootstrapping_key,
            lwe_dimension, glwe_dimension, polynomial_size, grouping_factor,
            base_log, level_count, lwe_offset, chunk_size,
            keybundle_size_per_input);
    check_cuda_error(cudaGetLastError());

    dim3 grid_fft(num_samples * chunk_size * (glwe_dimension + 1) *
                  (glwe_dimension + 1) * level_count);
    device_apply_nsmfft_inplace<params>
        <<<grid_fft, thds, full_sm_keybundle, producer_stream->stream>>>(
            keybundle_array);
    check_cuda_error(cudaGetLastError());

    keybundle_buffer.push(keybundle_array);
    if (keybundle_buffer.size() > max_pool_size || queue.empty() ||
        lwe_offset + num_producers * lwe_chunk_size >=
            (lwe_dimension / grouping_factor)) {
      // Dump keybundle_buffer into queue if it is too big, queue is empty (so
      // the consumer may be waiting) or this is the last iteration

      // We need to be sure the keybundle was computed before we can send it to
      // the consumer
      cuda_synchronize_stream(producer_stream);
      while (!keybundle_buffer.empty()) {
        std::unique_lock<std::mutex> lock(mtx);
        // Wait until queue is almost empty
        cv_producer.wait(lock, [&] { return queue.size() <= 1; });
        queue.push({lwe_offset, keybundle_buffer.front()});
        cv_consumer.notify_one();
        keybundle_buffer.pop();
      }
    }
  }
}

template <typename Torus, class params>
void consumer_thread(
    cuda_stream_t *consumer_stream, Torus *lwe_array_out,
    Torus *lwe_output_indexes, Torus *lut_vector, Torus *lut_vector_indexes,
    Torus *lwe_array_in, Torus *lwe_input_indexes,
    pbs_multibit_buffer<Torus> *pbs_buffer, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t grouping_factor,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples,
    std::vector<std::condition_variable> &cv_producers,
    std::condition_variable &cv_consumer, std::vector<std::mutex> &mtx,
    std::vector<std::queue<std::pair<uint32_t, double2 *>>> &keybundle_pool) {

  uint32_t lwe_chunk_size = pbs_buffer->lwe_chunk_size;

  uint64_t full_sm_accumulate_step_one =
      get_buffer_size_full_sm_multibit_bootstrap_step_one<Torus>(
          polynomial_size);
  uint64_t full_sm_accumulate_step_two =
      get_buffer_size_full_sm_multibit_bootstrap_step_two<Torus>(
          polynomial_size);

  dim3 thds(polynomial_size / params::opt, 1, 1);

  //
  double2 *global_accumulator_fft = pbs_buffer->global_accumulator_fft;
  Torus *global_accumulator = pbs_buffer->global_accumulator;

  dim3 grid_accumulate_step_one(level_count, glwe_dimension + 1, num_samples);
  dim3 grid_accumulate_step_two(num_samples, glwe_dimension + 1);

  int num_producers = keybundle_pool.size();

  for (uint32_t lwe_offset = 0; lwe_offset < (lwe_dimension / grouping_factor);
       lwe_offset += lwe_chunk_size) {

    uint32_t chunk_size = std::min(
        lwe_chunk_size, (lwe_dimension / grouping_factor) - lwe_offset);

    int producer_id = (lwe_offset / lwe_chunk_size) % num_producers;
    auto &producer_queue = keybundle_pool[producer_id];
    std::pair<uint32_t, double2 *> pair;
    {
      std::unique_lock<std::mutex> lock(mtx[producer_id]);
      // Wait until the producer inserts the right keybundle in the pool
      //                  printf("consumer - %d) Will wait (%d elements) for
      //                  queue %p\n", lwe_offset,
      //                         producer_queue.size(), &producer_queue);
      cv_consumer.wait(lock, [&] { return !producer_queue.empty(); });
      //                  printf("consumer - %d) Consuming...(%d elements)\n",
      //                  lwe_offset, producer_queue.size());
      pair = producer_queue.front();
      producer_queue.pop();
      cv_producers[producer_id].notify_one();
    }
    assert(pair.first == lwe_offset);
    double2 *keybundle_fft = pair.second;
    // Accumulate
    for (int j = 0; j < chunk_size; j++) {
      device_multi_bit_bootstrap_accumulate_step_one<Torus, params>
          <<<grid_accumulate_step_one, thds, full_sm_accumulate_step_one,
             consumer_stream->stream>>>(
              lwe_array_in, lwe_input_indexes, lut_vector, lut_vector_indexes,
              global_accumulator, global_accumulator_fft, lwe_dimension,
              glwe_dimension, polynomial_size, base_log, level_count,
              j + lwe_offset);
      check_cuda_error(cudaGetLastError());

      device_multi_bit_bootstrap_accumulate_step_two<Torus, params>
          <<<grid_accumulate_step_two, thds, full_sm_accumulate_step_two,
             consumer_stream->stream>>>(
              lwe_array_out, lwe_output_indexes, keybundle_fft,
              global_accumulator, global_accumulator_fft, lwe_dimension,
              glwe_dimension, polynomial_size, level_count, grouping_factor, j,
              lwe_offset, lwe_chunk_size);
      check_cuda_error(cudaGetLastError());
    }
    cuda_drop_async(keybundle_fft, consumer_stream);
  }
}

template <typename Torus, typename STorus, class params>
__host__ void host_multi_bit_pbs(
    cuda_stream_t *stream, Torus *lwe_array_out, Torus *lwe_output_indexes,
    Torus *lut_vector, Torus *lut_vector_indexes, Torus *lwe_array_in,
    Torus *lwe_input_indexes, Torus *bootstrapping_key,
    pbs_multibit_buffer<Torus> *pbs_buffer, uint32_t glwe_dimension,
    uint32_t lwe_dimension, uint32_t polynomial_size, uint32_t grouping_factor,
    uint32_t base_log, uint32_t level_count, uint32_t num_samples,
    uint32_t num_luts, uint32_t lwe_idx) {
  cudaSetDevice(stream->gpu_index);

  int num_producers = pbs_buffer->num_producers;
  int max_pool_size = pbs_buffer->max_pool_size;

  //
  std::vector<std::queue<std::pair<uint32_t, double2 *>>> keybundle_pool(
      num_producers);
  std::vector<std::condition_variable> cv_producers(num_producers);
  std::condition_variable cv_consumer;
  std::vector<std::mutex> mtx(num_producers);

  std::vector<std::thread> producer_threads;

  // We have to assert everything on the main stream is done to safely launch
  // the producer streams
  //  cuda_synchronize_stream(stream);
  cudaEvent_t main_stream_event;
  cudaEventCreateWithFlags(&main_stream_event, cudaEventDisableTiming);
  cudaEventRecord(main_stream_event, stream->stream);
  for (int producer_id = 0; producer_id < num_producers; producer_id++) {

    std::thread producer([stream, main_stream_event, producer_id, num_producers,
                          lwe_array_in, lwe_input_indexes, bootstrapping_key,
                          pbs_buffer, glwe_dimension, lwe_dimension,
                          polynomial_size, grouping_factor, base_log,
                          level_count, num_samples, &cv_producers, &cv_consumer,
                          &mtx, &keybundle_pool, max_pool_size]() {
      auto producer_stream = cuda_create_stream(stream->gpu_index);
      cudaStreamWaitEvent(producer_stream->stream, main_stream_event, 0);
      producer_thread<Torus, params>(
          producer_stream, producer_id, num_producers, lwe_array_in,
          lwe_input_indexes, bootstrapping_key, pbs_buffer, glwe_dimension,
          lwe_dimension, polynomial_size, grouping_factor, base_log,
          level_count, num_samples, cv_producers[producer_id], cv_consumer,
          mtx[producer_id], keybundle_pool[producer_id], max_pool_size);
      cuda_synchronize_stream(producer_stream);
      cuda_destroy_stream(producer_stream);
    });

    producer_threads.emplace_back(std::move(producer));
  }
  //    cudaEventDestroy(main_stream_event);

  // std::thread consumer([&]() {
  //   auto consumer_stream = cuda_create_stream(gpu_index);
  consumer_thread<Torus, params>(
      stream, lwe_array_out, lwe_output_indexes, lut_vector, lut_vector_indexes,
      lwe_array_in, lwe_input_indexes, pbs_buffer, glwe_dimension,
      lwe_dimension, polynomial_size, grouping_factor, base_log, level_count,
      num_samples, cv_producers, cv_consumer, mtx, keybundle_pool);
  //  cuda_synchronize_stream(consumer_stream);
  //  cuda_destroy_stream(consumer_stream);
  //});

  for (auto &producer : producer_threads) {
    producer.join();
  }
  //  consumer.join();
}
#endif // MULTIBIT_PBS_H
