#ifndef HELPER_MULTI_GPU_CUH
#define HELPER_MULTI_GPU_CUH

#include "helper_multi_gpu.h"

/// Allocates the input/output vector for all devices
/// Initializes also the related indexing and initializes it to the trivial
/// index
template <typename Torus>
void multi_gpu_lwe_init(cudaStream_t *streams, uint32_t *gpu_indexes,
                        uint32_t gpu_count, std::vector<Torus *> &dest,
                        std::vector<Torus *> &dest_indexes, uint32_t num_inputs,
                        uint32_t elements_per_input) {
  auto active_gpu_count = get_active_gpu_count(num_inputs, gpu_count);

  auto h_lwe_trivial_indexes = (Torus *)malloc(num_inputs * sizeof(Torus));
  for (int i = 0; i < num_inputs; i++)
    h_lwe_trivial_indexes[i] = i;

  dest.resize(active_gpu_count);
  dest_indexes.resize(active_gpu_count);
#pragma omp parallel for num_threads(active_gpu_count)
  for (uint i = 0; i < active_gpu_count; i++) {
    auto inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
    Torus *d_array = (Torus *)cuda_malloc_async(
        inputs_on_gpu * elements_per_input * sizeof(Torus), streams[i],
        gpu_indexes[i]);
    Torus *d_index_array = (Torus *)cuda_malloc_async(
        inputs_on_gpu * sizeof(Torus), streams[i], gpu_indexes[i]);

    cuda_memcpy_async_to_gpu(d_index_array, h_lwe_trivial_indexes,
                             inputs_on_gpu * sizeof(Torus), streams[i],
                             gpu_indexes[i]);

    dest[i] = d_array;
    dest_indexes[i] = d_index_array;
  }

  for (uint i = 0; i < active_gpu_count; i++)
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);

  free(h_lwe_trivial_indexes);
}
/// Load an array residing on one GPU to all active gpus
/// and split the array among them.
/// The input indexing logic is given by an index array.
/// The output indexing is always the trivial one
template <typename Torus>
void multi_gpu_lwe_scatter(cudaStream_t *streams, uint32_t *gpu_indexes,
                           uint32_t gpu_count, std::vector<Torus *> &dest,
                           Torus *src, Torus *d_src_indexes,
                           uint32_t num_inputs, uint32_t elements_per_input) {

  auto active_gpu_count = get_active_gpu_count(num_inputs, gpu_count);

  auto h_src_indexes = (Torus *)malloc(num_inputs * sizeof(Torus));
  cuda_memcpy_async_to_cpu(h_src_indexes, d_src_indexes,
                           num_inputs * sizeof(Torus), streams[0],
                           gpu_indexes[0]);
  cuda_synchronize_stream(streams[0], gpu_indexes[0]);

  dest.resize(active_gpu_count);

#pragma omp parallel for num_threads(active_gpu_count)
  for (uint i = 0; i < active_gpu_count; i++) {
    auto inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
    auto gpu_offset = 0;
    for (uint j = 0; j < i; j++) {
      gpu_offset += get_num_inputs_on_gpu(num_inputs, j, gpu_count);
    }
    auto src_indexes = h_src_indexes + gpu_offset;

    // TODO Check if we can increase parallelization by adding another omp
    // clause here
    for (uint j = 0; j < inputs_on_gpu; j++) {
      auto d_dest = dest[i] + j * elements_per_input;
      auto d_src = src + src_indexes[j] * elements_per_input;

      cuda_memcpy_async_gpu_to_gpu(d_dest, d_src,
                                   elements_per_input * sizeof(Torus),
                                   streams[i], gpu_indexes[i]);
    }
  }

  for (uint i = 0; i < active_gpu_count; i++)
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  free(h_src_indexes);
}

/// Copy data from multiple GPUs back to GPU 0 following the indexing given in
/// dest_indexes
/// The input indexing should be the trivial one
template <typename Torus>
void multi_gpu_lwe_gather(cudaStream_t *streams, uint32_t *gpu_indexes,
                          uint32_t gpu_count, Torus *dest,
                          const std::vector<Torus *> &src,
                          Torus *d_dest_indexes, uint32_t num_inputs,
                          uint32_t elements_per_input) {

  auto active_gpu_count = get_active_gpu_count(num_inputs, gpu_count);

  auto h_dest_indexes = (Torus *)malloc(num_inputs * sizeof(Torus));
  cuda_memcpy_async_to_cpu(h_dest_indexes, d_dest_indexes,
                           num_inputs * sizeof(Torus), streams[0],
                           gpu_indexes[0]);
  cuda_synchronize_stream(streams[0], gpu_indexes[0]);

#pragma omp parallel for num_threads(active_gpu_count)
  for (uint i = 0; i < active_gpu_count; i++) {
    auto inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
    auto gpu_offset = 0;
    for (uint j = 0; j < i; j++) {
      gpu_offset += get_num_inputs_on_gpu(num_inputs, j, gpu_count);
    }
    auto dest_indexes = h_dest_indexes + gpu_offset;

    // TODO Check if we can increase parallelization by adding another omp
    // clause here
    for (uint j = 0; j < inputs_on_gpu; j++) {
      auto d_dest = dest + dest_indexes[j] * elements_per_input;
      auto d_src = src[i] + j * elements_per_input;

      cuda_memcpy_async_gpu_to_gpu(d_dest, d_src,
                                   elements_per_input * sizeof(Torus),
                                   streams[i], gpu_indexes[i]);
    }
  }

  for (uint i = 0; i < active_gpu_count; i++)
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  free(h_dest_indexes);
}

template <typename Torus>
void multi_gpu_lwe_release(cudaStream_t *streams, uint32_t *gpu_indexes,
                           std::vector<Torus *> &vec) {

#pragma omp parallel for num_threads(vec.size())
  for (uint i = 0; i < vec.size(); i++) {
    cuda_drop_async(vec[i], streams[i], gpu_indexes[i]);
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  }
  vec.clear();
}

#endif
