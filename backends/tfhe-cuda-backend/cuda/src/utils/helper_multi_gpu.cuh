#ifndef HELPER_MULTI_GPU_CUH
#define HELPER_MULTI_GPU_CUH

#include "helper_multi_gpu.h"

/// Load an array residing on one GPU to all active gpus
/// and split the array among them.
/// The indexing logic is given by an index array.
template <typename Torus>
void multi_gpu_scatter(cudaStream_t *streams, uint32_t *gpu_indexes,
                       uint32_t gpu_count, std::vector<Torus *> &dest,
                       Torus *src, std::vector<Torus *> &dest_indexes,
                       Torus *src_indexes, uint32_t num_inputs,
                       uint32_t elements_per_input) {

  auto active_gpu_count = get_active_gpu_count(num_inputs, gpu_count);

  auto cpu_indexes = (Torus *)malloc(num_inputs * sizeof(Torus));
  cuda_memcpy_async_to_cpu(cpu_indexes, src_indexes, num_inputs * sizeof(Torus),
                           streams[0], gpu_indexes[0]);
  cuda_synchronize_stream(streams[0], gpu_indexes[0]);

  // TODO move allocation/drop to scratch/cleanup
  for (uint i = 0; i < active_gpu_count; i++) {
    auto inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
    Torus *array = (Torus *)cuda_malloc_async(
        inputs_on_gpu * elements_per_input * sizeof(Torus), streams[i],
        gpu_indexes[i]);
    Torus *index_array = (Torus *)cuda_malloc_async(
        inputs_on_gpu * sizeof(Torus), streams[i], gpu_indexes[i]);
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
    dest.push_back(array);
    dest_indexes.push_back(index_array);
  }

#pragma omp parallel for num_threads(num_inputs)
  for (uint j = 0; j < num_inputs; j++) {
    int gpu_index = 0;
    Torus index_on_gpu = 0;
    Torus accumulated_inputs = 0;
    for (uint i = 0; i < active_gpu_count; i++) {
      int inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
      if (j < accumulated_inputs + inputs_on_gpu) {
        gpu_index = i;
        index_on_gpu = j - accumulated_inputs;
        printf("input j: %d, gpu_index: %d, index on gpu: %d\n", j, gpu_indexes,
               index_on_gpu);
      }
      accumulated_inputs += inputs_on_gpu;
    }
    cuda_memcpy_async_gpu_to_gpu(dest[gpu_index] +
                                     index_on_gpu * elements_per_input,
                                 src + cpu_indexes[j] * elements_per_input,
                                 elements_per_input * sizeof(Torus),
                                 streams[gpu_index], gpu_indexes[gpu_index]);
    cuda_memset_async(dest_indexes[gpu_index] + index_on_gpu, index_on_gpu,
                      sizeof(Torus), streams[gpu_index],
                      gpu_indexes[gpu_index]);
  }
  for (uint i = 0; i < active_gpu_count; i++) {
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  }
  free(cpu_indexes);
}

/// Copy data from multiple GPUs back to GPU 0 following the indexing given in
/// dest_indexes
template <typename Torus>
void multi_gpu_gather(cudaStream_t *streams, uint32_t *gpu_indexes,
                      uint32_t gpu_count, Torus *dest,
                      const std::vector<Torus *> &src, Torus *dest_indexes,
                      uint32_t num_inputs, uint32_t elements_per_input) {

  auto active_gpu_count = get_active_gpu_count(num_inputs, gpu_count);

  auto dest_cpu_indexes = (Torus *)malloc(num_inputs * sizeof(Torus));
  cuda_memcpy_async_to_cpu(dest_cpu_indexes, dest_indexes,
                           num_inputs * sizeof(Torus), streams[0],
                           gpu_indexes[0]);
  cuda_synchronize_stream(streams[0], gpu_indexes[0]);

#pragma omp parallel for num_threads(num_inputs)
  for (uint j = 0; j < num_inputs; j++) {
    int gpu_index = 0;
    Torus index_on_gpu = 0;
    Torus accumulated_inputs = 0;
    for (uint i = 0; i < active_gpu_count; i++) {
      int inputs_on_gpu = get_num_inputs_on_gpu(num_inputs, i, gpu_count);
      if (j < accumulated_inputs + inputs_on_gpu) {
        gpu_index = i;
        index_on_gpu = j - accumulated_inputs;
        break;
      }
      accumulated_inputs += inputs_on_gpu;
    }
    cuda_memcpy_async_gpu_to_gpu(
        dest + dest_cpu_indexes[j] * elements_per_input,
        src[gpu_index] + index_on_gpu * elements_per_input,
        elements_per_input * sizeof(Torus), streams[gpu_index],
        gpu_indexes[gpu_index]);
  }
  for (uint i = 0; i < active_gpu_count; i++) {
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  }
  free(dest_cpu_indexes);
}

template <typename Torus>
void multi_gpu_release(cudaStream_t *streams, uint32_t *gpu_indexes,
                       std::vector<Torus *> &vec) {

#pragma omp parallel for num_threads(vec.size())
  for (uint i = 0; i < vec.size(); i++) {
    cuda_drop_async(vec[i], streams[i], gpu_indexes[i]);
    cuda_synchronize_stream(streams[i], gpu_indexes[i]);
  }
}

#endif
