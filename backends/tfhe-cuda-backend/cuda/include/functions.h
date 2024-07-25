#ifndef CUDA_FUNCTIONS_H_
#define CUDA_FUNCTIONS_H_

#include "polynomial/functions.cuh"
#include "polynomial/parameters.cuh"
#include <cstdint>

extern "C" {
void cuda_glwe_sample_extract_64(void **streams, uint32_t *gpu_indexes,
                                 uint32_t gpu_count, void *lwe_array_out,
                                 void *glwe_array_in, uint32_t *nth_array,
                                 uint32_t num_samples, uint32_t glwe_dimension,
                                 uint32_t polynomial_size);
}

#endif
