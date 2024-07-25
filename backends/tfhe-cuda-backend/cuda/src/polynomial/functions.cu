#include "functions.h"

void cuda_glwe_sample_extract_64(void **streams, uint32_t *gpu_indexes,
                                 uint32_t gpu_count, void *lwe_array_out,
                                 void *glwe_in, uint32_t *nth_array,
                                 uint32_t num_samples, uint32_t glwe_dimension,
                                 uint32_t polynomial_size) {

  switch (polynomial_size) {
  case 256:
    host_sample_extract<uint64_t, AmortizedDegree<256>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 512:
    host_sample_extract<uint64_t, AmortizedDegree<512>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 1024:
    host_sample_extract<uint64_t, AmortizedDegree<1024>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 2048:
    host_sample_extract<uint64_t, AmortizedDegree<2048>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 4096:
    host_sample_extract<uint64_t, AmortizedDegree<4096>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 8192:
    host_sample_extract<uint64_t, AmortizedDegree<8192>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  case 16384:
    host_sample_extract<uint64_t, AmortizedDegree<16384>>(
        (cudaStream_t *)(streams), (uint64_t *)lwe_array_out,
        (uint64_t *)glwe_in, (uint32_t *)nth_array, num_samples,
        glwe_dimension);
    break;
  default:
    PANIC("Cuda error: unsupported polynomial size. Supported "
          "N's are powers of two in the interval [256..16384].")
  }
}
