#include "compression.cuh"

void scratch_cuda_compression_integer_radix_ciphertext_64(
    void **streams, uint32_t *gpu_indexes, uint32_t gpu_count, int8_t **mem_ptr,
    uint32_t glwe_dimension, uint32_t polynomial_size, uint32_t lwe_dimension,
    uint32_t ks_level, uint32_t ks_base_log, uint32_t pbs_level,
    uint32_t pbs_base_log, uint32_t grouping_factor, uint32_t num_lwes,
    uint32_t message_modulus, uint32_t carry_modulus, PBS_TYPE pbs_type,
    uint32_t lwe_per_glwe, uint32_t storage_log_modulus, COMPRESSION_MODE mode,
    bool allocate_gpu_memory) {

  int_radix_params params(pbs_type, glwe_dimension, polynomial_size,
                          glwe_dimension * polynomial_size, lwe_dimension,
                          ks_level, ks_base_log, pbs_level, pbs_base_log,
                          grouping_factor, message_modulus, carry_modulus);

  scratch_cuda_compression_integer_radix_ciphertext_64(
      (cudaStream_t *)(streams), gpu_indexes, gpu_count,
      (int_compression<uint64_t> **)mem_ptr, num_lwes, params, lwe_per_glwe,
      storage_log_modulus, mode, allocate_gpu_memory);
}
void cuda_compression_compress_integer_radix_ciphertext_64(
    void **streams, uint32_t *gpu_indexes, uint32_t gpu_count,
    void *glwe_array_out, void *lwe_array_in, void **fp_ksk, uint32_t num_lwes,
    int8_t *mem_ptr) {

  host_integer_compression_compress<uint64_t>(
      (cudaStream_t *)(streams), gpu_indexes, gpu_count,
      static_cast<uint64_t *>(glwe_array_out),
      static_cast<uint64_t *>(lwe_array_in), (uint64_t **)(fp_ksk), num_lwes,
      (int_compression<uint64_t> *)mem_ptr);
}
void cuda_compression_decompress_integer_radix_ciphertext_64(
    void **streams, uint32_t *gpu_indexes, uint32_t gpu_count, void *lwe_out,
    void *glwe_array_in, void **bsks, int8_t *mem_ptr) {

  host_integer_compression_decompress<uint64_t>(
      (cudaStream_t *)(streams), gpu_indexes, gpu_count,
      static_cast<uint64_t *>(lwe_out), static_cast<uint64_t *>(glwe_array_in),
      bsks, (int_compression<uint64_t> *)mem_ptr);
}

void cleanup_cuda_compression_integer_radix_ciphertext_64(
    void **streams, uint32_t *gpu_indexes, uint32_t gpu_count,
    int8_t **mem_ptr_void) {

  int_compression<uint64_t> *mem_ptr =
      (int_compression<uint64_t> *)(*mem_ptr_void);
  mem_ptr->release((cudaStream_t *)(streams), gpu_indexes, gpu_count);
}
