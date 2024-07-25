use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::gpu::{CudaLweList, CudaStreams};
use crate::core_crypto::prelude::{
    CiphertextModulus, Container, LweCiphertext, LweCiphertextCount, LweCiphertextList,
    LweDimension, LweSize, UnsignedInteger,
};

/// A structure representing a vector of LWE ciphertexts with 64 bits of precision on the GPU.
#[derive(Debug)]
pub struct CudaLweCiphertext<T: UnsignedInteger>(pub(crate) CudaLweList<T>);

#[allow(dead_code)]
impl<T: UnsignedInteger> CudaLweCiphertext<T> {
    pub fn new(
        lwe_dimension: LweDimension,
        ciphertext_modulus: CiphertextModulus<T>,
        streams: &CudaStreams,
    ) -> Self {
        // Allocate memory in the gpu_index
        let d_vec = unsafe { CudaVec::new_async(lwe_dimension.to_lwe_size().0, streams, 0) };
        streams.synchronize();

        let cuda_lwe_list = CudaLweList {
            d_vec,
            lwe_ciphertext_count: LweCiphertextCount(1),
            lwe_dimension,
            ciphertext_modulus,
        };

        Self(cuda_lwe_list)
    }

    pub fn from_cuda_vec(d_vec: CudaVec<T>, ciphertext_modulus: CiphertextModulus<T>) -> Self {
        let lwe_dimension = LweSize(d_vec.len()).to_lwe_dimension();
        let cuda_lwe_list = CudaLweList {
            d_vec,
            lwe_ciphertext_count: LweCiphertextCount(1),
            lwe_dimension,
            ciphertext_modulus,
        };
        Self(cuda_lwe_list)
    }

    pub fn to_lwe_ciphertext_list(&self, streams: &CudaStreams) -> LweCiphertextList<Vec<T>> {
        let lwe_ct_size = self.0.lwe_dimension.to_lwe_size().0;
        let mut container: Vec<T> = vec![T::ZERO; lwe_ct_size];

        unsafe {
            self.0
                .d_vec
                .copy_to_cpu_async(container.as_mut_slice(), streams, 0);
        }
        streams.synchronize();

        LweCiphertextList::from_container(
            container,
            self.lwe_dimension().to_lwe_size(),
            self.ciphertext_modulus(),
        )
    }

    pub fn from_lwe_ciphertext<C: Container<Element = T>>(
        h_ct: &LweCiphertext<C>,
        streams: &CudaStreams,
    ) -> Self {
        let lwe_dimension = h_ct.lwe_size().to_lwe_dimension();
        let lwe_ciphertext_count = LweCiphertextCount(1);
        let ciphertext_modulus = h_ct.ciphertext_modulus();

        // Copy to the GPU
        let mut d_vec = CudaVec::new(lwe_dimension.to_lwe_size().0, streams, 0);
        unsafe {
            d_vec.copy_from_cpu_async(h_ct.as_ref(), streams, 0);
        }
        streams.synchronize();

        let cuda_lwe_list = CudaLweList {
            d_vec,
            lwe_ciphertext_count,
            lwe_dimension,
            ciphertext_modulus,
        };
        Self(cuda_lwe_list)
    }

    pub fn into_lwe_ciphertext(&self, streams: &CudaStreams) -> LweCiphertext<Vec<T>> {
        let lwe_ct_size = self.0.lwe_dimension.to_lwe_size().0;
        let mut container: Vec<T> = vec![T::ZERO; lwe_ct_size];

        unsafe {
            self.0
                .d_vec
                .copy_to_cpu_async(container.as_mut_slice(), streams, 0);
        }
        streams.synchronize();

        LweCiphertext::from_container(container, self.ciphertext_modulus())
    }

    /// ```rust
    /// use tfhe::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
    /// use tfhe::core_crypto::gpu::CudaStreams;
    /// use tfhe::core_crypto::prelude::{
    ///     CiphertextModulus, LweCiphertextCount, LweCiphertextList, LweSize,
    /// };
    ///
    /// let mut streams = CudaStreams::new_single_gpu(0);
    ///
    /// let lwe_size = LweSize(743);
    /// let ciphertext_modulus = CiphertextModulus::new_native();
    /// let lwe_ciphertext_count = LweCiphertextCount(2);
    ///
    /// // Create a new LweCiphertextList
    /// let lwe_list = LweCiphertextList::new(0u64, lwe_size, lwe_ciphertext_count, ciphertext_modulus);
    ///
    /// // Copy to GPU
    /// let d_lwe_list = CudaLweCiphertextList::from_lwe_ciphertext_list(&lwe_list, &mut streams);
    /// let d_lwe_list_copied = d_lwe_list.duplicate(&mut streams);
    ///
    /// let lwe_list_copied = d_lwe_list_copied.to_lwe_ciphertext_list(&mut streams);
    ///
    /// assert_eq!(lwe_list, lwe_list_copied);
    /// ```
    pub fn duplicate(&self, streams: &CudaStreams) -> Self {
        let lwe_dimension = self.lwe_dimension();
        let ciphertext_modulus = self.ciphertext_modulus();

        // Copy to the GPU
        let d_vec = unsafe { self.0.d_vec.duplicate(streams, 0) };

        let cuda_lwe_list = CudaLweList {
            d_vec,
            lwe_ciphertext_count: LweCiphertextCount(1),
            lwe_dimension,
            ciphertext_modulus,
        };
        Self(cuda_lwe_list)
    }

    pub(crate) fn lwe_dimension(&self) -> LweDimension {
        self.0.lwe_dimension
    }

    pub(crate) fn ciphertext_modulus(&self) -> CiphertextModulus<T> {
        self.0.ciphertext_modulus
    }
}
