use crate::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::gpu::CudaStreams;
use crate::core_crypto::prelude::LweCiphertextCount;
use crate::integer::ciphertext::DataKind;
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::ciphertext::{
    CudaIntegerRadixCiphertext, CudaRadixCiphertext, CudaSignedRadixCiphertext,
    CudaUnsignedRadixCiphertext,
};
use crate::integer::gpu::list_compression::server_keys::{
    CudaCompressionKey, CudaDecompressionKey,
};
use itertools::Itertools;
use tfhe_cuda_backend::cuda_bind::cuda_memcpy_async_gpu_to_gpu;

pub struct CudaCompressedCiphertextList {
    pub(crate) packed_list: (CudaGlweCiphertextList<u64>, Vec<CudaBlockInfo>),
    info: Vec<DataKind>,
}
impl CudaCompressedCiphertextList {
    pub fn len(&self) -> usize {
        self.info.len()
    }

    pub fn is_empty(&self) -> bool {
        self.info.len() == 0
    }

    pub fn blocks_of(
        &self,
        index: usize,
        decomp_key: &CudaDecompressionKey,
        streams: &CudaStreams,
        gpu_index: u32,
    ) -> Option<(CudaRadixCiphertext, DataKind)> {
        let preceding_infos = self.info.get(..index)?;
        let current_info = self.info.get(index).copied()?;

        let start_block_index: usize = preceding_infos
            .iter()
            .copied()
            .map(DataKind::num_blocks)
            .sum();

        let end_block_index = start_block_index + current_info.num_blocks();

        Some((
            decomp_key.unpack(
                &self.packed_list,
                start_block_index,
                end_block_index,
                streams,
                gpu_index,
            ),
            current_info,
        ))
    }
}

pub trait CudaCompressible {
    fn compress_into(
        self,
        messages: &mut Vec<CudaRadixCiphertext>,
        streams: &CudaStreams,
    ) -> DataKind;
}

// Todo: Can we combine these two impl using CudaIntegerRadixCiphertext?
impl CudaCompressible for CudaSignedRadixCiphertext {
    fn compress_into(
        self,
        messages: &mut Vec<CudaRadixCiphertext>,
        streams: &CudaStreams,
    ) -> DataKind {
        let x = self.ciphertext.duplicate(streams);

        let copy = x.duplicate(streams);
        messages.push(copy);

        let num_blocks = x.d_blocks.lwe_ciphertext_count().0;
        DataKind::Signed(num_blocks)
    }
}
impl CudaCompressible for CudaUnsignedRadixCiphertext {
    fn compress_into(
        self,
        messages: &mut Vec<CudaRadixCiphertext>,
        streams: &CudaStreams,
    ) -> DataKind {
        let x = self.ciphertext.duplicate(streams);

        let copy = x.duplicate(streams);
        messages.push(copy);

        let num_blocks = x.d_blocks.lwe_ciphertext_count().0;

        DataKind::Unsigned(num_blocks)
    }
}

pub struct CompressedCudaCiphertextListBuilder {
    pub(crate) ciphertexts: Vec<CudaRadixCiphertext>,
    pub(crate) info: Vec<DataKind>,
}

impl CompressedCudaCiphertextListBuilder {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            ciphertexts: vec![],
            info: vec![],
        }
    }

    /// ```rust
    /// use tfhe::CompressedCiphertextListBuilder;use tfhe::core_crypto::gpu::CudaStreams;use tfhe::integer::gpu::ciphertext::{CudaSignedRadixCiphertext, CudaUnsignedRadixCiphertext};
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;use tfhe::integer::{IntegerCiphertext, RadixCiphertext, RadixClientKey, SignedRadixCiphertext};
    /// use tfhe::integer::gpu::ciphertext::compressed_ciphertext_list::CompressedCudaCiphertextListBuilder;
    /// use tfhe::shortint::parameters::list_compression::COMP_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M64;
    /// use tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    /// use tfhe::integer::gpu::list_compression::server_keys::*;
    ///
    /// let gpu_index = 0;
    /// let mut streams = CudaStreams::new_single_gpu(gpu_index);
    ///
    /// // Generate the client key and the server key:
    /// let num_blocks = 4;
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut streams);    ///
    ///
    /// let private_compression_key =
    ///       cks.new_compression_private_key(COMP_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M64);
    ///
    /// let (cuda_compression_key, cuda_decompression_key) =
    ///     cks.new_cuda_compression_decompression_keys(&private_compression_key, &streams);
    ///
    /// let ct1 = cks.encrypt(3_u32);
    /// let ct2 = cks.encrypt(2_u32);
    /// let ct3 = cks.encrypt_signed(-2);
    /// // Copy to GPU
    /// let d_ct1 = CudaUnsignedRadixCiphertext::from_radix_ciphertext(&ct1, &mut streams);
    /// let d_ct2 = CudaUnsignedRadixCiphertext::from_radix_ciphertext(&ct2, &mut streams);
    /// let d_ct3 = CudaSignedRadixCiphertext::from_signed_radix_ciphertext(&ct3, &mut streams);
    ///
    /// let compressed = CompressedCudaCiphertextListBuilder::new()
    ///       .push(d_ct1, &streams)
    ///       .push(d_ct2, &streams).push(d_ct3, &streams).
    ///     build(&cuda_compression_key, &streams,0);
    ///
    /// let a = CudaUnsignedRadixCiphertext {ciphertext: compressed.blocks_of(0, &cuda_decompression_key,
    /// &streams, 0).unwrap().0};
    ///
    /// let result = a.to_radix_ciphertext(&streams);
    /// let decrypted: u32 = cks.decrypt(&result);
    /// assert_eq!(decrypted, 3_u32);
    ///
    /// let b = CudaUnsignedRadixCiphertext {ciphertext: compressed.blocks_of(1,
    /// &cuda_decompression_key, &streams, 0).unwrap().0};
    ///
    /// let result = b.to_radix_ciphertext(&streams);
    /// let decrypted: u32 =
    /// cks.decrypt(&result);
    /// assert_eq!(decrypted, 2_u32);
    ///
    /// let c = CudaSignedRadixCiphertext {ciphertext: compressed.blocks_of(2,
    /// &cuda_decompression_key, &streams, 0).unwrap().0};
    ///
    /// let result = c.to_signed_radix_ciphertext(&streams);
    /// let decrypted: i32 =
    /// cks.decrypt_signed(&result);
    /// assert_eq!(decrypted, -2);
    pub fn push<T: CudaCompressible>(&mut self, data: T, streams: &CudaStreams) -> &mut Self {
        let kind = data.compress_into(&mut self.ciphertexts, streams);

        if kind.num_blocks() != 0 {
            self.info.push(kind);
        }

        self
    }

    pub fn build(
        &self,
        comp_key: &CudaCompressionKey,
        streams: &CudaStreams,
        gpu_index: u32,
    ) -> CudaCompressedCiphertextList {
        let packed_list =
            comp_key.compress_ciphertexts_into_list(&self.ciphertexts, streams, gpu_index);
        CudaCompressedCiphertextList {
            packed_list: packed_list,
            info: self.info.clone(),
        }
    }
}
