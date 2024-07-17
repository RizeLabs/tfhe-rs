use crate::core_crypto::entities::LweCiphertext;
use crate::core_crypto::gpu::entities::lwe_packing_keyswitch_key::CudaLwePackingKeyswitchKey;
use crate::core_crypto::gpu::glwe_ciphertext_list::CudaGlweCiphertextList;
use crate::core_crypto::gpu::lwe_bootstrap_key::CudaLweBootstrapKey;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::gpu::CudaStreams;
use crate::core_crypto::prelude::{
    allocate_and_generate_new_lwe_packing_keyswitch_key,
    par_allocate_and_generate_new_lwe_bootstrap_key, CiphertextModulusLog, GlweCiphertextCount,
    LweBootstrapKeyOwned, LweCiphertextCount, LweCiphertextList, LweSize,
};
use crate::integer::ciphertext::DataKind;
use crate::integer::gpu::ciphertext::compressed_ciphertext_list::CudaCompressedCiphertextList;
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::ciphertext::CudaRadixCiphertext;
use crate::integer::gpu::server_key::CudaBootstrappingKey;
use crate::integer::gpu::{
    cuda_memcpy_async_gpu_to_gpu, unchecked_compression_compress_integer_radix_async, PBSType,
};
use crate::shortint::client_key::ClientKey;
use crate::shortint::engine::ShortintEngine;
use crate::shortint::list_compression::{CompressionKey, CompressionPrivateKeys};
use crate::shortint::{
    CiphertextModulus, ClassicPBSParameters, EncryptionKeyChoice, PBSParameters,
};
use itertools::Itertools;

#[derive(Debug)]
pub struct CudaCompressionKey {
    pub packing_key_switching_key: CudaLwePackingKeyswitchKey<u64>,
    pub lwe_per_glwe: LweCiphertextCount,
    pub storage_log_modulus: CiphertextModulusLog,
}

pub struct CudaDecompressionKey {
    pub blind_rotate_key: CudaBootstrappingKey,
    pub lwe_per_glwe: LweCiphertextCount,
    pub ciphertext_modulus: CiphertextModulus, //TODO: Remove this
}

impl CudaCompressionKey {
    pub fn from_compression_key(compression_key: &CompressionKey, streams: &CudaStreams) -> Self {
        CudaCompressionKey {
            packing_key_switching_key: CudaLwePackingKeyswitchKey::from_lwe_packing_keyswitch_key(
                &compression_key.packing_key_switching_key,
                streams,
            ),
            lwe_per_glwe: compression_key.lwe_per_glwe,
            storage_log_modulus: compression_key.storage_log_modulus,
        }
    }

    fn flatten(
        &self,
        vec_ciphertexts: &Vec<CudaRadixCiphertext>,
        streams: &CudaStreams,
        gpu_index: u32,
    ) -> (CudaLweCiphertextList<u64>, Vec<CudaBlockInfo>) {
        let first_ct = &vec_ciphertexts.first().unwrap().d_blocks;

        // We assume all ciphertexts will have the same lwe dimension
        let lwe_dimension = first_ct.lwe_dimension();
        let ciphertext_modulus = first_ct.ciphertext_modulus();

        // Compute total number of lwe ciphertexts we will be handling
        let total_num_blocks: usize = vec_ciphertexts
            .iter()
            .map(|x| x.d_blocks.lwe_ciphertext_count().0)
            .sum();

        let lwe_ciphertext_count = LweCiphertextCount(total_num_blocks);

        let d_vec = unsafe {
            let mut d_vec = CudaVec::new_async(
                lwe_dimension.to_lwe_size().0 * lwe_ciphertext_count.0,
                streams,
                gpu_index,
            );
            let mut offset: usize = 0;
            for ciphertext in vec_ciphertexts {
                // Todo: We might use copy_self_range_gpu_to_gpu_async here
                let dest_ptr = d_vec
                    .as_mut_c_ptr(gpu_index)
                    .add(offset * std::mem::size_of::<u64>());
                let size = ciphertext.d_blocks.0.d_vec.len * std::mem::size_of::<u64>();
                cuda_memcpy_async_gpu_to_gpu(
                    dest_ptr,
                    ciphertext.d_blocks.0.d_vec.as_c_ptr(gpu_index),
                    size as u64,
                    streams.ptr[gpu_index as usize],
                    streams.gpu_indexes[gpu_index as usize],
                );

                offset += ciphertext.d_blocks.0.d_vec.len;
            }

            streams.synchronize();
            d_vec
        };

        let flattened_ciphertexts =
            CudaLweCiphertextList::from_cuda_vec(d_vec, lwe_ciphertext_count, ciphertext_modulus);

        let info = vec_ciphertexts
            .iter()
            .flat_map(|x| x.info.blocks.clone())
            .collect_vec();

        (flattened_ciphertexts, info)
    }

    pub fn compress_ciphertexts_into_list(
        &self,
        ciphertexts: &Vec<CudaRadixCiphertext>,
        streams: &CudaStreams,
        gpu_index: u32,
    ) -> (CudaGlweCiphertextList<u64>, Vec<CudaBlockInfo>) {
        let lwe_pksk = &self.packing_key_switching_key;

        let polynomial_size = lwe_pksk.output_polynomial_size();
        let ciphertext_modulus = lwe_pksk.ciphertext_modulus();
        let glwe_size = lwe_pksk.output_glwe_size();
        let lwe_size = lwe_pksk.input_key_lwe_dimension().to_lwe_size();
        println!("compress lwe_size: {:?}", lwe_size);
        println!("compress polynomial_size: {:?}", polynomial_size);

        let first_ct_info = ciphertexts.first().unwrap().info.blocks.first().unwrap();
        let message_modulus = first_ct_info.message_modulus;
        let carry_modulus = first_ct_info.carry_modulus;

        let num_lwes: usize = ciphertexts
            .iter()
            .map(|x| x.d_blocks.lwe_ciphertext_count().0)
            .sum();

        let mut output_glwe = CudaGlweCiphertextList::new(
            glwe_size.to_glwe_dimension(),
            polynomial_size,
            GlweCiphertextCount(ciphertexts.len()),
            ciphertext_modulus,
            streams,
        );

        let (input_lwes, info) = self.flatten(ciphertexts, streams, gpu_index);

        unsafe {
            unchecked_compression_compress_integer_radix_async(
                streams,
                &mut output_glwe.0.d_vec,
                &input_lwes.0.d_vec,
                &self.packing_key_switching_key.d_vec,
                message_modulus,
                carry_modulus,
                glwe_size.to_glwe_dimension(),
                polynomial_size,
                lwe_size.to_lwe_dimension(),
                lwe_pksk.decomposition_base_log(),
                lwe_pksk.decomposition_level_count(),
                self.lwe_per_glwe.0 as u32,
                self.storage_log_modulus.0 as u32,
                num_lwes as u32,
            );
        }

        (output_glwe, info)
    }
}

impl CudaDecompressionKey {
    pub fn unpack(
        &self,
        packed_list: &(CudaGlweCiphertextList<u64>, Vec<CudaBlockInfo>),
        start_block_index: usize,
        end_block_index: usize,
        streams: &CudaStreams,
        gpu_index: u32,
    ) -> CudaRadixCiphertext {
        let glwe_dimension = packed_list.0.glwe_dimension();
        let polynomial_size = packed_list.0.polynomial_size();
        let lwe_ciphertext_count = LweCiphertextCount(end_block_index - start_block_index);
        //let lwe_size = self.blind_rotate_key.output_lwe_dimension().to_lwe_size();
        let lwe_size = LweSize(0);
        println!("decompress lwe_size: {:?}", lwe_size);
        println!(
            "decompress lwe_ciphertext_count: {:?}",
            lwe_ciphertext_count
        );
        println!("decompress polynomial_size: {:?}", polynomial_size);
        let output_lwe = CudaLweCiphertextList::new(
            lwe_size.to_lwe_dimension(),
            lwe_ciphertext_count,
            self.ciphertext_modulus,
            streams,
        );
        unsafe {
            // unchecked_compression_compress_integer_radix_async(streams,
            //                                                    &mut output_glwe.0.d_vec,
            //                                                    &input_lwes.0.d_vec,
            //
            // &self.packing_key_switching_key.d_vec,
            // message_modulus,
            // carry_modulus,
            // glwe_size.to_glwe_dimension(),
            // polynomial_size,
            // lwe_size.to_lwe_dimension(),
            // lwe_pksk.decomposition_base_log(),
            // lwe_pksk.decomposition_level_count(),
            // self.lwe_per_glwe.0 as u32,
            // self.storage_log_modulus.0 as u32,
            // num_lwes as u32, );
        }

        CudaRadixCiphertext {
            d_blocks: output_lwe,
            info: CudaRadixCiphertextInfo {
                blocks: packed_list.1.clone(),
            },
        }
    }
}

impl ClientKey {
    pub fn new_cuda_compression_decompression_keys(
        &self,
        private_compression_key: &CompressionPrivateKeys,
        streams: &CudaStreams,
    ) -> (CudaCompressionKey, CudaDecompressionKey) {
        let params = &private_compression_key.params;
        let cks_params: ClassicPBSParameters = match self.parameters.pbs_parameters().unwrap() {
            PBSParameters::PBS(a) => a,
            PBSParameters::MultiBitPBS(_) => {
                panic!("Compression is currently not compatible with Multi Bit PBS")
            }
        };

        assert_eq!(
            cks_params.encryption_key_choice,
            EncryptionKeyChoice::Big,
            "Compression is only compatible with ciphertext in post PBS dimension"
        );

        let packing_key_switching_key = ShortintEngine::with_thread_local_mut(|engine| {
            allocate_and_generate_new_lwe_packing_keyswitch_key(
                &self.large_lwe_secret_key(),
                &private_compression_key.post_packing_ks_key,
                params.packing_ks_base_log,
                params.packing_ks_level,
                params.packing_ks_key_noise_distribution,
                self.parameters.ciphertext_modulus(),
                &mut engine.encryption_generator,
            )
        });

        assert!(
            private_compression_key.params.storage_log_modulus.0
                <= cks_params
                    .polynomial_size
                    .to_blind_rotation_input_modulus_log()
                    .0,
            "Compression parameters say to store more bits than useful"
        );

        let glwe_compression_key = CompressionKey {
            packing_key_switching_key,
            lwe_per_glwe: params.lwe_per_glwe,
            storage_log_modulus: private_compression_key.params.storage_log_modulus,
        };

        let mut engine = ShortintEngine::new();
        let h_bootstrap_key: LweBootstrapKeyOwned<u64> =
            par_allocate_and_generate_new_lwe_bootstrap_key(
                &private_compression_key
                    .post_packing_ks_key
                    .as_lwe_secret_key(),
                &self.glwe_secret_key,
                private_compression_key.params.br_base_log,
                private_compression_key.params.br_level,
                self.parameters.glwe_noise_distribution(),
                self.parameters.ciphertext_modulus(),
                &mut engine.encryption_generator,
            );

        let d_bootstrap_key =
            CudaLweBootstrapKey::from_lwe_bootstrap_key(&h_bootstrap_key, streams);

        let blind_rotate_key = CudaBootstrappingKey::Classic(d_bootstrap_key);

        let cuda_glwe_decompression_key = CudaDecompressionKey {
            blind_rotate_key,
            lwe_per_glwe: params.lwe_per_glwe,
            ciphertext_modulus: self.parameters.ciphertext_modulus(),
        };

        (
            CudaCompressionKey::from_compression_key(&glwe_compression_key, streams),
            cuda_glwe_decompression_key,
        )
    }
}
