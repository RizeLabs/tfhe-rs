use crate::core_crypto::gpu::lwe_bootstrap_key::CudaLweBootstrapKey;
use crate::core_crypto::gpu::lwe_ciphertext_list::CudaLweCiphertextList;
use crate::core_crypto::gpu::lwe_keyswitch_key::CudaLweKeyswitchKey;
use crate::core_crypto::gpu::lwe_multi_bit_bootstrap_key::CudaLweMultiBitBootstrapKey;
use crate::core_crypto::gpu::CudaStream;
use crate::core_crypto::prelude::{
    allocate_and_generate_new_lwe_keyswitch_key, par_allocate_and_generate_new_lwe_bootstrap_key,
    par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key, ContiguousEntityContainerMut,
    LweBootstrapKeyOwned, LweCiphertextCount, LweCiphertextList, LweMultiBitBootstrapKeyOwned,
};
use crate::integer::block_decomposition::{BlockDecomposer, DecomposableInto};
use crate::integer::gpu::ciphertext::info::{CudaBlockInfo, CudaRadixCiphertextInfo};
use crate::integer::gpu::ciphertext::{CudaIntegerRadixCiphertext, CudaRadixCiphertext};
use crate::integer::ClientKey;
use crate::shortint::ciphertext::{Degree, MaxDegree, NoiseLevel};
use crate::shortint::engine::ShortintEngine;
use crate::shortint::{CarryModulus, CiphertextModulus, MessageModulus, PBSOrder};

mod radix;

pub enum CudaBootstrappingKey {
    Classic(CudaLweBootstrapKey),
    MultiBit(CudaLweMultiBitBootstrapKey),
}

/// A structure containing the server public key.
///
/// The server key is generated by the client and is meant to be published: the client
/// sends it to the server so it can compute homomorphic circuits.
// #[derive(PartialEq, Serialize, Deserialize)]
pub struct CudaServerKey {
    pub key_switching_key: CudaLweKeyswitchKey<u64>,
    pub bootstrapping_key: CudaBootstrappingKey,
    // Size of the message buffer
    pub message_modulus: MessageModulus,
    // Size of the carry buffer
    pub carry_modulus: CarryModulus,
    // Maximum number of operations that can be done before emptying the operation buffer
    pub max_degree: MaxDegree,
    // Modulus use for computations on the ciphertext
    pub ciphertext_modulus: CiphertextModulus,
    pub pbs_order: PBSOrder,
}

impl CudaServerKey {
    /// Generates a server key that stores keys in the device memory.
    ///
    /// # Example
    ///
    /// ```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::CudaServerKey;
    /// use tfhe::integer::ClientKey;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key:
    /// let cks = ClientKey::new(PARAM_MESSAGE_2_CARRY_2_KS_PBS);
    ///
    /// // Generate the server key:
    /// let sks = CudaServerKey::new(&cks, &mut stream);
    /// ```
    pub fn new<C>(cks: C, stream: &CudaStream) -> Self
    where
        C: AsRef<ClientKey>,
    {
        // It should remain just enough space to add a carry
        let client_key = cks.as_ref();
        let max_degree = MaxDegree::integer_radix_server_key(
            client_key.key.parameters.message_modulus(),
            client_key.key.parameters.carry_modulus(),
        );
        Self::new_server_key_with_max_degree(client_key, max_degree, stream)
    }

    pub(crate) fn new_server_key_with_max_degree(
        cks: &ClientKey,
        max_degree: MaxDegree,
        stream: &CudaStream,
    ) -> Self {
        let mut engine = ShortintEngine::new();

        // Generate a regular keyset and convert to the GPU
        let pbs_params_base = &cks.parameters();
        let d_bootstrapping_key = match pbs_params_base {
            crate::shortint::PBSParameters::PBS(pbs_params) => {
                let h_bootstrap_key: LweBootstrapKeyOwned<u64> =
                    par_allocate_and_generate_new_lwe_bootstrap_key(
                        &cks.key.small_lwe_secret_key(),
                        &cks.key.glwe_secret_key,
                        pbs_params.pbs_base_log,
                        pbs_params.pbs_level,
                        pbs_params.glwe_modular_std_dev,
                        pbs_params.ciphertext_modulus,
                        &mut engine.encryption_generator,
                    );

                let d_bootstrap_key =
                    CudaLweBootstrapKey::from_lwe_bootstrap_key(&h_bootstrap_key, stream);

                CudaBootstrappingKey::Classic(d_bootstrap_key)
            }
            crate::shortint::PBSParameters::MultiBitPBS(pbs_params) => {
                let h_bootstrap_key: LweMultiBitBootstrapKeyOwned<u64> =
                    par_allocate_and_generate_new_lwe_multi_bit_bootstrap_key(
                        &cks.key.small_lwe_secret_key(),
                        &cks.key.glwe_secret_key,
                        pbs_params.pbs_base_log,
                        pbs_params.pbs_level,
                        pbs_params.grouping_factor,
                        pbs_params.glwe_modular_std_dev,
                        pbs_params.ciphertext_modulus,
                        &mut engine.encryption_generator,
                    );

                let d_bootstrap_key = CudaLweMultiBitBootstrapKey::from_lwe_multi_bit_bootstrap_key(
                    &h_bootstrap_key,
                    stream,
                );

                CudaBootstrappingKey::MultiBit(d_bootstrap_key)
            }
        };

        // Creation of the key switching key
        let h_key_switching_key = allocate_and_generate_new_lwe_keyswitch_key(
            &cks.key.large_lwe_secret_key(),
            &cks.key.small_lwe_secret_key(),
            cks.parameters().ks_base_log(),
            cks.parameters().ks_level(),
            cks.parameters().lwe_modular_std_dev(),
            cks.parameters().ciphertext_modulus(),
            &mut engine.encryption_generator,
        );

        let d_key_switching_key =
            CudaLweKeyswitchKey::from_lwe_keyswitch_key(&h_key_switching_key, stream);

        assert!(matches!(
            cks.parameters().encryption_key_choice().into(),
            PBSOrder::KeyswitchBootstrap
        ));

        // Pack the keys in the server key set:
        Self {
            key_switching_key: d_key_switching_key,
            bootstrapping_key: d_bootstrapping_key,
            message_modulus: cks.parameters().message_modulus(),
            carry_modulus: cks.parameters().carry_modulus(),
            max_degree,
            ciphertext_modulus: cks.parameters().ciphertext_modulus(),
            pbs_order: cks.parameters().encryption_key_choice().into(),
        }
    }

    /// Create a trivial ciphertext filled with zeros
    ///
    /// # Example
    ///
    /// ```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::{gen_keys_radix, RadixCiphertext};
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// let num_blocks = 4;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let d_ctxt: CudaRadixCiphertext = sks.create_trivial_zero_radix(num_blocks, &mut stream);
    /// let ctxt = d_ctxt.to_radix_ciphertext(&mut stream);
    ///
    /// // Decrypt:
    /// let dec: u64 = cks.decrypt(&ctxt);
    /// assert_eq!(0, dec);
    /// ```
    pub fn create_trivial_zero_radix(
        &self,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        self.create_trivial_radix(0, num_blocks, stream)
    }

    /// Create a trivial ciphertext
    ///
    /// # Example
    ///
    /// ```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::{gen_keys_radix, RadixCiphertext};
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// let num_blocks = 4;
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let d_ctxt: CudaRadixCiphertext = sks.create_trivial_radix(212u64, num_blocks, &mut stream);
    /// let ctxt = d_ctxt.to_radix_ciphertext(&mut stream);
    ///
    /// // Decrypt:
    /// let dec: u64 = cks.decrypt(&ctxt);
    /// assert_eq!(212, dec);
    /// ```
    pub fn create_trivial_radix<T>(
        &self,
        scalar: T,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext
    where
        T: DecomposableInto<u64>,
    {
        let lwe_size = match self.pbs_order {
            PBSOrder::KeyswitchBootstrap => self.key_switching_key.input_key_lwe_size(),
            PBSOrder::BootstrapKeyswitch => self.key_switching_key.output_key_lwe_size(),
        };

        let delta = (1_u64 << 63) / (self.message_modulus.0 * self.carry_modulus.0) as u64;

        let decomposer = BlockDecomposer::new(scalar, self.message_modulus.0.ilog2())
            .iter_as::<u64>()
            .chain(std::iter::repeat(0))
            .take(num_blocks);
        let mut cpu_lwe_list = LweCiphertextList::new(
            0,
            lwe_size,
            LweCiphertextCount(num_blocks),
            self.ciphertext_modulus,
        );
        let mut info = Vec::with_capacity(num_blocks);
        for (block_value, mut lwe) in decomposer.zip(cpu_lwe_list.iter_mut()) {
            *lwe.get_mut_body().data = block_value * delta;
            info.push(CudaBlockInfo {
                degree: Degree::new(block_value as usize),
                message_modulus: self.message_modulus,
                carry_modulus: self.carry_modulus,
                pbs_order: self.pbs_order,
                noise_level: NoiseLevel::ZERO,
            });
        }

        let d_blocks = CudaLweCiphertextList::from_lwe_ciphertext_list(&cpu_lwe_list, stream);

        CudaRadixCiphertext {
            d_blocks,
            info: CudaRadixCiphertextInfo { blocks: info },
        }
    }

    /// # Safety
    ///
    /// - `stream` __must__ be synchronized to guarantee computation has finished, and inputs must
    ///   not be dropped until stream is synchronized
    pub(crate) unsafe fn propagate_single_carry_assign_async(
        &self,
        ct: &mut CudaRadixCiphertext,
        stream: &CudaStream,
    ) {
        let num_blocks = ct.d_blocks.lwe_ciphertext_count().0 as u32;
        match &self.bootstrapping_key {
            CudaBootstrappingKey::Classic(d_bsk) => {
                stream.propagate_single_carry_classic_assign_async(
                    &mut ct.d_blocks.0.d_vec,
                    &d_bsk.d_vec,
                    &self.key_switching_key.d_vec,
                    d_bsk.input_lwe_dimension(),
                    d_bsk.glwe_dimension(),
                    d_bsk.polynomial_size(),
                    self.key_switching_key.decomposition_level_count(),
                    self.key_switching_key.decomposition_base_log(),
                    d_bsk.decomp_level_count(),
                    d_bsk.decomp_base_log(),
                    num_blocks,
                    ct.info.blocks.first().unwrap().message_modulus,
                    ct.info.blocks.first().unwrap().carry_modulus,
                );
            }
            CudaBootstrappingKey::MultiBit(d_multibit_bsk) => {
                stream.propagate_single_carry_multibit_assign_async(
                    &mut ct.d_blocks.0.d_vec,
                    &d_multibit_bsk.d_vec,
                    &self.key_switching_key.d_vec,
                    d_multibit_bsk.input_lwe_dimension(),
                    d_multibit_bsk.glwe_dimension(),
                    d_multibit_bsk.polynomial_size(),
                    self.key_switching_key.decomposition_level_count(),
                    self.key_switching_key.decomposition_base_log(),
                    d_multibit_bsk.decomp_level_count(),
                    d_multibit_bsk.decomp_base_log(),
                    d_multibit_bsk.grouping_factor,
                    num_blocks,
                    ct.info.blocks.first().unwrap().message_modulus,
                    ct.info.blocks.first().unwrap().carry_modulus,
                );
            }
        };
        ct.info
            .blocks
            .iter_mut()
            .for_each(|b| b.degree = Degree::new(b.message_modulus.0 - 1));
    }

    /// # Safety
    ///
    /// - `stream` __must__ be synchronized to guarantee computation has finished, and inputs must
    ///   not be dropped until stream is synchronized
    pub(crate) unsafe fn full_propagate_assign_async(
        &self,
        ct: &mut CudaRadixCiphertext,
        stream: &CudaStream,
    ) {
        let num_blocks = ct.d_blocks.lwe_ciphertext_count().0 as u32;
        match &self.bootstrapping_key {
            CudaBootstrappingKey::Classic(d_bsk) => {
                stream.full_propagate_classic_assign_async(
                    &mut ct.d_blocks.0.d_vec,
                    &d_bsk.d_vec,
                    &self.key_switching_key.d_vec,
                    d_bsk.input_lwe_dimension(),
                    d_bsk.glwe_dimension(),
                    d_bsk.polynomial_size(),
                    self.key_switching_key.decomposition_level_count(),
                    self.key_switching_key.decomposition_base_log(),
                    d_bsk.decomp_level_count(),
                    d_bsk.decomp_base_log(),
                    num_blocks,
                    ct.info.blocks.first().unwrap().message_modulus,
                    ct.info.blocks.first().unwrap().carry_modulus,
                );
            }
            CudaBootstrappingKey::MultiBit(d_multibit_bsk) => {
                stream.full_propagate_multibit_assign_async(
                    &mut ct.d_blocks.0.d_vec,
                    &d_multibit_bsk.d_vec,
                    &self.key_switching_key.d_vec,
                    d_multibit_bsk.input_lwe_dimension(),
                    d_multibit_bsk.glwe_dimension(),
                    d_multibit_bsk.polynomial_size(),
                    self.key_switching_key.decomposition_level_count(),
                    self.key_switching_key.decomposition_base_log(),
                    d_multibit_bsk.decomp_level_count(),
                    d_multibit_bsk.decomp_base_log(),
                    d_multibit_bsk.grouping_factor,
                    num_blocks,
                    ct.info.blocks.first().unwrap().message_modulus,
                    ct.info.blocks.first().unwrap().carry_modulus,
                );
            }
        };
        ct.info
            .blocks
            .iter_mut()
            .for_each(|b| b.degree = Degree::new(b.message_modulus.0 - 1));
    }

    /// Prepend trivial zero LSB blocks to an existing [`CudaRadixCiphertext`] and returns the
    /// result as a new [`CudaRadixCiphertext`]. This can be useful for casting operations.
    ///
    /// # Example
    ///
    ///```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::IntegerCiphertext;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let num_blocks = 4;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let mut d_ct1: CudaRadixCiphertext = sks.create_trivial_radix(7u64, num_blocks, &mut stream);
    /// let ct1 = d_ct1.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct1.blocks().len(), 4);
    ///
    /// let added_blocks = 2;
    /// let d_ct_res =
    ///     sks.extend_radix_with_trivial_zero_blocks_lsb(&mut d_ct1, added_blocks, &mut stream);
    /// let ct_res = d_ct_res.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct_res.blocks().len(), 6);
    ///
    /// // Decrypt
    /// let res: u64 = cks.decrypt(&ct_res);
    /// assert_eq!(
    ///     7 * (PARAM_MESSAGE_2_CARRY_2_KS_PBS.message_modulus.0 as u64).pow(added_blocks as u32),
    ///     res
    /// );
    /// ```
    pub fn extend_radix_with_trivial_zero_blocks_lsb(
        &self,
        ct: &CudaRadixCiphertext,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        let new_num_blocks = ct.d_blocks.lwe_ciphertext_count().0 + num_blocks;
        let ciphertext_modulus = ct.d_blocks.ciphertext_modulus();
        let lwe_size = ct.d_blocks.lwe_dimension().to_lwe_size();
        let shift = num_blocks * lwe_size.0;

        let mut extended_ct_vec =
            unsafe { CudaVec::new_async(new_num_blocks * lwe_size.0, stream) };
        unsafe {
            extended_ct_vec.memset_async(0u64, stream);
            extended_ct_vec.copy_self_range_gpu_to_gpu_async(shift.., &ct.d_blocks.0.d_vec, stream);
        }
        stream.synchronize();
        let extended_ct_list = CudaLweCiphertextList::from_cuda_vec(
            extended_ct_vec,
            LweCiphertextCount(new_num_blocks),
            ciphertext_modulus,
        );

        let extended_ct_info = ct
            .info
            .after_extend_radix_with_trivial_zero_blocks_lsb(num_blocks);
        CudaRadixCiphertext::new(extended_ct_list, extended_ct_info)
    }

    /// Append trivial zero MSB blocks to an existing [`CudaRadixCiphertext`] and returns the result
    /// as a new [`CudaRadixCiphertext`]. This can be useful for casting operations.
    ///
    /// # Example
    ///
    ///```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::IntegerCiphertext;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let num_blocks = 4;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let mut d_ct1: CudaRadixCiphertext = sks.create_trivial_radix(7u64, num_blocks, &mut stream);
    /// let ct1 = d_ct1.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct1.blocks().len(), 4);
    ///
    /// let d_ct_res = sks.extend_radix_with_trivial_zero_blocks_msb(&d_ct1, 2, &mut stream);
    /// let ct_res = d_ct_res.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct_res.blocks().len(), 6);
    ///
    /// // Decrypt
    /// let res: u64 = cks.decrypt(&ct_res);
    /// assert_eq!(7, res);
    /// ```
    pub fn extend_radix_with_trivial_zero_blocks_msb(
        &self,
        ct: &CudaRadixCiphertext,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        let new_num_blocks = ct.d_blocks.lwe_ciphertext_count().0 + num_blocks;
        let ciphertext_modulus = ct.d_blocks.ciphertext_modulus();
        let lwe_size = ct.d_blocks.lwe_dimension().to_lwe_size();

        let mut extended_ct_vec =
            unsafe { CudaVec::new_async(new_num_blocks * lwe_size.0, stream) };
        unsafe {
            extended_ct_vec.memset_async(0u64, stream);
            extended_ct_vec.copy_from_gpu_async(&ct.d_blocks.0.d_vec, stream);
        }
        stream.synchronize();
        let extended_ct_list = CudaLweCiphertextList::from_cuda_vec(
            extended_ct_vec,
            LweCiphertextCount(new_num_blocks),
            ciphertext_modulus,
        );

        let extended_ct_info = ct
            .info
            .after_extend_radix_with_trivial_zero_blocks_msb(num_blocks);
        CudaRadixCiphertext::new(extended_ct_list, extended_ct_info)
    }

    /// Remove LSB blocks from an existing [`CudaRadixCiphertext`] and returns the result as a new
    /// [`CudaRadixCiphertext`]. This can be useful for casting operations.
    ///
    /// # Example
    ///
    ///```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::IntegerCiphertext;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let num_blocks = 4;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let mut d_ct1: CudaRadixCiphertext = sks.create_trivial_radix(119u64, num_blocks, &mut stream);
    /// let ct1 = d_ct1.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct1.blocks().len(), 4);
    ///
    /// let d_ct_res = sks.trim_radix_blocks_lsb(&d_ct1, 2, &mut stream);
    /// let ct_res = d_ct_res.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct_res.blocks().len(), 2);
    ///
    /// // Decrypt
    /// let res: u64 = cks.decrypt(&ct_res);
    /// assert_eq!(7, res);
    /// ```
    pub fn trim_radix_blocks_lsb(
        &self,
        ct: &CudaRadixCiphertext,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        let new_num_blocks = ct.d_blocks.lwe_ciphertext_count().0 - num_blocks;
        let ciphertext_modulus = ct.d_blocks.ciphertext_modulus();
        let lwe_size = ct.d_blocks.lwe_dimension().to_lwe_size();
        let shift = num_blocks * lwe_size.0;

        let mut trimmed_ct_vec = unsafe { CudaVec::new_async(new_num_blocks * lwe_size.0, stream) };
        unsafe {
            trimmed_ct_vec.copy_src_range_gpu_to_gpu_async(shift.., &ct.d_blocks.0.d_vec, stream);
        }
        stream.synchronize();
        let trimmed_ct_list = CudaLweCiphertextList::from_cuda_vec(
            trimmed_ct_vec,
            LweCiphertextCount(new_num_blocks),
            ciphertext_modulus,
        );

        let trimmed_ct_info = ct.info.after_trim_radix_blocks_lsb(num_blocks);
        CudaRadixCiphertext::new(trimmed_ct_list, trimmed_ct_info)
    }

    /// Remove MSB blocks from an existing [`CudaRadixCiphertext`] and returns the result as a new
    /// [`CudaRadixCiphertext`]. This can be useful for casting operations.
    ///
    /// # Example
    ///
    ///```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::IntegerCiphertext;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let num_blocks = 4;
    ///
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let mut d_ct1: CudaRadixCiphertext = sks.create_trivial_radix(119u64, num_blocks, &mut stream);
    /// let ct1 = d_ct1.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct1.blocks().len(), 4);
    ///
    /// let d_ct_res = sks.trim_radix_blocks_msb(&d_ct1, 2, &mut stream);
    /// let ct_res = d_ct_res.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct_res.blocks().len(), 2);
    ///
    /// // Decrypt
    /// let res: u64 = cks.decrypt(&ct_res);
    /// assert_eq!(7, res);
    /// ```
    pub fn trim_radix_blocks_msb(
        &self,
        ct: &CudaRadixCiphertext,
        num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        let new_num_blocks = ct.d_blocks.lwe_ciphertext_count().0 - num_blocks;
        let ciphertext_modulus = ct.d_blocks.ciphertext_modulus();
        let lwe_size = ct.d_blocks.lwe_dimension().to_lwe_size();
        let shift = new_num_blocks * lwe_size.0;

        let mut trimmed_ct_vec = unsafe { CudaVec::new_async(new_num_blocks * lwe_size.0, stream) };
        unsafe {
            trimmed_ct_vec.copy_src_range_gpu_to_gpu_async(0..shift, &ct.d_blocks.0.d_vec, stream);
        }
        stream.synchronize();
        let trimmed_ct_list = CudaLweCiphertextList::from_cuda_vec(
            trimmed_ct_vec,
            LweCiphertextCount(new_num_blocks),
            ciphertext_modulus,
        );

        let trimmed_ct_info = ct.info.after_trim_radix_blocks_msb(num_blocks);
        CudaRadixCiphertext::new(trimmed_ct_list, trimmed_ct_info)
    }

    /// Cast a CudaRadixCiphertext to a CudaRadixCiphertext
    /// with a possibly different number of blocks
    ///
    /// # Example
    ///
    ///```rust
    /// use tfhe::core_crypto::gpu::{CudaDevice, CudaStream};
    /// use tfhe::integer::gpu::ciphertext::CudaRadixCiphertext;
    /// use tfhe::integer::gpu::gen_keys_radix_gpu;
    /// use tfhe::integer::IntegerCiphertext;
    /// use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
    ///
    /// let num_blocks = 4;
    /// let gpu_index = 0;
    /// let device = CudaDevice::new(gpu_index);
    /// let mut stream = CudaStream::new_unchecked(device);
    ///
    /// // Generate the client key and the server key:
    /// let (cks, sks) = gen_keys_radix_gpu(PARAM_MESSAGE_2_CARRY_2_KS_PBS, num_blocks, &mut stream);
    ///
    /// let msg = 2u8;
    ///
    /// let mut d_ct1: CudaRadixCiphertext = sks.create_trivial_radix(msg, num_blocks, &mut stream);
    /// let ct1 = d_ct1.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct1.blocks().len(), 4);
    ///
    /// let d_ct_res = sks.cast_to_unsigned(d_ct1, 8, &mut stream);
    /// let ct_res = d_ct_res.to_radix_ciphertext(&mut stream);
    /// assert_eq!(ct_res.blocks().len(), 8);
    ///
    /// // Decrypt
    /// let res: u16 = cks.decrypt(&ct_res);
    /// assert_eq!(msg as u16, res);
    /// ```
    pub fn cast_to_unsigned(
        &self,
        mut source: CudaRadixCiphertext,
        target_num_blocks: usize,
        stream: &CudaStream,
    ) -> CudaRadixCiphertext {
        if !source.as_ref().block_carries_are_empty() {
            unsafe {
                self.full_propagate_assign_async(source.as_mut(), stream);
            }
            stream.synchronize();
        }
        let current_num_blocks = source.info.blocks.len();
        // Casting from unsigned to unsigned, this is just about trimming/extending with zeros
        if target_num_blocks > current_num_blocks {
            let num_blocks_to_add = target_num_blocks - current_num_blocks;
            self.extend_radix_with_trivial_zero_blocks_msb(source.as_ref(), num_blocks_to_add, stream)
        } else {
            let num_blocks_to_remove = current_num_blocks - target_num_blocks;
            self.trim_radix_blocks_msb(source.as_ref(), num_blocks_to_remove, stream)
        }
    }
}
