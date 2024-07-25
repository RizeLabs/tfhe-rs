use crate::core_crypto::gpu::glwe_ciphertext::CudaGlweCiphertext;
use crate::core_crypto::gpu::lwe_ciphertext::CudaLweCiphertext;
use crate::core_crypto::gpu::vec::CudaVec;
use crate::core_crypto::gpu::{extract_lwe_sample_from_glwe_ciphertext_async, CudaStreams};
use crate::core_crypto::prelude::{LweCiphertextCount, MonomialDegree, UnsignedTorus};

/// Extract the nth coefficient from the body of a [`GLWE Ciphertext`](`CudaGlweCiphertext`) as an
/// [`LWE ciphertext`](`CudaLweCiphertext`). This variant is GPU-accelerated.
pub fn cuda_extract_lwe_sample_from_glwe_ciphertext<Scalar>(
    input_glwe: &CudaGlweCiphertext<Scalar>,
    output_lwe: &mut CudaLweCiphertext<Scalar>,
    nth: MonomialDegree,
    streams: &CudaStreams,
) where
    // CastInto required for PBS modulus switch which returns a usize
    Scalar: UnsignedTorus,
{
    let in_lwe_dim = input_glwe
        .glwe_dimension()
        .to_equivalent_lwe_dimension(input_glwe.polynomial_size());

    let out_lwe_dim = output_lwe.lwe_dimension();

    assert_eq!(
        in_lwe_dim, out_lwe_dim,
        "Mismatch between equivalent LweDimension of input ciphertext and output ciphertext. \
        Got {in_lwe_dim:?} for input and {out_lwe_dim:?} for output.",
    );

    assert_eq!(
        input_glwe.ciphertext_modulus(),
        output_lwe.ciphertext_modulus(),
        "Mismatched moduli between input_glwe ({:?}) and output_lwe ({:?})",
        input_glwe.ciphertext_modulus(),
        output_lwe.ciphertext_modulus()
    );

    let nth_array: Vec<u32> = vec![nth.0 as u32];
    let gpu_indexes = &streams.gpu_indexes;
    unsafe {
        let d_nth_array = CudaVec::from_cpu_async(&nth_array, streams, gpu_indexes[0]);
        extract_lwe_sample_from_glwe_ciphertext_async(
            streams,
            &mut output_lwe.0.d_vec,
            &input_glwe.0.d_vec,
            &d_nth_array,
            LweCiphertextCount(nth_array.len()),
            input_glwe.glwe_dimension(),
            input_glwe.polynomial_size(),
        );
    }
}
