use anyhow::Result;
use candle_core::{Device, Tensor};
use ndarray;

pub fn ndarray_to_tensor_f32(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
) -> Result<Tensor> {
    let shape: Vec<usize> = arr.shape().to_vec();
    let raw_data = arr
        .as_slice()
        .unwrap()
        .iter()
        .flat_map(|&x| x.to_ne_bytes())
        .collect::<Vec<u8>>();

    Tensor::from_raw_buffer(&raw_data, candle_core::DType::F32, &shape, &Device::Cpu)
        .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))
}

pub fn tensor_to_ndarray_f32(
    tensor: Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
    let shape = tensor.dims().to_vec();
    let flattened = tensor.flatten_all()?;
    let f32_data = flattened.to_vec1::<f32>()?;
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), f32_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}

pub fn tensor_to_ndarray_i64(
    tensor: Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    let shape = tensor.dims().to_vec();
    let flattened = tensor.flatten_all()?;
    let i64_data = flattened.to_vec1::<i64>()?;
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), i64_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}
