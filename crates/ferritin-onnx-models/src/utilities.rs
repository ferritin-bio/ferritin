use anyhow::Result;
use candle_core::{Device, Tensor};
use ndarray;

pub fn ndarray_to_tensor_f32(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
) -> Result<Tensor> {
    // Get shape from ndarray
    let shape: Vec<usize> = arr.shape().to_vec();

    println!("Shape: {:?}", shape);
    // Get raw bytes from ndarray (converting f32 values to bytes)
    let raw_data = arr
        .as_slice()
        .unwrap()
        .iter()
        .flat_map(|&x| x.to_ne_bytes())
        .collect::<Vec<u8>>();

    // Create Tensor from raw bytes
    Tensor::from_raw_buffer(
        &raw_data,               // Raw byte data
        candle_core::DType::F32, // Data type
        &shape,                  // Shape vector
        &Device::Cpu,            // Device
    )
    .map_err(|e| anyhow::anyhow!("Failed to create tensor: {}", e))
}

pub fn tensor_to_ndarray_f32(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
    // Get shape from tensor
    let shape = tensor.dims().to_vec();

    // Flatten the tensor (safely handles multi-dimensional tensors)
    let flattened = tensor.flatten_all()?;

    // Get the data as a 1D vector (now appropriate since we flattened it)
    let f32_data = flattened.to_vec1::<f32>()?;

    // Create ndarray from data with original shape
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), f32_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}

pub fn tensor_to_ndarray_i64(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    // Get shape from tensor
    let shape = tensor.dims().to_vec();

    // Flatten the tensor
    let flattened = tensor.flatten_all()?;

    // Get the data as a 1D vector
    let i64_data = flattened.to_vec1::<i64>()?;

    // Create ndarray from data with original shape
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), i64_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}
