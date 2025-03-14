use anyhow::Result;
use candle_core::{Device, Tensor};
use ndarray;
// use ndarray_safetensors::{TensorViewWithDataBuffer, parse_tensors};
// use safetensors::{SafeTensors, serialize};

// pub fn ndarray_to_tensor(tensor: ValueRef) -> Result<candle_core::Tensor> {
//     let tmp_data = [("_", tensor)];
//     let st = serialize(tmp_data, &None)?;
//     let tensors = SafeTensors::deserialize(&st).unwrap();
//     let arrays = parse_tensors::<i64>(&tensors).unwrap();
//     Ok(arrays.into_iter().next().unwrap().1)
// }

// pub fn ndarray_to_tensor_f32(
//     arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
// ) -> Result<Tensor> {
//     let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
//     let serialized_data = safetensors::serialize(data, &None).unwrap();
//     let tensor_hash = candle_core::safetensors::load_buffer(&serialized_data, &Device::Cpu)?;
//     Ok(tensor_hash
//         .get("arr")
//         .ok_or(anyhow::anyhow!("array not found"))?
//         .clone())
// }

// pub fn ndarray_to_tensor_i64(
//     arr: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>,
// ) -> Result<Tensor> {
//     let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
//     let serialized_data = safetensors::serialize(data, &None).unwrap();
//     let tensor_hash = candle_core::safetensors::load_buffer(&serialized_data, &Device::Cpu)?;
//     Ok(tensor_hash
//         .get("arr")
//         .ok_or(anyhow::anyhow!("array not found"))?
//         .clone())
// }

// pub fn tensor_to_ndarray_f32(
//     tensor: candle_core::Tensor,
// ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
//     let tmp_data = [("_", tensor)];
//     let st = serialize(tmp_data, &None)?;
//     let tensors = SafeTensors::deserialize(&st).unwrap();
//     let arrays = parse_tensors::<f32>(&tensors).unwrap();
//     Ok(arrays.into_iter().next().unwrap().1)
// }

// pub fn tensor_to_ndarray_i64(
//     tensor: candle_core::Tensor,
// ) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
//     let tmp_data = [("_", tensor)];
//     let st = serialize(tmp_data, &None)?;
//     let tensors = SafeTensors::deserialize(&st).unwrap();
//     let arrays = parse_tensors::<i64>(&tensors).unwrap();
//     Ok(arrays.into_iter().next().unwrap().1)
// }

pub fn ndarray_to_tensor_f32(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
) -> Result<Tensor> {
    // Get shape from ndarray
    let shape: Vec<usize> = arr.shape().to_vec();

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
    // Get raw bytes from tensor
    let raw_data = tensor
        .to_vec1::<f32>()?
        .as_slice()
        .iter()
        .flat_map(|&x| x.to_ne_bytes())
        .collect::<Vec<u8>>();

    // Create ndarray shape
    let shape = tensor.dims();

    // Convert raw bytes back to f32 values
    let f32_data: Vec<f32> = raw_data
        .chunks_exact(4)
        .map(|chunk| {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(chunk);
            f32::from_ne_bytes(bytes)
        })
        .collect();

    // Create ndarray from data
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), f32_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}

pub fn tensor_to_ndarray_i64(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    // Get values from tensor
    let shape = tensor.dims().to_vec();
    let i64_data = tensor.to_vec1::<i64>()?;

    // Create ndarray from data
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&shape), i64_data)
        .map_err(|e| anyhow::anyhow!("Failed to create ndarray: {}", e))
}
