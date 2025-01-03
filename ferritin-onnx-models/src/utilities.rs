use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use ndarray;
use ndarray_safetensors::{parse_tensors, TensorViewWithDataBuffer};
use safetensors::{self as st, serialize, tensor::TensorView, Dtype as ST_Dtype, SafeTensors};

use std::collections::HashMap;

// pub fn ndarray_to_tensor(tensor: ValueRef) -> Result<candle_core::Tensor> {
//     let tmp_data = [("_", tensor)];
//     let st = serialize(tmp_data, &None)?;
//     let tensors = SafeTensors::deserialize(&st).unwrap();
//     let arrays = parse_tensors::<i64>(&tensors).unwrap();
//     Ok(arrays.into_iter().next().unwrap().1)
// }

pub fn ndarray_to_tensor_f32(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
) -> Result<Tensor> {
    let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
    let serialized_data = safetensors::serialize(data, &None).unwrap();
    let tensor_hash = candle_core::safetensors::load_buffer(&serialized_data, &Device::Cpu)?;
    Ok(tensor_hash
        .get("arr")
        .ok_or(anyhow::anyhow!("array not found"))?
        .clone())
}

pub fn ndarray_to_tensor_i64(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>,
) -> Result<Tensor> {
    let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
    let serialized_data = safetensors::serialize(data, &None).unwrap();
    let tensor_hash = candle_core::safetensors::load_buffer(&serialized_data, &Device::Cpu)?;
    Ok(tensor_hash
        .get("arr")
        .ok_or(anyhow::anyhow!("array not found"))?
        .clone())
}

pub fn tensor_to_ndarray_f32(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st).unwrap();
    let arrays = parse_tensors::<f32>(&tensors).unwrap();
    Ok(arrays.into_iter().next().unwrap().1)
}

pub fn tensor_to_ndarray_i64(
    tensor: candle_core::Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st).unwrap();
    let arrays = parse_tensors::<i64>(&tensors).unwrap();
    Ok(arrays.into_iter().next().unwrap().1)
}
