use anyhow::Result;
use candle_core::{Device, Tensor};
use ndarray;
use ndarray_safetensors::{parse_tensors, TensorViewWithDataBuffer};
use safetensors::{serialize, SafeTensors};

pub fn ndarray_to_tensor_f32(
    arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>,
) -> Result<Tensor> {
    let data = vec![("arr", TensorViewWithDataBuffer::new(&arr))];
    let serialized_data = serialize(data, &None)?;
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
    let serialized_data = serialize(data, &None)?;
    let tensor_hash = candle_core::safetensors::load_buffer(&serialized_data, &Device::Cpu)?;
    Ok(tensor_hash
        .get("arr")
        .ok_or(anyhow::anyhow!("array not found"))?
        .clone())
}

pub fn tensor_to_ndarray_f32(
    tensor: Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st)?;
    let arrays = parse_tensors::<f32>(&tensors)?;
    Ok(arrays.into_iter().next().unwrap().1)
}

pub fn tensor_to_ndarray_i64(
    tensor: Tensor,
) -> Result<ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::IxDyn>> {
    let tmp_data = [("_", tensor)];
    let st = serialize(tmp_data, &None)?;
    let tensors = SafeTensors::deserialize(&st)?;
    let arrays = parse_tensors::<i64>(&tensors)?;
    Ok(arrays.into_iter().next().unwrap().1)
}
