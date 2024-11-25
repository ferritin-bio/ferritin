#[cfg(test)]
mod tests {
    use candle_core::pickle::read_pth_tensor_info;
    use candle_core::pickle::PthTensors;
    use candle_core::{DType, Device, Error};
    use candle_nn::VarBuilder;
    use ferritin_featurizers::{ProteinMPNN, ProteinMPNNConfig};
    use ferritin_test_data::TestFile;

    #[test]
    fn test_load_ligandmpnn_01() -> Result<(), Error> {
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let mut tensors = read_pth_tensor_info(mpnn_file, false, Some("model_state_dict"))?;
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        for tensor_info in tensors.iter() {
            println!(
                "{}: [{:?}; {:?}]",
                tensor_info.name,
                tensor_info.layout.shape(),
                tensor_info.dtype,
            );

            println!("    {:?}", tensor_info);
        }
        Ok(())
    }
    #[test]
    fn test_load_ligandmpnn_02() -> Result<(), Error> {
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let vb = VarBuilder::from_pth(mpnn_file, DType::F32, &Device::Cpu)?;
        println!(
            "Check Tensor Membership for Norm1 {:?}",
            vb.contains_tensor("encoder_layers.0.norm1.bias")
        );
        let pconf = ProteinMPNNConfig::proteinmpnn();
        let pmpnn = ProteinMPNN::load(vb, &pconf);

        Ok(())
    }
    #[test]
    fn test_load_ligandmpnn_03() -> Result<(), Error> {
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let pth = PthTensors::new(mpnn_file, Some("model_state_dict"))?;
        let vb = VarBuilder::from_backend(Box::new(pth), DType::F32, Device::Cpu);
        let pconf = ProteinMPNNConfig::proteinmpnn();
        let pmpnn = ProteinMPNN::load(vb, &pconf);
        Ok(())
    }
}
