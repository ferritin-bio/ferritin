#[cfg(test)]
mod tests {
    use candle_core::pickle::read_pth_tensor_info;
    use candle_core::{DType, Device, Error};
    use candle_nn::VarBuilder;
    use ferritin_test_data::TestFile;
    use ferritin_featurizers::{ProteinMPNN, ProteinMPNNConfig};

    #[test]

    fn test_load_ligandmpnn() -> Result<(), Error> {
        let (mpnn_file, _handle) = TestFile::ligmpnn_pmpnn_01().create_temp()?;
        let device = Device::Cpu;

        // Read the tensor info from a .pth file.
        //
        // # Arguments
        // * `file` - The path to the .pth file.
        // * `verbose` - Whether to print debug information.
        // * `key` - Optional key to retrieve `state_dict` from the pth file.

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

        let vb = VarBuilder();
        let config =
        let pmpnn_model = ProteinMPNN::load(vb, config);
        Ok(())
    }
}
