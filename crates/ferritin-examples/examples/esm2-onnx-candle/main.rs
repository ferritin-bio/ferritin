// use anyhow::{anyhow, Result};
// use candle_core::{DType, Device, Tensor, D};
// use candle_onnx::onnx::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
// use ferritin_onnx_models::{ESM2Models, ESM2};

// fn main() -> Result<()> {
//     let sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYHSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH";
//     let model_path = ESM2::load_model_path(ESM2Models::ESM2_T6_8M)?;
//     let onnx_model = ESM2::new(ESM2Models::ESM2_T6_8M)?;
//     let tokens = onnx_model
//         .tokenizer
//         .encode(sequence, false)
//         .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
//     let token_ids: Vec<i64> = tokens.get_ids().iter().map(|&x| x as i64).collect();
//     let shape = (1, tokens.len() as usize);
//     let model = candle_onnx::read_file(model_path)?;
//     let graph = model.graph.as_ref().unwrap();

//     println!("\n\nGraph Inputs: \n ");
//     for input in graph.input.clone() {
//         println!("Inputs: {:?}\n", input);
//     }

//     let mut inputs = std::collections::HashMap::new();
//     let mask = Tensor::ones(shape, DType::I64, &Device::Cpu)?;
//     let in_01 = Tensor::from_vec(token_ids, shape, &Device::Cpu)?;
//     println!("\nmask dimensions: {:?}", mask.shape());
//     println!("input_ids dimensions: {:?}", in_01.shape());
//     inputs.insert("input_ids".to_string(), in_01);
//     inputs.insert("attention_mask".to_string(), mask);
//     println!("\n\nInputs: {:?}\n", inputs);

//     // let outputs = candle_onnx::simple_eval(&model, inputs)?;

//     let output = match candle_onnx::simple_eval(&model, inputs) {
//         Ok(ret) => ret,
//         Err(err) => {
//             if let candle_core::Error::WithBacktrace { inner, .. } = &err {
//                 match &**inner {
//                     candle_core::Error::ShapeMismatch { buffer_size, shape } => {
//                         eprintln!("buffer_size: {}", buffer_size);
//                         eprintln!("shape: {:?}", shape);
//                         eprintln!("shape.elem_count: {}", shape.elem_count());
//                     }
//                     _ => (),
//                 }
//             }
//             return Err(err.into());
//         }
//     };

//     // for (name, value) in outputs.iter() {
//     //     println!("output {name}: {value:?}")
//     // }
//     // let logits = esm2.run_model(protein)?;
//     // let normed = esm2.extract_logits(&logits)?;
//     Ok(())
// }

// // Todo: Are we masking this correctly?
// // let mask_array: Array2<i64> = Array2::from_shape_vec(shape, vec![1; tokens.len()])?;
// // let tokens_array: Array2<i64> = Array2::from_shape_vec(
// //     shape,
// //     token_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
// // )?;
// // let outputs =
// //     model.run(ort::inputs!["input_ids" => tokens_array,"attention_mask" => mask_array]?)?;
// // let logits = outputs["logits"].try_extract_tensor::<f32>()?.to_owned();

// // RUST_BACKTRACE=1 cargo run --example esm2-onnx-candle

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_onnx::onnx::tensor_shape_proto::dimension::Value::{DimParam, DimValue};
use ferritin_onnx_models::{ESM2Models, ESM2};

fn main() -> Result<()> {
    let sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH";
    let model_path = ESM2::load_model_path(ESM2Models::ESM2_T6_8M)?;
    let onnx_model = ESM2::new(ESM2Models::ESM2_T6_8M)?;

    // Test tokenization
    let tokens = onnx_model
        .tokenizer
        .encode(sequence, false)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

    let token_ids: Vec<i64> = tokens.get_ids().iter().map(|x| *x as i64).collect();
    println!("Token_ids length: {}", token_ids.len());

    let batch_size = 1usize;
    let seq_length = token_ids.len();
    let shape = (batch_size, seq_length);

    // Test creating input_ids tensor
    println!("Testing input_ids tensor creation...");
    let input_ids = match Tensor::from_vec(token_ids.clone(), shape, &Device::Cpu) {
        Ok(t) => {
            println!(
                "✓ input_ids tensor created successfully with shape: {:?}",
                t.shape()
            );
            t
        }
        Err(e) => {
            println!("✗ Failed to create input_ids tensor: {:?}", e);
            return Err(anyhow!("Failed to create input_ids tensor"));
        }
    };

    // Test creating attention_mask tensor
    println!("\nTesting attention_mask tensor creation...");
    let attention_mask = match Tensor::ones(shape, DType::I64, &Device::Cpu) {
        Ok(t) => {
            println!(
                "✓ attention_mask tensor created successfully with shape: {:?}",
                t.shape()
            );
            t
        }
        Err(e) => {
            println!("✗ Failed to create attention_mask tensor: {:?}", e);
            return Err(anyhow!("Failed to create attention_mask tensor"));
        }
    };

    // Test model loading and examine structure
    println!("\nLoading and examining model...");
    let model = candle_onnx::read_file(model_path)?;
    let graph = model.graph.as_ref().unwrap();

    println!("\nModel inputs:");
    for input in graph.input.iter() {
        println!("{:#?}", input);
    }

    println!("\nFirst few nodes in the model:");
    for (idx, node) in graph.node.iter().take(5).enumerate() {
        println!("\nNode {}:", idx);
        println!("  Op Type: {}", node.op_type);
        println!("  Inputs: {:?}", node.input);
        println!("  Outputs: {:?}", node.output);
    }

    let mut inputs = std::collections::HashMap::new();
    inputs.insert("input_ids".to_string(), input_ids);
    inputs.insert("attention_mask".to_string(), attention_mask);

    // Try evaluation with detailed error reporting
    println!("\nAttempting model evaluation...");
    match candle_onnx::simple_eval(&model, inputs) {
        Ok(outputs) => {
            println!("✓ Model evaluation successful!");
            for (name, value) in outputs.iter() {
                println!("Output {}: shape: {:?}", name, value.shape());
            }
        }
        Err(e) => {
            println!("✗ Model evaluation failed with error: {:?}", e);
        }
    }

    Ok(())
}
