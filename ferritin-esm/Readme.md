# ferritin-esm

- [esm](https://github.com/evolutionaryscale/esm)
- [cambrian](https://www.evolutionaryscale.ai/blog/esm-cambrian)



# ESMC

## Model Layers

```rust
let model_id = "EvolutionaryScale/esmc-300m-2024-12";
let revision = "main";
let api = Api::new()?;
let repo = api.repo(Repo::with_revision(
    model_id.to_string(),
    RepoType::Model,
    revision.to_string(),
));
let weights_path = repo.get("data/weights/esmc_300m_2024_12_v0.pth")?;
let pth = PthTensors::new(weights_path, None)?;

// print the names
for (name, tensor) in pth.tensor_infos() {
    println!("{}: {:?}", name, tensor);
}
```


```text
embed.weight: TensorInfo { name: "embed.weight", dtype: F32, layout: Layout { shape: [64, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/301", storage_size: 61440 }
sequence_head.0.bias: TensorInfo { name: "sequence_head.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/303", storage_size: 960 }
sequence_head.0.weight: TensorInfo { name: "sequence_head.0.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/302", storage_size: 921600 }
sequence_head.2.bias: TensorInfo { name: "sequence_head.2.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/305", storage_size: 960 }
sequence_head.2.weight: TensorInfo { name: "sequence_head.2.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/304", storage_size: 960 }
sequence_head.3.bias: TensorInfo { name: "sequence_head.3.bias", dtype: F32, layout: Layout { shape: [64], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/307", storage_size: 64 }
sequence_head.3.weight: TensorInfo { name: "sequence_head.3.weight", dtype: F32, layout: Layout { shape: [64, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/306", storage_size: 61440 }
transformer.blocks.0.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.0.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/4", storage_size: 960 }
transformer.blocks.0.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.0.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/1", storage_size: 960 }
transformer.blocks.0.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.0.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/0", storage_size: 960 }
transformer.blocks.0.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.0.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/2", storage_size: 2764800 }
transformer.blocks.0.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.0.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/5", storage_size: 921600 }
transformer.blocks.0.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.0.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/3", storage_size: 960 }
transformer.blocks.0.ffn.0.bias: TensorInfo { name: "transformer.blocks.0.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/7", storage_size: 960 }
transformer.blocks.0.ffn.0.weight: TensorInfo { name: "transformer.blocks.0.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/6", storage_size: 960 }
transformer.blocks.0.ffn.1.weight: TensorInfo { name: "transformer.blocks.0.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/8", storage_size: 4915200 }
transformer.blocks.0.ffn.3.weight: TensorInfo { name: "transformer.blocks.0.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/9", storage_size: 2457600 }
transformer.blocks.1.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.1.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/14", storage_size: 960 }
transformer.blocks.1.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.1.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/11", storage_size: 960 }
transformer.blocks.1.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.1.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/10", storage_size: 960 }
transformer.blocks.1.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.1.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/12", storage_size: 2764800 }
transformer.blocks.1.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.1.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/15", storage_size: 921600 }
transformer.blocks.1.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.1.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/13", storage_size: 960 }
transformer.blocks.1.ffn.0.bias: TensorInfo { name: "transformer.blocks.1.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/17", storage_size: 960 }
transformer.blocks.1.ffn.0.weight: TensorInfo { name: "transformer.blocks.1.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/16", storage_size: 960 }
transformer.blocks.1.ffn.1.weight: TensorInfo { name: "transformer.blocks.1.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/18", storage_size: 4915200 }
transformer.blocks.1.ffn.3.weight: TensorInfo { name: "transformer.blocks.1.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/19", storage_size: 2457600 }
transformer.blocks.2.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.2.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/24", storage_size: 960 }
transformer.blocks.2.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.2.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/21", storage_size: 960 }
transformer.blocks.2.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.2.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/20", storage_size: 960 }
transformer.blocks.2.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.2.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/22", storage_size: 2764800 }
transformer.blocks.2.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.2.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/25", storage_size: 921600 }
transformer.blocks.2.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.2.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/23", storage_size: 960 }
transformer.blocks.2.ffn.0.bias: TensorInfo { name: "transformer.blocks.2.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/27", storage_size: 960 }
transformer.blocks.2.ffn.0.weight: TensorInfo { name: "transformer.blocks.2.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/26", storage_size: 960 }
transformer.blocks.2.ffn.1.weight: TensorInfo { name: "transformer.blocks.2.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/28", storage_size: 4915200 }
transformer.blocks.2.ffn.3.weight: TensorInfo { name: "transformer.blocks.2.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/29", storage_size: 2457600 }
transformer.blocks.3.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.3.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/34", storage_size: 960 }
transformer.blocks.3.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.3.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/31", storage_size: 960 }
transformer.blocks.3.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.3.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/30", storage_size: 960 }
transformer.blocks.3.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.3.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/32", storage_size: 2764800 }
transformer.blocks.3.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.3.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/35", storage_size: 921600 }
transformer.blocks.3.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.3.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/33", storage_size: 960 }
transformer.blocks.3.ffn.0.bias: TensorInfo { name: "transformer.blocks.3.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/37", storage_size: 960 }
transformer.blocks.3.ffn.0.weight: TensorInfo { name: "transformer.blocks.3.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/36", storage_size: 960 }
transformer.blocks.3.ffn.1.weight: TensorInfo { name: "transformer.blocks.3.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/38", storage_size: 4915200 }
transformer.blocks.3.ffn.3.weight: TensorInfo { name: "transformer.blocks.3.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/39", storage_size: 2457600 }
transformer.blocks.4.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.4.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/44", storage_size: 960 }
transformer.blocks.4.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.4.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/41", storage_size: 960 }
transformer.blocks.4.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.4.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/40", storage_size: 960 }
transformer.blocks.4.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.4.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/42", storage_size: 2764800 }
transformer.blocks.4.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.4.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/45", storage_size: 921600 }
transformer.blocks.4.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.4.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/43", storage_size: 960 }
transformer.blocks.4.ffn.0.bias: TensorInfo { name: "transformer.blocks.4.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/47", storage_size: 960 }
transformer.blocks.4.ffn.0.weight: TensorInfo { name: "transformer.blocks.4.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/46", storage_size: 960 }
transformer.blocks.4.ffn.1.weight: TensorInfo { name: "transformer.blocks.4.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/48", storage_size: 4915200 }
transformer.blocks.4.ffn.3.weight: TensorInfo { name: "transformer.blocks.4.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/49", storage_size: 2457600 }
transformer.blocks.5.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.5.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/54", storage_size: 960 }
transformer.blocks.5.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.5.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/51", storage_size: 960 }
transformer.blocks.5.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.5.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/50", storage_size: 960 }
transformer.blocks.5.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.5.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/52", storage_size: 2764800 }
transformer.blocks.5.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.5.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/55", storage_size: 921600 }
transformer.blocks.5.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.5.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/53", storage_size: 960 }
transformer.blocks.5.ffn.0.bias: TensorInfo { name: "transformer.blocks.5.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/57", storage_size: 960 }
transformer.blocks.5.ffn.0.weight: TensorInfo { name: "transformer.blocks.5.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/56", storage_size: 960 }
transformer.blocks.5.ffn.1.weight: TensorInfo { name: "transformer.blocks.5.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/58", storage_size: 4915200 }
transformer.blocks.5.ffn.3.weight: TensorInfo { name: "transformer.blocks.5.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/59", storage_size: 2457600 }
transformer.blocks.6.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.6.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/64", storage_size: 960 }
transformer.blocks.6.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.6.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/61", storage_size: 960 }
transformer.blocks.6.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.6.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/60", storage_size: 960 }
transformer.blocks.6.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.6.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/62", storage_size: 2764800 }
transformer.blocks.6.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.6.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/65", storage_size: 921600 }
transformer.blocks.6.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.6.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/63", storage_size: 960 }
transformer.blocks.6.ffn.0.bias: TensorInfo { name: "transformer.blocks.6.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/67", storage_size: 960 }
transformer.blocks.6.ffn.0.weight: TensorInfo { name: "transformer.blocks.6.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/66", storage_size: 960 }
transformer.blocks.6.ffn.1.weight: TensorInfo { name: "transformer.blocks.6.ffn.1.weight", dtype: F32, layout: Layout { shape: [5120, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/68", storage_size: 4915200 }
transformer.blocks.6.ffn.3.weight: TensorInfo { name: "transformer.blocks.6.ffn.3.weight", dtype: F32, layout: Layout { shape: [960, 2560], stride: [2560, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/69", storage_size: 2457600 }
transformer.blocks.7.attn.k_ln.weight: TensorInfo { name: "transformer.blocks.7.attn.k_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/74", storage_size: 960 }
transformer.blocks.7.attn.layernorm_qkv.0.bias: TensorInfo { name: "transformer.blocks.7.attn.layernorm_qkv.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/71", storage_size: 960 }
transformer.blocks.7.attn.layernorm_qkv.0.weight: TensorInfo { name: "transformer.blocks.7.attn.layernorm_qkv.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/70", storage_size: 960 }
transformer.blocks.7.attn.layernorm_qkv.1.weight: TensorInfo { name: "transformer.blocks.7.attn.layernorm_qkv.1.weight", dtype: F32, layout: Layout { shape: [2880, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/72", storage_size: 2764800 }
transformer.blocks.7.attn.out_proj.weight: TensorInfo { name: "transformer.blocks.7.attn.out_proj.weight", dtype: F32, layout: Layout { shape: [960, 960], stride: [960, 1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/75", storage_size: 921600 }
transformer.blocks.7.attn.q_ln.weight: TensorInfo { name: "transformer.blocks.7.attn.q_ln.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/73", storage_size: 960 }
transformer.blocks.7.ffn.0.bias: TensorInfo { name: "transformer.blocks.7.ffn.0.bias", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/77", storage_size: 960 }
transformer.blocks.7.ffn.0.weight: TensorInfo { name: "transformer.blocks.7.ffn.0.weight", dtype: F32, layout: Layout { shape: [960], stride: [1], start_offset: 0 }, path: "esmc_300m_v0_fp32/data/76", storage_size: 960 }
transformer.blocks.7.ffn.1.weight: TensorInfo { name: "transformer.blocks.7.
