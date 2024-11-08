## Amplify Inference

- [amplify model](https://github.com/chandar-lab/AMPLIFY)
- [amplify hf - 120M](https://huggingface.co/chandar-lab/AMPLIFY_120M)



```sh
cargo run --example amplify
```


## Model Tensors

## Pytorch 350

```sh
AMPLIFY(
  (encoder): Embedding(27, 960, padding_idx=0)
  (transformer_encoder): ModuleList(
    (0-31): 32 x EncoderBlock(
      (q): Linear(in_features=960, out_features=960, bias=False)
      (k): Linear(in_features=960, out_features=960, bias=False)
      (v): Linear(in_features=960, out_features=960, bias=False)
      (wo): Linear(in_features=960, out_features=960, bias=False)
      (resid_dropout): Dropout(p=0, inplace=False)
      (ffn): SwiGLU(
        (w12): Linear(in_features=960, out_features=5120, bias=False)
        (w3): Linear(in_features=2560, out_features=960, bias=False)
      )
      (attention_norm): RMSNorm()
      (ffn_norm): RMSNorm()
      (ffn_dropout): Dropout(p=0, inplace=False)
    )
  )
  (layer_norm_2): RMSNorm()
  (decoder): Linear(in_features=960, out_features=27, bias=True)
)
```

### Amplify 120M in Candle

Sorting this alphabetically its a bit clearer to see the architecture:

```rust
pub struct AMPLIFY {
    encoder: Embedding,              // <- encoder.weight
    layer_norm_1: Option<RMSNorm>,   // <- not used
    encoder: Vec<EncoderBlock>,      // <- 24x levels
    layer_norm_2: Option<RMSNorm>,   // <- layer_norm_2
    decoder: Linear,                 // <- decoder.weight
    freqs_cis: Tensor,               // <- decoder.bias
}

// hidden_size: 640,
//
// transformer_encoder.<>.
//    .attention_norm.weight   ||  Shape: [640]
//    .k.weight                ||  Shape: [640, 640]
//    .q.weight                ||  Shape: [640, 640]
//    .v.weight                ||  Shape: [640, 640]
//    .0.wo.weight             ||  Shape: [640, 640]
//    .ffn.w12.weight          ||  Shape: [3424, 640]
//    .ffn.w3.weight           ||  Shape: [640, 1712]
//    .ffn_norm.weight         ||  Shape: [640]
```

```txt
Model tensors:
Tensor: decoder.bias                                  ||  Shape: [27]
Tensor: decoder.weight                                ||  Shape: [27, 640]
Tensor: encoder.weight                                ||  Shape: [27, 640]
Tensor: layer_norm_2.weight                           ||  Shape: [640]
Tensor: transformer_encoder.0.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.0.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.0.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.0.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.0.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.0.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.0.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.0.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.1.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.1.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.1.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.1.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.1.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.1.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.1.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.1.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.2.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.2.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.2.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.2.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.2.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.2.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.2.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.2.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.3.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.3.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.3.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.3.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.3.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.3.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.3.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.3.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.4.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.4.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.4.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.4.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.4.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.4.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.4.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.4.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.5.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.5.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.5.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.5.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.5.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.5.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.5.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.5.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.6.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.6.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.6.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.6.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.6.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.6.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.6.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.6.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.7.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.7.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.7.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.7.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.7.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.7.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.7.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.7.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.8.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.8.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.8.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.8.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.8.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.8.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.8.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.8.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.9.attention_norm.weight   ||  Shape: [640]
Tensor: transformer_encoder.9.ffn.w12.weight          ||  Shape: [3424, 640]
Tensor: transformer_encoder.9.ffn.w3.weight           ||  Shape: [640, 1712]
Tensor: transformer_encoder.9.ffn_norm.weight         ||  Shape: [640]
Tensor: transformer_encoder.9.k.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.9.q.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.9.v.weight                ||  Shape: [640, 640]
Tensor: transformer_encoder.9.wo.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.10.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.10.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.10.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.10.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.10.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.10.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.10.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.10.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.11.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.11.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.11.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.11.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.11.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.11.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.11.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.11.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.12.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.12.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.12.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.12.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.12.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.12.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.12.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.12.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.13.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.13.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.13.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.13.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.13.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.13.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.13.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.13.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.14.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.14.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.14.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.14.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.14.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.14.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.14.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.14.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.15.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.15.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.15.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.15.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.15.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.15.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.15.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.15.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.16.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.16.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.16.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.16.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.16.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.16.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.16.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.16.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.17.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.17.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.17.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.17.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.17.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.17.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.17.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.17.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.18.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.18.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.18.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.18.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.18.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.18.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.18.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.18.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.19.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.19.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.19.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.19.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.19.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.19.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.19.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.19.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.20.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.20.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.20.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.20.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.20.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.20.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.20.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.20.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.21.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.21.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.21.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.21.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.21.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.21.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.21.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.21.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.22.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.22.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.22.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.22.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.22.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.22.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.22.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.22.wo.weight              ||  Shape: [640, 640]
Tensor: transformer_encoder.23.attention_norm.weight  ||  Shape: [640]
Tensor: transformer_encoder.23.ffn.w12.weight         ||  Shape: [3424, 640]
Tensor: transformer_encoder.23.ffn.w3.weight          ||  Shape: [640, 1712]
Tensor: transformer_encoder.23.ffn_norm.weight        ||  Shape: [640]
Tensor: transformer_encoder.23.k.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.23.q.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.23.v.weight               ||  Shape: [640, 640]
Tensor: transformer_encoder.23.wo.weight              ||  Shape: [640, 640]
```
