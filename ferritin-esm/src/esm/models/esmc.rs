use candle_core::{DType, Device, Module, Result, Tensor};

#[derive(Debug)]
struct ESMCOutput {
    sequence_logits: Tensor,
    embeddings: Option<Tensor>,
}

struct ESMC {
    embed: candle_nn::Embedding,
    transformer: TransformerStack,
    sequence_head: RegressionHead,
    tokenizer: EsmSequenceTokenizer,
}

impl ESMC {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        tokenizer: EsmSequenceTokenizer,
    ) -> Result<Self> {
        Ok(Self {
            embed: candle_nn::embedding(64, d_model, Default::default())?,
            transformer: TransformerStack::new(d_model, n_heads, None, n_layers, 0)?,
            sequence_head: RegressionHead::new(d_model, 64)?,
            tokenizer,
        })
    }

    pub fn from_pretrained(model_name: impl Into<String>, device: Option<Device>) -> Result<Self> {
        let device = device.unwrap_or(Device::cuda_if_available()?);
        let model = load_local_model(&model_name.into(), &device)?;
        if device.is_cuda() {
            model.to_dtype(DType::BF16)?;
        }
        Ok(model)
    }

    pub fn forward(
        &self,
        sequence_tokens: Option<&Tensor>,
        sequence_id: Option<&Tensor>,
    ) -> Result<ESMCOutput> {
        let sequence_id = sequence_id
            .unwrap_or({ &(sequence_tokens.unwrap().eq(self.tokenizer.pad_token_id)?)? });

        let x = self.embed.forward(sequence_tokens.unwrap())?;
        let (x, _) = self.transformer.forward(&x, Some(sequence_id))?;
        let sequence_logits = self.sequence_head.forward(&x)?;

        Ok(ESMCOutput {
            sequence_logits,
            embeddings: Some(x),
        })
    }

    pub fn encode(&self, input: &ESMProtein) -> Result<ESMProteinTensor> {
        let sequence_tokens = if let Some(seq) = &input.sequence {
            Some(encoding::tokenize_sequence(seq, &self.tokenizer, true)?)
        } else {
            None
        };

        Ok(ESMProteinTensor::new(sequence_tokens)?.to_device(&self.device())?)
    }

    pub fn decode(&self, input: &ESMProteinTensor) -> Result<ESMProtein> {
        let sequence = input.sequence.as_ref().ok_or("Missing sequence")?;
        let sequence = decode_sequence(&sequence.slice(1..-1)?, &self.tokenizer)?;
        Ok(ESMProtein::new(Some(sequence)))
    }

    pub fn logits(&self, input: &ESMProteinTensor, config: &LogitsConfig) -> Result<LogitsOutput> {
        let input = if !input.is_batched() {
            BatchedESMProteinTensor::from_protein_tensor(input)?
        } else {
            input.clone()
        };

        candle_core::no_grad(|| {
            let output = self.forward(Some(&input.sequence), None)?;

            Ok(LogitsOutput {
                logits: ForwardTrackData {
                    sequence: if config.sequence {
                        Some(output.sequence_logits)
                    } else {
                        None
                    },
                },
                embeddings: if config.return_embeddings {
                    output.embeddings
                } else {
                    None
                },
            })
        })
    }
}
