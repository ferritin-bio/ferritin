use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{embedding, VarBuilder};

pub struct EMACodebook {
    embeddings: Tensor,
    n: Tensor,
    z_avg: Tensor,
    n_codes: usize,
    embedding_dim: usize,
    need_init: bool,
    no_random_restart: bool,
    restart_thres: f64,
    freeze_codebook: bool,
    ema_decay: f64,
}

impl EMACodebook {
    pub fn new(
        vb: VarBuilder,
        n_codes: usize,
        embedding_dim: usize,
        no_random_restart: bool,
        restart_thres: f64,
        ema_decay: f64,
        device: &Device,
    ) -> Result<Self> {
        let embeddings = Tensor::randn(0f32, 1f32, (n_codes, embedding_dim), device)?;
        let n = Tensor::zeros((n_codes,), DType::F32, device)?;
        let z_avg = embeddings.clone();

        Ok(Self {
            embeddings,
            n,
            z_avg,
            n_codes,
            embedding_dim,
            need_init: true,
            no_random_restart,
            restart_thres,
            freeze_codebook: false,
            ema_decay,
        })
    }

    fn reset_parameters(&mut self) {
        // For meta init
    }

    fn tile(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let d = shape.dims()[0];
        let ew = shape.dims()[1];

        if d < self.n_codes {
            let n_repeats = (self.n_codes + d - 1) / d;
            let std = 0.01 / (ew as f64).sqrt();
            let x = x.repeat((n_repeats, 1))?;
            let noise = Tensor::randn(0f32, std as f32, x.shape(), x.device())?;
            Ok(x.add(&noise)?)
        } else {
            Ok(x.clone())
        }
    }

    fn init_embeddings(&mut self, z: &Tensor) -> Result<()> {
        self.need_init = false;
        let flat_inputs = z.reshape((-1, self.embedding_dim as i64))?;
        let y = self.tile(&flat_inputs)?;

        let indices = Tensor::randperm(y.shape()[0], y.device())?;
        let k_rand = y
            .index_select(&indices, 0)?
            .narrow(0, 0, self.n_codes as i64)?;

        self.embeddings = k_rand.clone();
        self.z_avg = k_rand;
        self.n = Tensor::ones((self.n_codes,), DType::F32, self.embeddings.device())?;
        Ok(())
    }

    pub fn forward(&mut self, z: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        if self.need_init && !self.freeze_codebook {
            self.init_embeddings(z)?;
        }

        let flat_inputs = z.reshape((-1, self.embedding_dim as i64))?;

        let squared_inputs = flat_inputs.sqr()?.sum_keepdim(1)?;
        let embeddings_t = self.embeddings.transpose(0, 1)?;
        let product = flat_inputs.matmul(&embeddings_t)?;
        let squared_embeddings = embeddings_t.sqr()?.sum_keepdim(0)?;
        let distances = squared_inputs
            .sub(&product.mul_scalar(2f64)?)?
            .add(&squared_embeddings)?;

        let encoding_indices = distances.argmin(1)?;
        let z_shape = z.shape();
        let encoding_indices = encoding_indices.reshape((z_shape[0], z_shape[1]))?;

        let embeddings = embedding(&encoding_indices, &self.embeddings)?;

        let detached_embeddings = embeddings.detach()?;
        let commitment_loss = z.mse_loss(&detached_embeddings)?.mul_scalar(0.25)?;

        if !self.freeze_codebook {
            panic!("EMA update not implemented");
        }

        let embeddings_st = embeddings.sub(z)?.detach()?.add(z)?;

        Ok((embeddings_st, encoding_indices, commitment_loss))
    }

    pub fn dictionary_lookup(&self, encodings: &Tensor) -> Result<Tensor> {
        embedding(encodings, &self.embeddings)
    }

    pub fn soft_codebook_lookup(&self, weights: &Tensor) -> Result<Tensor> {
        weights.matmul(&self.embeddings)
    }
}
