// Note: this is currently a direct Python --> Rust translation using Torch.
// I'd like to move this to Candle

use candle_core::{Result, Tensor, D};
use candle_nn::encoding::one_hot;
use ferritin_core::AtomCollection;
use std::collections::HashMap;

pub fn get_nearest_neighbours(
    cb: &Tensor,
    mask: &Tensor,
    y: &Tensor,
    y_t: &Tensor,
    y_m: &Tensor,
    number_of_ligand_atoms: i64,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let device = cb.device();
    let mask_cby = mask.unsqueeze(1)?.broadcast_mul(&y_m.unsqueeze(0)?)?; // [A,B]
    let l2_ab = ((cb.unsqueeze(1)? - y.unsqueeze(0)?).powf(2.0)?)?.sum_keepdim(D::Minus1)?;
    let l2_ab = l2_ab.broadcast_mul(&mask_cby)? + (mask_cby.neg()?.add(1.0)?)?.mul(1000.0)?;

    let nn_idx = l2_ab
        .argsort(D::Minus1, false)?
        .narrow(D::Minus1, 0, number_of_ligand_atoms)?;
    let l2_ab_nn = l2_ab.gather(&nn_idx, 1)?;
    let d_ab_closest = l2_ab_nn.narrow(D::Minus1, 0, 1)?.sqrt()?;

    let y_r = y.unsqueeze(0)?.expand((cb.dim(0)?, y.dim(0)?, y.dim(1)?))?;
    let y_t_r = y_t.unsqueeze(0)?.expand((cb.dim(0)?, y_t.dim(0)?))?;
    let y_m_r = y_m.unsqueeze(0)?.expand((cb.dim(0)?, y_m.dim(0)?))?;

    let y_tmp = y_r.gather(
        &nn_idx
            .unsqueeze(D::Minus1)?
            .expand((cb.dim(0)?, number_of_ligand_atoms, 3))?,
        1,
    )?;
    let y_t_tmp = y_t_r.gather(&nn_idx, 1)?;
    let y_m_tmp = y_m_r.gather(&nn_idx, 1)?;

    let mut y = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms, 3), DType::F32, device)?;
    let mut y_t = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms), DType::I32, device)?;
    let mut y_m = Tensor::zeros((cb.dim(0)?, number_of_ligand_atoms), DType::I32, device)?;

    let num_nn_update = y_tmp.dim(1)?;
    y.narrow(1, 0, num_nn_update)?.copy_(&y_tmp)?;
    y_t.narrow(1, 0, num_nn_update)?.copy_(&y_t_tmp)?;
    y_m.narrow(1, 0, num_nn_update)?.copy_(&y_m_tmp)?;

    Ok((y, y_t, y_m, d_ab_closest))
}

pub fn featurize(
    input_dict: &HashMap<String, Tensor>,
    cutoff_for_score: f32,
    use_atom_context: bool,
    number_of_ligand_atoms: i64,
    model_type: &str,
) -> Result<HashMap<String, Tensor>> {
    let mut output_dict = HashMap::new();
    if model_type == "ligand_mpnn" {
        let mask = input_dict.get("mask").unwrap();
        let y = input_dict.get("Y").unwrap();
        let y_t = input_dict.get("Y_t").unwrap();
        let y_m = input_dict.get("Y_m").unwrap();
        let n = input_dict.get("X").unwrap().slice(1, 0, 1, 1)?;
        let ca = input_dict.get("X").unwrap().slice(1, 1, 2, 1)?;
        let c = input_dict.get("X").unwrap().slice(1, 2, 3, 1)?;
        let b = &ca - &n;
        let c = &c - &ca;
        let a = b.cross(&c)?;
        let cb = &(-0.58273431 * &a + 0.56802827 * &b - 0.54067466 * &c) + &ca;
        let (y, y_t, y_m, d_xy) =
            get_nearest_neighbours(&cb, mask, y, y_t, y_m, number_of_ligand_atoms)?;
        let mask_xy = (&d_xy.lt(cutoff_for_score)? * mask * &y_m.slice(1, 0, 1, 1)?)?;
        output_dict.insert("mask_XY".to_string(), mask_xy.unsqueeze(0)?);
        if input_dict.contains_key("side_chain_mask") {
            output_dict.insert(
                "side_chain_mask".to_string(),
                input_dict.get("side_chain_mask").unwrap().unsqueeze(0)?,
            );
        }
        output_dict.insert("Y".to_string(), y.unsqueeze(0)?);
        output_dict.insert("Y_t".to_string(), y_t.unsqueeze(0)?);
        output_dict.insert("Y_m".to_string(), y_m.unsqueeze(0)?);
        if !use_atom_context {
            output_dict.insert("Y_m".to_string(), (&output_dict["Y_m"] * 0.0)?);
        }
    } else if model_type == "per_residue_label_membrane_mpnn"
        || model_type == "global_label_membrane_mpnn"
    {
        output_dict.insert(
            "membrane_per_residue_labels".to_string(),
            input_dict
                .get("membrane_per_residue_labels")
                .unwrap()
                .unsqueeze(0)?,
        );
    }

    let r_idx = input_dict.get("R_idx").unwrap();
    let mut r_idx_list = Vec::new();
    let mut count = 0;
    let mut r_idx_prev = -100000;
    for &r_idx_val in r_idx.iter::<i64>()?.iter() {
        if r_idx_prev == r_idx_val {
            count += 1;
        }
        r_idx_list.push(r_idx_val + count);
        r_idx_prev = r_idx_val;
    }
    let r_idx_renumbered = Tensor::from_slice(&r_idx_list, r_idx.device())?;
    output_dict.insert("R_idx".to_string(), r_idx_renumbered.unsqueeze(0)?);
    output_dict.insert("R_idx_original".to_string(), r_idx.unsqueeze(0)?);
    output_dict.insert(
        "chain_labels".to_string(),
        input_dict.get("chain_labels").unwrap().unsqueeze(0)?,
    );
    output_dict.insert("S".to_string(), input_dict.get("S").unwrap().unsqueeze(0)?);
    output_dict.insert(
        "chain_mask".to_string(),
        input_dict.get("chain_mask").unwrap().unsqueeze(0)?,
    );
    output_dict.insert(
        "mask".to_string(),
        input_dict.get("mask").unwrap().unsqueeze(0)?,
    );

    output_dict.insert("X".to_string(), input_dict.get("X").unwrap().unsqueeze(0)?);

    if input_dict.contains_key("xyz_37") {
        output_dict.insert(
            "xyz_37".to_string(),
            input_dict.get("xyz_37").unwrap().unsqueeze(0)?,
        );
        output_dict.insert(
            "xyz_37_m".to_string(),
            input_dict.get("xyz_37_m").unwrap().unsqueeze(0)?,
        );
    }
    Ok(output_dict)
}
