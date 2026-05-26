use crate::utils::{misc::unbinpack, residue_constants, structure::affine3d::Affine3D};
use candle_core::{Device, Result, Tensor};
use candle_nn::ops;
use std::collections::HashMap;

type ArrayOrTensor = Tensor;

fn index_by_atom_name(atom37: &Tensor, atom_names: &[String], dim: i64) -> Result<Tensor> {
    let squeeze = atom_names.len() == 1;
    let atom_order: HashMap<String, usize> = residue_constants::atom_order();
    let indices: Vec<_> = atom_names
        .iter()
        .map(|name| atom_order.get(name).unwrap())
        .collect();

    let dim = if dim < 0 {
        atom37.dims().len() as i64 + dim
    } else {
        dim
    };
    let selected = atom37.select(dim as usize, &indices)?;

    if squeeze {
        selected.squeeze(dim as usize)
    } else {
        Ok(selected)
    }
}

fn infer_cbeta_from_atom37(atom37: &Tensor, l: f64, a: f64, d: f64) -> Result<Tensor> {
    let n = index_by_atom_name(atom37, &["N".to_string()], -2)?;
    let ca = index_by_atom_name(atom37, &["CA".to_string()], -2)?;
    let c = index_by_atom_name(atom37, &["C".to_string()], -2)?;

    let vec_nca = &n - &ca;
    let vec_nc = &n - &c;

    let nca = ops::normalize(&vec_nca, -1)?;
    let n = ops::normalize(&ops::cross(&vec_nc, &nca)?, -1)?;
    let m = vec![nca.clone(), ops::cross(&n, &nca)?, n];

    let d = vec![l * a.cos(), l * a.sin() * d.cos(), -l * a.sin() * d.sin()];

    let mut result = ca.clone();
    for (mi, di) in m.iter().zip(d.iter()) {
        result = result.add(&mi.mul_scalar(*di)?)?;
    }
    Ok(result)
}

fn compute_alignment_tensors(
    mobile: &Tensor,
    target: &Tensor,
    atom_exists_mask: Option<&Tensor>,
    sequence_id: Option<&Tensor>,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (mobile, target, atom_exists_mask) = if let Some(seq_id) = sequence_id {
        let mobile = unbinpack(mobile, seq_id, f32::NAN)?;
        let target = unbinpack(target, seq_id, f32::NAN)?;
        let mask = if let Some(mask) = atom_exists_mask {
            unbinpack(mask, seq_id, 0)?
        } else {
            target.isfinite()?.all_dim(-1, true)?
        };
        (mobile, target, mask)
    } else {
        (
            mobile.clone(),
            target.clone(),
            atom_exists_mask
                .cloned()
                .unwrap_or_else(|| target.isfinite()?.all_dim(-1, true)?),
        )
    };

    assert_eq!(
        mobile.shape(),
        target.shape(),
        "Batch structure shapes do not match!"
    );

    let batch_size = mobile.dims()[0];
    let mut mobile = mobile;
    let mut target = target;
    let mut atom_exists_mask = atom_exists_mask;

    if mobile.dim() == 4 {
        mobile = mobile.reshape(&[batch_size, -1, 3])?;
    }
    if target.dim() == 4 {
        target = target.reshape(&[batch_size, -1, 3])?;
    }
    if atom_exists_mask.dim() == 3 {
        atom_exists_mask = atom_exists_mask.reshape(&[batch_size, -1])?;
    }

    let num_atoms = mobile.dims()[1];

    mobile = mobile.where_cond(
        &atom_exists_mask.unsqueeze(-1)?,
        &Tensor::zeros_like(&mobile)?,
    )?;
    target = target.where_cond(
        &atom_exists_mask.unsqueeze(-1)?,
        &Tensor::zeros_like(&target)?,
    )?;

    let num_valid_atoms = atom_exists_mask.sum_dim(-1, true)?;

    let centroid_mobile = mobile
        .sum_dim(-2, true)?
        .div(&num_valid_atoms.unsqueeze(-1)?)?;
    let centroid_target = target
        .sum_dim(-2, true)?
        .div(&num_valid_atoms.unsqueeze(-1)?)?;

    centroid_mobile.where_cond(
        &(num_valid_atoms.eq(0.)?),
        &Tensor::zeros_like(&centroid_mobile)?,
    )?;
    centroid_target.where_cond(
        &(num_valid_atoms.eq(0.)?),
        &Tensor::zeros_like(&centroid_target)?,
    )?;

    let centered_mobile = mobile.sub(&centroid_mobile)?;
    let centered_target = target.sub(&centroid_target)?;

    let centered_mobile = centered_mobile.where_cond(
        &atom_exists_mask.unsqueeze(-1)?,
        &Tensor::zeros_like(&centered_mobile)?,
    )?;
    let centered_target = centered_target.where_cond(
        &atom_exists_mask.unsqueeze(-1)?,
        &Tensor::zeros_like(&centered_target)?,
    )?;

    let covariance_matrix = centered_mobile
        .transpose(-2, -1)?
        .matmul(&centered_target)?;

    let (u, _, v) = covariance_matrix.svd(false)?;
    let rotation_matrix = u.matmul(&v.transpose(-2, -1)?)?;

    Ok((
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    ))
}

fn compute_rmsd_no_alignment(
    aligned: &Tensor,
    target: &Tensor,
    num_valid_atoms: &Tensor,
    reduction: &str,
) -> Result<Tensor> {
    if !["per_residue", "per_sample", "batch"].contains(&reduction) {
        return Err(candle_core::Error::Msg(format!(
            "Unrecognized reduction: {}",
            reduction
        )));
    }

    let diff = aligned.sub(target)?;

    let mean_squared_error = if reduction == "per_residue" {
        diff.square()?
            .reshape(&[diff.dims()[0], -1, 9])?
            .mean_dim(-1, true)?
    } else {
        diff.square()?
            .sum_dim(&[1, 2], true)?
            .div(&(num_valid_atoms.squeeze(-1)?.mul_scalar(3.)?)?)?
    };

    let rmsd = mean_squared_error.sqrt()?;

    match reduction {
        "per_sample" | "per_residue" => Ok(rmsd),
        "batch" => {
            let zero_masked = rmsd.where_cond(
                &num_valid_atoms.squeeze(-1)?.eq(0.)?,
                &Tensor::zeros_like(&rmsd)?,
            )?;
            let sum = zero_masked.sum(0)?;
            let valid_count = num_valid_atoms.greater(0.)?.sum(0)?;
            Ok(sum.div(&(valid_count.add_scalar(1e-8)?))?)
        }
        _ => Err(candle_core::Error::Msg(format!(
            "Unrecognized reduction: {}",
            reduction
        ))),
    }
}

fn compute_affine_and_rmsd(
    mobile: &Tensor,
    target: &Tensor,
    atom_exists_mask: Option<&Tensor>,
    sequence_id: Option<&Tensor>,
) -> Result<(Affine3D, Tensor)> {
    let (
        centered_mobile,
        centroid_mobile,
        centered_target,
        centroid_target,
        rotation_matrix,
        num_valid_atoms,
    ) = compute_alignment_tensors(mobile, target, atom_exists_mask, sequence_id)?;

    let translation = centroid_mobile
        .neg()?
        .matmul(&rotation_matrix)?
        .add(&centroid_target)?;
    let affine = Affine3D::from_tensor_pair(
        &translation,
        &rotation_matrix.unsqueeze(-3)?.transpose(-2, -1)?,
    )?;

    let rotated_mobile = centered_mobile.matmul(&rotation_matrix)?;
    let avg_rmsd =
        compute_rmsd_no_alignment(&rotated_mobile, &centered_target, &num_valid_atoms, "batch")?;

    Ok((affine, avg_rmsd))
}

fn compute_gdt_ts_no_alignment(
    aligned: &Tensor,
    target: &Tensor,
    atom_exists_mask: Option<&Tensor>,
    reduction: &str,
) -> Result<Tensor> {
    if !["per_sample", "batch"].contains(&reduction) {
        return Err(candle_core::Error::Msg(format!(
            "Unrecognized reduction: {}",
            reduction
        )));
    }

    let atom_exists_mask = atom_exists_mask.unwrap_or(&target.isfinite()?.all_dim(-1, true)?);
    let deviation = ops::vector_norm(&(aligned.sub(target)?), 2, -1, true)?;
    let num_valid_atoms = atom_exists_mask.sum_dim(-1, false)?;

    let thresholds = [1., 2., 4., 8.];
    let mut scores = Vec::new();

    for threshold in thresholds.iter() {
        let below_threshold = deviation.less(*threshold)?;
        let valid_count = below_threshold
            .logical_and(&atom_exists_mask)?
            .sum_dim(-1, false)?
            .to_dtype(candle_core::DType::F32)?;
        scores.push(valid_count.div(&num_valid_atoms)?);
    }

    let score = scores
        .into_iter()
        .try_fold(Tensor::zeros_like(&num_valid_atoms)?, |acc, x| acc.add(&x))?
        .mul_scalar(0.25)?;

    match reduction {
        "batch" => score.mean_dim(0, false),
        "per_sample" => Ok(score),
        _ => Err(candle_core::Error::Msg(format!(
            "Unrecognized reduction: {}",
            reduction
        ))),
    }
}
