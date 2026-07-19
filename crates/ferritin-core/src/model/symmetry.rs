//! Symmetry operators and zero-copy assembly definitions.

/// Row-major homogeneous 4×4 affine matrix.
pub type Mat4 = [[f32; 4]; 4];

/// Identity affine transform.
pub const IDENTITY_MAT4: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// A named affine transform applied lazily to model coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryOperator {
    pub label: String,
    pub matrix: Mat4,
}

impl SymmetryOperator {
    pub fn identity() -> Self {
        Self {
            label: "identity".to_string(),
            matrix: IDENTITY_MAT4,
        }
    }

    pub fn new(label: impl Into<String>, matrix: Mat4) -> Self {
        Self {
            label: label.into(),
            matrix,
        }
    }

    /// Apply the affine transform to a Cartesian point.
    pub fn apply(&self, point: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = point;
        [
            self.matrix[0][0] * x
                + self.matrix[0][1] * y
                + self.matrix[0][2] * z
                + self.matrix[0][3],
            self.matrix[1][0] * x
                + self.matrix[1][1] * y
                + self.matrix[1][2] * z
                + self.matrix[1][3],
            self.matrix[2][0] * x
                + self.matrix[2][1] * y
                + self.matrix[2][2] * z
                + self.matrix[2][3],
        ]
    }

    /// Compose two operators. The returned operator applies `other` first,
    /// followed by `self`.
    pub fn compose(&self, other: &Self, label: impl Into<String>) -> Self {
        let mut matrix = [[0.0; 4]; 4];
        for (row, values) in matrix.iter_mut().enumerate() {
            for (column, value) in values.iter_mut().enumerate() {
                *value = (0..4)
                    .map(|k| self.matrix[row][k] * other.matrix[k][column])
                    .sum();
            }
        }
        Self::new(label, matrix)
    }
}

impl Default for SymmetryOperator {
    fn default() -> Self {
        Self::identity()
    }
}

/// One transformed subset in a biological or crystallographic assembly.
#[derive(Clone, Debug, PartialEq)]
pub struct AssemblyUnit {
    /// Canonical `label_asym_id` values included in this unit.
    pub asym_ids: Vec<String>,
    pub operator: SymmetryOperator,
}

/// A named collection of transformed model subsets.
#[derive(Clone, Debug, PartialEq)]
pub struct Assembly {
    pub id: String,
    pub details: Option<String>,
    pub units: Vec<AssemblyUnit>,
}

/// Crystallographic metadata from the mmCIF `_cell` and `_symmetry` categories.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrystalSymmetry {
    pub space_group_name: Option<String>,
    pub cell: Option<[f32; 6]>,
    pub operators: Vec<SymmetryOperator>,
}

/// Symmetry and assembly metadata associated with a model topology.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymmetryData {
    pub assemblies: Vec<Assembly>,
    pub crystal: CrystalSymmetry,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_applies_translation() {
        let mut matrix = IDENTITY_MAT4;
        matrix[0][3] = 10.0;
        matrix[2][3] = -2.0;
        let operator = SymmetryOperator::new("translated", matrix);
        assert_eq!(operator.apply([1.0, 2.0, 3.0]), [11.0, 2.0, 1.0]);
    }

    #[test]
    fn operator_composition_preserves_application_order() {
        let mut translate = IDENTITY_MAT4;
        translate[0][3] = 2.0;
        let mut scale = IDENTITY_MAT4;
        scale[0][0] = 3.0;
        let combined = SymmetryOperator::new("scale", scale)
            .compose(&SymmetryOperator::new("translate", translate), "combined");
        assert_eq!(combined.apply([1.0, 0.0, 0.0]), [9.0, 0.0, 0.0]);
    }
}
