/// Bond
///
/// Struct for creating Bonds of type [BondOrder]
///
#[derive(Debug, PartialEq)]
pub struct Bond {
    atom1: i32,
    atom2: i32,
    order: BondOrder,
    // id
    // stereo
    // unique_id
    // has_setting
}

impl Bond {
    pub fn new(atom1: i32, atom2: i32, order: impl Into<BondOrder>) -> Self {
        Bond {
            atom1,
            atom2,
            order: order.into(),
        }
    }
    pub fn get_atom_indices(&self) -> (i32, i32) {
        (self.atom1, self.atom2)
    }
}

/// BondOrder:
///
/// Enum for defining Bond orders.
/// **Note: subject to change.**
/// Needs more research on which convention to follow.
/// - [biotite](https://www.biotite-python.org/latest/apidoc/biotite.structure.BondType.html#biotite.structure.BondType)
/// - see also [cdk](http://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/Bond.html)
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BondOrder {
    Unset = 0,
    Single = 1,
    Double = 2,
    Triple = 3,
    Quadruple = 4,
}

impl From<i32> for BondOrder {
    fn from(bond_int: i32) -> Self {
        match bond_int {
            0 => BondOrder::Unset,
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            _ => BondOrder::Quadruple,
        }
    }
}
