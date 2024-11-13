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
    pub fn new(atom1: i32, atom2: i32, order: BondOrder) -> Self {
        Bond {
            atom1,
            atom2,
            order,
        }
    }
    pub fn get_atom_indices(&self) -> (i32, i32) {
        (self.atom1, self.atom2)
    }
}

#[repr(u8)]
#[derive(Debug, PartialEq)]
/// BondOrder:
///
/// Enum for defining Bond orders.
/// **Note: subject to change.**
/// Needs more research on which convention to follow.
/// - [biotite](https://www.biotite-python.org/latest/apidoc/biotite.structure.BondType.html#biotite.structure.BondType)
/// - see also [cdk](http://cdk.github.io/cdk/latest/docs/api/org/openscience/cdk/Bond.html)
pub enum BondOrder {
    /// Used if the actual type is unknown
    Unset,
    /// Single bond
    Single,
    /// Double bond
    Double,
    /// Triple bond
    Triple,
    /// A quadruple bond
    Quadruple,
}

impl BondOrder {
    pub fn match_bond(bond_int: i32) -> BondOrder {
        match bond_int {
            0 => BondOrder::Unset,
            1 => BondOrder::Single,
            2 => BondOrder::Double,
            3 => BondOrder::Triple,
            4 | 5 | 6 => BondOrder::Quadruple,
            _ => {
                println!("Bond Order not found: {}", bond_int);
                panic!()
            }
        }
    }
}
