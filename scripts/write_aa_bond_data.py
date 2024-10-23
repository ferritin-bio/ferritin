import json
import biotite.structure as struct


# List of standard amino acids by three letter code
STANDARD_AAS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]

# Dictionary to store all bond information
aa_bonds = {}

# Collect bond info for each amino acid
for aa in STANDARD_AAS:
    try:

        bonds = struct.info.bonds_in_residue(aa)
        bond_list = []
        for atoms, bond_type_int in sorted(bonds.items()):
            atom1, atom2 = sorted(atoms)
            # print(f"{atom1:3} + {atom2:3} -> {BondType(bond_type_int).name}")
            bond_list.append([atom1, atom2, bond_type_int])
        aa_bonds[aa] = bond_list
    except Exception as e:
        print(f"Error processing {aa}: {str(e)}")
        aa_bonds[aa] = None

# Save to JSON file
with open('data/amino_acid_bonds.json', 'w') as f:
    json.dump(aa_bonds, f, indent=2)

# Print summary
print("\nBond counts:")
for aa in STANDARD_AAS:
    if aa_bonds[aa]:
        print(f"{aa}: {len(aa_bonds[aa])} bonds")
    else:
        print(f"{aa}: Error retrieving bonds")
