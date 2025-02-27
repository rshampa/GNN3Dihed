from rdkit import Chem

# Some data related configuration

# This only matters if the dataset is processing data from scratch.
data_loader_explicit_hydrogens = True
data_loader_use_chirality = True
data_loader_use_stereochemistry = True

# Instead of dealing with global configuration parameters, we implement a more modular alternative using configuration classes, for
# both the graph generation process and the model.

class MolecularGraphGeneratorConfig:

    def __init__(self, explicit_hydrogens = True, use_chirality = True, use_stereochemistry = True, num_conformer_samples = 500):
        self.explicit_hydrogens = explicit_hydrogens
        self.use_chirality = use_chirality
        self.use_stereochemistry = use_stereochemistry
        self.num_conformer_samples = num_conformer_samples

        # Dictionary of all parameters and their possible values
        self.options = {
            "atom_types":  ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                            'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb',
                            'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                            'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                            'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                            'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                            'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                            'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
                            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Unknown'],
            "number_of_heavy_atoms": [0, 1, 2, 3, 4, "More than four"],
            "formal_charges": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, "Extreme"],
            "hybridization": ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
            "bond_types": [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC],
            "chirality": ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"],
            "number_of_hydrogens": [0, 1, 2, 3, 4, 5, 6, "MoreThanSix"],
            "stereochemistry": ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        }

    def __setitem__(self, key, value):
        # Should check if the key exists
        self.options[key] = value

    def __getitem__(self, key):
        # Should check if the key exists
        return self.options[key]

    def add_option(self, key, value):
        # Should check if the key exists
        self.options[key].append(value)
