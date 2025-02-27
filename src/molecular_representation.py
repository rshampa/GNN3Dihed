import numpy as np

from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

from config import *
from utils import *

import time

# It would make more sense if this was in the utils.py file (probably).
class Graph:

  def __init__(self, nodes, edge_indices, edges):
    if not nodes:
      self.nodes = []
      self.indices = []
      self.edges = []
      return

    self.nodes = nodes

    if not edge_indices or not edges:
      self.indices = []
      self.edges = []
      return

    self.indices = edge_indices
    self.edges = edges

  def add_node(self, node):
    self.nodes.append(node)

  def add_edge(self, edge_index, edge):
    self.indices.append(edge_index)
    self.edges.append(edge)

  def get_marginal_graph(self, edge_index_generator, edge_generator, args):
    # Making a graph with no edges but that has the nodes as the edges of the current graph
    marginal_graph = Graph(self.edges, None, None)

    #print("Making Marginal Graph", time.strftime("%H:%M:%S", time.localtime()))
    edge_indices = edge_index_generator(self.indices)
    #print("Finished making edge indices, current time: ", time.strftime("%H:%M:%S", time.localtime()))
    args.append(edge_indices)
    edges = edge_generator(args)
    #print("Finished making edges, current time: ", time.strftime("%H:%M:%S", time.localtime()))


    for edge_index, edge in zip(edge_indices, edges):
      marginal_graph.add_edge(edge_index, edge)

    return marginal_graph

  def print(self):
    print(f"Nodes: {self.nodes}")
    print(f"Edge Indices: {self.indices}")
    print(f"Edges: {self.edges}")


# Must refactor the name of the class below to MolecularGraphGenerator
class MolecularGraphGenerator:
  def __init__(self, config : MolecularGraphGeneratorConfig):
    print("Initializing Molecular Representation Generator")
    self.config = config
    self.generator = MakeGenerator(("rdkit2dnormalized",))

  def smiles_to_molecular_representation(self, smiles):
        """
        Convert SMILES to a molecule representation
        I.E A collection of the following vectors
            1. Atomic Feature Vectors
            2. Bond Feature Vectors
            3. Bond Indices
            4. Angle Feature Vectors
            5. Angle Indices
            6. Dihedral Feature Vectors
            7. Dihedral Indices
            8. Global Molecular Features
        """

        #print("Processing SMILES, current time: ", time.strftime("%H:%M:%S", time.localtime()))

        global_molecular_features = np.asarray(self.generator.process(smiles), dtype = np.float64)

        if not global_molecular_features[0]: # First value tells us if the generation of successful or not
            print(f"Failed to generate global molecular features for molecule: {smiles}")
            return False, []

        global_molecular_features = global_molecular_features[1:] # Ignore first flag element.

        # Checking if feature generation was successful
        if np.isnan(global_molecular_features).any():
            print(f"NaN value generated in global molecular features for molecule: {smiles}")
            return False, []

        #print("Global Molecular Features Made., current time: ", time.strftime("%H:%M:%S", time.localtime()))

        molecule = Chem.MolFromSmiles(smiles)

        # Adding hydrogens if needed
        if data_loader_explicit_hydrogens:
            molecule = Chem.AddHs(molecule)

        #print("Hydrogens Added, current time: ", time.strftime("%H:%M:%S", time.localtime()))

        # Using self.mol2data to get the atomic, bond and angle features
        success, molecular_features = self.mol2data(molecule)

        if not success:
          return False, []

        atomic_features, bond_features, bond_indices, angle_features, angle_indices, dihedral_features, dihedral_indices = molecular_features

        return True, [atomic_features, bond_features, bond_indices, angle_features, angle_indices, dihedral_features, dihedral_indices, global_molecular_features]

  def mol2data(self, mol):
    try:
      data = self.generator.process(mol)

      #print("Making First Graph, current time: ", time.strftime("%H:%M:%S", time.localtime()))
      # Computing atomic features
      atomic_vectors = [self.get_atomic_features(atom) for atom in mol.GetAtoms()]

      # Computing bond features
      bond_indices  = []
      bond_vectors = []

      for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # We only get bonds in one direction and get the reverse duplicates later
        bond_indices.append([i, j])
        bond_vectors.append(self.get_bond_features(bond))

      # We now add the reverse duplicates
      reverse_duplicates = []
      reverse_duplicate_bond_vectors = []
      for i, bond in enumerate(bond_indices):
        reverse_duplicates.append([bond[1], bond[0]])
        reverse_duplicate_bond_vectors.append(bond_vectors[i].copy())

      bond_indices += reverse_duplicates
      bond_vectors += reverse_duplicate_bond_vectors

      # We now make the primary graph
      graph = Graph(atomic_vectors, bond_indices, bond_vectors)

      #print("First (Atom-Bond Graph) made., current time: ", time.strftime("%H:%M:%S", time.localtime()))

      def geometry_edge_index_generator(edge_indices):
        half_point = int(len(edge_indices) / 2)

        new_edge_indices = []

        """for i, edge_index_i in enumerate(edge_indices[0:half_point]):
          for j, edge_index_j in enumerate(edge_indices[0:half_point]):
            if (edge_index_i[1] == edge_index_j[0] and edge_index_i[0] != edge_index_j[1]):
              new_edge_indices.append([i, j])

        # Adding reverse duplicates
        reverse_duplicates = []
        for edge in new_edge_indices:
          reverse_duplicates.append([edge[1] + half_point, edge[0] + half_point])

        new_edge_indices += reverse_duplicates"""

        for i, edge_index_i in enumerate(edge_indices):
          for j, edge_index_j in enumerate(edge_indices):
            if (edge_index_i[1] == edge_index_j[0] and edge_index_i[0] != edge_index_j[1]):
              new_edge_indices.append([i, j])

        return new_edge_indices

      def get_angle(args):

        mol, bond_indices, bond_pairs = args

        angles = []

        #AllChem.Compute2DCoords(mol)
        #AllChem.EmbedMultipleConfs(mol, conformer_samples)#, randomSeed=0xf00d)
        params = Chem.rdDistGeom.srETKDGv3()
        params.randomSeed = 12412
        params.clearConfs = True
        AllChem.EmbedMultipleConfs(mol, self.config.num_conformer_samples, params=params)

        for bond_pair in bond_pairs:
          # Extracting the atoms from the bond indices
          atom1 = bond_indices[bond_pair[0]][0]
          atom2 = bond_indices[bond_pair[0]][1]
          atom3 = bond_indices[bond_pair[1]][1]

          angle = []

          for i in range(self.config.num_conformer_samples):
            conformer = mol.GetConformer(i)
            #angle += (Chem.rdMolTransforms.GetAngleDeg(conformer, atom1, atom2, atom3) - 180) / 180;
            angle_per_conf = Chem.rdMolTransforms.GetAngleDeg(conformer, atom1, atom2, atom3)
            angle.append(angle_per_conf)

          #===========================================#
          """ Calculate mean (dihedral) angle within the bin that has the maximum count in its histogram"""
          n, bins_edges = np.histogram(angle, bins=36)
          max_count_bin = np.argmax(n)
          bin_left_edge = bins_edges[max_count_bin]
          bin_right_edge = bins_edges[max_count_bin + 1]
          mean_angle = np.mean([angle_elem for angle_elem in angle if bin_left_edge <= angle_elem < bin_right_edge])
          angles.append(mean_angle)
          #===========================================#
          #angles.append(angle / conformer_samples)

        return angles

      def get_dihedral_angle(args):

        mol, bond_indices, angle_indices, angle_pairs = args

        dihedral_angles = []

        #AllChem.EmbedMultipleConfs(mol, conformer_samples)#, randomSeed=0xef00d)
        params = Chem.rdDistGeom.srETKDGv3()
        params.randomSeed = 12412
        params.clearConfs = True
        AllChem.EmbedMultipleConfs(mol, self.config.num_conformer_samples, params=params)

        for angle_pair in angle_pairs:
          # Extracting the 3 bonds
          bond1 = bond_indices[angle_indices[angle_pair[0]][0]]
          bond2 = bond_indices[angle_indices[angle_pair[0]][1]]
          bond3 = bond_indices[angle_indices[angle_pair[1]][1]]

          # Asserting that the bonds are connected
          assert bond1[1] == bond2[0] and bond2[1] == bond3[0]

          # Extracting the 4 atoms from these bonds
          atom1 = bond1[0]
          atom2 = bond1[1]
          atom3 = bond2[1]
          atom4 = bond3[1]

          dihedral_angle = []

          for i in range(self.config.num_conformer_samples):
            conformer = mol.GetConformer(i)
            #dihedral_angle += (Chem.rdMolTransforms.GetDihedralDeg(conformer, atom1, atom2, atom3, atom4) - 180) / 180
            dihedral_angle_per_conf = Chem.rdMolTransforms.GetDihedralDeg(conformer, atom1, atom2, atom3, atom4)
            dihedral_angle.append(dihedral_angle_per_conf)

          #===========================================#
          """ Calculate mean (dihedral) angle within the bin that has the maximum count in its histogram"""
          n, bins_edges = np.histogram(dihedral_angle, bins=36)
          max_count_bin = np.argmax(n)
          bin_left_edge = bins_edges[max_count_bin]
          bin_right_edge = bins_edges[max_count_bin + 1]
          mean_dihed = np.mean([dihed for dihed in dihedral_angle if bin_left_edge <= dihed < bin_right_edge])
          dihedral_angles.append(mean_dihed)
          #===========================================#
          #dihedral_angles.append(dihedral_angle / conformer_samples)

        return dihedral_angles

      # Getting bond-angle graph
      bond_angle_graph = graph.get_marginal_graph(geometry_edge_index_generator, get_angle, [mol, bond_indices])
      #print("Bond Angle Graph Made, current time: ", time.strftime("%H:%M:%S", time.localtime()))


      # Getting angle-dihedral graph
      angle_dihedral_graph = bond_angle_graph.get_marginal_graph(geometry_edge_index_generator, get_dihedral_angle, [mol, bond_indices, bond_angle_graph.indices])
      #print("Angle Dihedral Graph Made, current time: ", time.strftime("%H:%M:%S", time.localtime()))


      # atomic_features, bond_features, bond_indices, angle_features, angle_indices, dihedral_features, dihedral_indices
      return True, [atomic_vectors, bond_vectors, bond_indices, bond_angle_graph.edges, bond_angle_graph.indices, angle_dihedral_graph.edges, angle_dihedral_graph.indices]

    except:
      return False, []

  def get_atomic_features(self, atom):

        # Listing options for various features that
        # can be one-hot encoded.


        # One hot encoding type of atom
        # suffix enc stands for one hot encoded.
        atom_type_enc = one_hot_encode(str(atom.GetSymbol()), self.config["atom_types"])
        num_heavy_neighbors_enc = one_hot_encode(int(atom.GetDegree()), self.config["number_of_heavy_atoms"])
        formal_charge_enc = one_hot_encode(int(atom.GetFormalCharge()), self.config["formal_charges"])
        hybridization_enc = one_hot_encode(str(atom.GetHybridization()), self.config["hybridization"])
        is_in_ring = [int(atom.IsInRing())]
        is_aromatic_enc = [int(atom.GetIsAromatic())]
        atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]

        # Concatinating all features into one vector
        atom_feature_vector = atom_type_enc + num_heavy_neighbors_enc + \
                            formal_charge_enc + hybridization_enc + \
                            is_in_ring + is_aromatic_enc + atomic_mass_scaled + \
                            vdw_radius_scaled + covalent_radius_scaled

        # Optional additional features

        if data_loader_use_chirality == True:
            chirality_type_enc = one_hot_encode(str(atom.GetChiralTag()), self.config["chirality"])
            atom_feature_vector += chirality_type_enc

        if data_loader_explicit_hydrogens == True:
            n_hydrogens_enc = one_hot_encode(int(atom.GetTotalNumHs()), self.config["number_of_hydrogens"])
            atom_feature_vector += n_hydrogens_enc

        return atom_feature_vector # Change to torch tensor ?

  def get_bond_features(self, bond):

      # Similar format to above function
      # except that it is for bond featurization


      bond_type_enc = one_hot_encode(bond.GetBondType(), self.config["bond_types"])
      bond_is_conj_enc = [int(bond.GetIsConjugated())]
      bond_is_in_ring_enc = [int(bond.IsInRing())]

      bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

      if data_loader_use_stereochemistry == True:
          stereo_type_enc = one_hot_encode(str(bond.GetStereo()), self.config["stereochemistry"])
          bond_feature_vector += stereo_type_enc

      return np.array(bond_feature_vector)
