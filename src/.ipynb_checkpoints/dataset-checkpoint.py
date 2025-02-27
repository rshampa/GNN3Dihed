import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

from src.molecular_representation import MolecularGraphGenerator
from tqdm import tqdm_gui, tqdm
from src.config import MolecularGraphGeneratorConfig

from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import cProfile
import profile

"""
New Graph Representation:
    Atomic Feature Vectors
    Bond Feature Vectors
    Bond Indices

    Bond Feature Vectors (Shared)
    Angle Feature Vectors
    Angle Indices

    Angle Feature Vectors (Shared)
    Dihedral Feature Vectors
    Dihedral Indices
"""

class MolecularDataSet(Dataset):

    def __init__(self, data_path, name, config = None):
        # Path to the data file
        self.data_path = data_path
        self.name = name

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            self.data_path = ""
            self.initialize(config)
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            self.data_path = data_path
            self.initialize(config)
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def initialize(self, config = None):
        # Target Embedding / Representation
        self.targets_normalized = [] # These are the actual tasks
        self.targets_minimum = None
        self.targets_maximum = None

        # Molecule Embedding / Representation
        self.atomic_features_all_molecules = []
        self.bond_features_all_molecules = []
        self.bond_indices_all_molecules = []

        self.angle_features_all_molecules = []
        self.angle_indices_all_molecules = []

        self.dihedral_features_all_molecules = []
        self.dihedral_indices_all_molecules = []

        self.global_molecular_features = []

        # Redundancy
        self.translated_indices = [] # To check indices accounting for molecule skips

        # Dataset usage
        self.active_targets = []

        # Molecular Representation Generator
        if config is None:
            self.generator_config = MolecularGraphGeneratorConfig()
        else:
            # Add a typecheck here.
            self.generator_config = config
        self.generator = MolecularGraphGenerator(self.generator_config)

        if self.data_path == "":
            return

        # Checking if the data path is valid
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

        # Raw Table Data extracted from the csv file
        self.raw_table_data = pd.read_csv(self.data_path)

    def __len__(self):
        return len(self.targets_normalized)

    def __getitem__(self, idx):
        return torch.tensor(self.atomic_features_all_molecules[idx], dtype = torch.float), \
            torch.tensor(self.bond_features_all_molecules[idx], dtype = torch.float), \
            torch.tensor(self.angle_features_all_molecules[idx], dtype = torch.float), \
            torch.tensor(self.dihedral_features_all_molecules[idx], dtype = torch.float), \
            torch.tensor(self.global_molecular_features[idx], dtype = torch.float), \
            torch.tensor(self.bond_indices_all_molecules[idx].astype(np.longlong), dtype = torch.long), \
            torch.tensor(self.angle_indices_all_molecules[idx].astype(np.longlong), dtype = torch.long), \
            torch.tensor(self.dihedral_indices_all_molecules[idx].astype(np.longlong), dtype = torch.long), \
            torch.tensor(self.targets_normalized[idx], dtype = torch.float)

    #@profile
    def process(self):
        smiles, targets = self.seperate_smiles_and_targets()
        self.process_smiles(smiles, targets)
        #self.process_targets(targets)

#shampa: Commented out this to utilize multi-threading. Check below
   #def process_smiles(self, smiles, targets):
   #    print("Loading Molecular Representations")
   #    # Could add multithreaded execution here.
   #    for i, smile in tqdm(enumerate(smiles), total = len(smiles)):
   #        success, molecular_features = self.generator.smiles_to_molecular_representation(smile)
   #        if not success:
   #            print("Skipping molecule:", i)
   #            continue
   #        atomic_features, bond_features, bond_indices, angle_features, angle_indices, dihedral_features, dihedral_indices, global_molecular_features = molecular_features

   #        # Converting all vectors to numpy
   #        atomic_features = np.array(atomic_features, dtype = np.float64)
   #        bond_features = np.array(bond_features, dtype = np.float64)
   #        dihedral_features = np.array(dihedral_features, dtype = np.float64)

   #        angle_features = np.array(angle_features, dtype = np.float64)
   #        global_molecular_features = np.array(global_molecular_features, dtype = np.float64)
   #        bond_indices = np.array(bond_indices, dtype = "object")
   #        angle_indices = np.array(angle_indices, dtype = "object")
   #        dihedral_indices = np.array(dihedral_indices, dtype = "object")

   #        self.atomic_features_all_molecules.append(atomic_features)
   #        self.bond_features_all_molecules.append(bond_features)
   #        self.angle_features_all_molecules.append(angle_features)
   #        self.global_molecular_features.append(global_molecular_features)
   #        self.bond_indices_all_molecules.append(bond_indices)
   #        self.angle_indices_all_molecules.append(angle_indices)
   #        self.dihedral_features_all_molecules.append(dihedral_features)
   #        self.dihedral_indices_all_molecules.append(dihedral_indices)
   #        self.targets_normalized.append(targets[i])
   #        self.translated_indices.append(i)


   #    # Converting all the lists to numpy arrays
   #    self.atomic_features_all_molecules = np.array(self.atomic_features_all_molecules, dtype = "object")
   #    self.bond_features_all_molecules = np.array(self.bond_features_all_molecules, dtype = "object")
   #    self.angle_features_all_molecules = np.array(self.angle_features_all_molecules, dtype="object")
   #    self.dihedral_features_all_molecules = np.array(self.dihedral_features_all_molecules, dtype="object")
   #    self.global_molecular_features = np.array(self.global_molecular_features, dtype = np.float64)
   #    self.bond_indices_all_molecules = np.array(self.bond_indices_all_molecules, dtype = "object")
   #    self.angle_indices_all_molecules = np.array(self.angle_indices_all_molecules, dtype = "object")
   #    self.dihedral_indices_all_molecules = np.array(self.dihedral_indices_all_molecules, dtype = "object")
   #    self.targets_normalized = np.array(self.targets_normalized, dtype = np.float64)
   #    self.translated_indices = np.array(self.targets_normalized, dtype = np.uint)
#shampa: Commented out this to utilize multi-threading. Check below

#shampa: Multi-threading starts here
    def process_smiles(self, smiles, targets):
        print("Loading Molecular Representations")

        results = []  # List to store results along with their original indices

        # Function to process each smile
        def process_smile(i, smile):
            success, molecular_features = self.generator.smiles_to_molecular_representation(smile)
            if not success:
                print("Skipping molecule:", i)
                return None

            (atomic_features, bond_features, bond_indices,
             angle_features, angle_indices, dihedral_features,
             dihedral_indices, global_molecular_features) = molecular_features

            # Convert features to numpy arrays
            atomic_features = np.array(atomic_features, dtype=np.float64)
            bond_features = np.array(bond_features, dtype=np.float64)
            dihedral_features = np.array(dihedral_features, dtype=np.float64)
            angle_features = np.array(angle_features, dtype=np.float64)
            global_molecular_features = np.array(global_molecular_features, dtype=np.float64)
            bond_indices = np.array(bond_indices, dtype="object")
            angle_indices = np.array(angle_indices, dtype="object")
            dihedral_indices = np.array(dihedral_indices, dtype="object")

            return (atomic_features, bond_features, angle_features,
                    global_molecular_features, bond_indices,
                    angle_indices, dihedral_features, dihedral_indices, targets[i], i)

        # Multithreading setup
        with ThreadPoolExecutor(max_workers=14) as executor:
            futures = [executor.submit(process_smile, i, smile) for i, smile in enumerate(smiles)]

            for future in tqdm(as_completed(futures), total=len(smiles)):
                result = future.result()
                if result:
                    results.append(result)

        # Sort results based on original indices
        results.sort(key=lambda x: x[-1])  # Sort based on the last element which is the original index

        # Unpack sorted results
        for result in results:
            (atomic_features, bond_features, angle_features,
             global_molecular_features, bond_indices,
             angle_indices, dihedral_features, dihedral_indices,
             target, index) = result

            # Append results to instance variables
            self.atomic_features_all_molecules.append(atomic_features)
            self.bond_features_all_molecules.append(bond_features)
            self.angle_features_all_molecules.append(angle_features)
            self.global_molecular_features.append(global_molecular_features)
            self.bond_indices_all_molecules.append(bond_indices)
            self.angle_indices_all_molecules.append(angle_indices)
            self.dihedral_features_all_molecules.append(dihedral_features)
            self.dihedral_indices_all_molecules.append(dihedral_indices)
            self.targets_normalized.append(target)
            self.translated_indices.append(index)

        # Convert lists to numpy arrays
        self.atomic_features_all_molecules = np.array(self.atomic_features_all_molecules, dtype="object")
        self.bond_features_all_molecules = np.array(self.bond_features_all_molecules, dtype="object")
        self.angle_features_all_molecules = np.array(self.angle_features_all_molecules, dtype="object")
        self.dihedral_features_all_molecules = np.array(self.dihedral_features_all_molecules, dtype="object")
        self.global_molecular_features = np.array(self.global_molecular_features, dtype=np.float64)
        self.bond_indices_all_molecules = np.array(self.bond_indices_all_molecules, dtype="object")
        self.angle_indices_all_molecules = np.array(self.angle_indices_all_molecules, dtype="object")
        self.dihedral_indices_all_molecules = np.array(self.dihedral_indices_all_molecules, dtype="object")
        self.targets_normalized = np.array(self.targets_normalized, dtype=np.float64)
        self.translated_indices = np.array(self.translated_indices, dtype=np.uint)
#shampa: Multi-threading ends here

    def process_targets(self, targets):
        minimum_targets = np.min(targets, axis=0)
        maximum_targets = np.max(targets, axis=0)

        self.targets_normalized = targets # Due to problem directly returning this for now.#(targets - minimum_targets) / (maximum_targets - minimum_targets)

    def seperate_smiles_and_targets(self):
        """
        Seperate the smhow do you center data machin learningiles and targets from the raw_table_data
        Input: raw_table_data
        Output: smiles, targets (target is a dictionary of numpy arrays)
        """
        pass

    def save(self, path):
        # path is meant to be a folder path

        # Serialize this entire database object
        # Storing the atomic features, bond features, angle features,
        # global molecular features, bond indices, angle indices, targets_normalized,
        # targets_minimum, targets_maximum
        # using numpy.save and numpy.load

        # Make save file folder if not already there
        if not os.path.exists(path):
            os.makedirs(path)

        # Save the atomic features
        np.save(path + "atomic_features.npy", self.atomic_features_all_molecules)

        # Save the bond features
        np.save(path + "bond_features.npy", self.bond_features_all_molecules)

        # Save the angle features
        np.save(path + "angle_features.npy", self.angle_features_all_molecules)

        # Saving the dihedral features
        np.save(path + "dihedral_features.npy", self.dihedral_features_all_molecules)

        # Save the global molecular features
        np.save(path + "global_molecular_features.npy", self.global_molecular_features)

        # Save the bond indices
        np.save(path + "bond_indices.npy", self.bond_indices_all_molecules)

        # Save the angle indices
        np.save(path + "angle_indices.npy", self.angle_indices_all_molecules)

        # Save the dihedral indices
        np.save(path + "dihedral_indices.npy", self.dihedral_indices_all_molecules)

        # Save the targets
        np.save(path + "targets_normalized.npy", self.targets_normalized)

        # Save the minimum targets
        np.save(path + "targets_minimum.npy", self.targets_minimum)

        # Save the maximum targets
        np.save(path + "targets_maximum.npy", self.targets_maximum)

        # Save translated indices
        np.save(path + "translated_indices.npy", self.translated_indices)

    def load(self, path):
        # Load the atomic features
        self.atomic_features_all_molecules = np.load(path + "atomic_features.npy", allow_pickle=True)

        # Load the bond features
        self.bond_features_all_molecules = np.load(path + "bond_features.npy", allow_pickle=True)

        # Load the angle features
        self.angle_features_all_molecules = np.load(path + "angle_features.npy", allow_pickle=True)

        # Load the dihedral features
        self.dihedral_features_all_molecules = np.load(path + "dihedral_features.npy", allow_pickle=True)

        # Load the global molecular features
        self.global_molecular_features = np.load(path + "global_molecular_features.npy", allow_pickle=True)

        # Load the bond indices
        self.bond_indices_all_molecules = np.load(path + "bond_indices.npy", allow_pickle=True)

        # Load the angle indices
        self.angle_indices_all_molecules = np.load(path + "angle_indices.npy", allow_pickle=True)

        # Load the dihedral indices
        self.dihedral_indices_all_molecules = np.load(path + "dihedral_indices.npy", allow_pickle=True)

        # Load the targets
        self.targets_normalized = np.load(path + "targets_normalized.npy", allow_pickle=True)

        # Load the minimum targets
        self.targets_minimum = np.load(path + "targets_minimum.npy", allow_pickle=True)

        # Load the maximum targets
        self.targets_maximum = np.load(path + "targets_maximum.npy", allow_pickle=True)

        # Load translated indices
        self.translated_indices = np.load(path + "translated_indices.npy", allow_pickle=True)

    def __str__(self):
        return f"Dataset Name: {self.name}\nNumber of Molecules Loaded: {len(self.atomic_features_all_molecules)}"

class QM9Dataset(MolecularDataSet):

    def __init__(self, data_path):
        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "QM9")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "QM9")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return

    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets

class LogSDataset(MolecularDataSet):

    def __init__(self, data_path):

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "LogS")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "LogS")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets


class LogPDataset(MolecularDataSet):

    def __init__(self, data_path):

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "LogP")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "LogP")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets


class FreeSolvDataset(MolecularDataSet):

    def __init__(self, data_path):

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "FreeSolv")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "FreeSolv")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets


class ESOLDataset(MolecularDataSet):

    def __init__(self, data_path):

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "ESOL")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "ESOL")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets


class ToxQDataset(MolecularDataSet):

    def __init__(self, data_path):

        # Check if the current provided path is a folder
        if os.path.isdir(data_path):
            print("Loading dataset from folder")
            super().__init__("", "ToxQ")
            data_path = data_path + "/"
            self.load(data_path)

        else:
            print("Loading Raw Data and processing it")
            save_path = data_path + "/"
            data_path = data_path + ".csv"
            super().__init__(data_path, "ToxQ")
            self.process()
            # Saving the processed data
            print("Saving data.")

            # Make save path folder if it is not already there.
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.save(save_path)


        if self.data_path == "":
            return


    def seperate_smiles_and_targets(self):
        # Ignoring the first column which is the index

        # Loading the smiles from the first column
        smiles = self.raw_table_data.iloc[:, 0].values

        # Storing all other columns together in the form of a matrix
        # as targets
        targets = self.raw_table_data.iloc[:, 1:].values

        return smiles, targets
