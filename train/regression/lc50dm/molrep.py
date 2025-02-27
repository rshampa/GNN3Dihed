## Know your libraries (KYL)================================
import sys
sys.path.append('../../../src')

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import dataset, molecular_representation, config, utils, model
from dataset import QM9Dataset,LogSDataset,LogPDataset,FreeSolvDataset,ESOLDataset,ToxQDataset
import numpy as np
import pandas as pd
from utils import *
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.stats import pearsonr
from tqdm import tqdm
from model import AtomAutoencoder,BondAutoencoder
from model import GNNBondAngle,GNN2D,GNN3D,GNN3DAtnON,GNN3DAtnOFF,GNN3Dihed,GNN3DConfig,GNN3DLayer,GNN3DClassifier

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=True
IPythonConsole.drawOptions.addStereoAnnotation = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc

import os
datadir="../../../data/regression/lc50dm/"
modeldir="./models/"
dataset_name="lc50dm"
# Create datadir and modeldir folders if they don't exist
if not os.path.exists(datadir):
    os.makedirs(datadir)

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


## Split your dataset (SYD)=================================
# Load dataset from CSV
dataset = pd.read_csv(datadir+dataset_name+".csv")

# Split dataset into train and test
train_dataset, test_dataset = split_dataset(dataset, 0.9)

# Write train_dataset and test_dataset to CSV files
train_dataset.to_csv(datadir+"train.csv", index=False)
test_dataset.to_csv(datadir+"test.csv", index=False)

print("Train and test datasets saved successfully.")


## Process your data (PYD)==================================
train_samples = ToxQDataset(datadir+"train")
print(train_samples)
print("===================================")
test_samples = ToxQDataset(datadir+"test")
print(test_samples)


## Know your featues (KYF)==================================
# Printing out the dimensions of all of these features with a description of what each feature is
print(f"Atomic Features: {(train_samples[0])[0].shape} - This represents the atomic features of the molecule")
print(f"Bond Features: {(train_samples[0])[1].shape} - This represents the bond features of the molecule")
print(f"Angle Features: {(train_samples[0])[2].shape} - This represents the angle features of the molecule")
print(f"Dihedral Features: {(train_samples[0])[3].shape} - This represents the dihedral features of the molecule")
print(f"Global Molecular Features: {(train_samples[0])[4].shape} - This represents the global molecular features of the molecule")
print(f"Bond Indices: {(train_samples[0])[5].shape} - This represents the bond indices of the molecule")
print(f"Angle Indices: {(train_samples[0])[6].shape} - This represents the angle indices of the molecule")
print(f"Dihedral Indices: {(train_samples[0])[7].shape} - This represents the dihedral indices of the molecule")
print(f"Target: {(train_samples[0])[8].shape} - This represents the target of the molecule")

