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
datadir="../../../data/classification/hdat/"
modeldir="./models/"
dataset_name="hdat"
# Create datadir and modeldir folders if they don't exist
if not os.path.exists(datadir):
    os.makedirs(datadir)

if not os.path.exists(modeldir):
    os.makedirs(modeldir)

import time
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


## Split your dataset (SYD)================================
# Load dataset from CSV
dataset = pd.read_csv(datadir+dataset_name+".csv")

# Split dataset into train and test
train_dataset, test_dataset = split_dataset(dataset, 0.9)

# Write train_dataset and test_dataset to CSV files
train_dataset.to_csv(datadir+"train.csv", index=False)
test_dataset.to_csv(datadir+"test.csv", index=False)

#print("Train and test datasets saved successfully.")


## Know your correlations (KYC)===========================
# Calculate R^2
def calculate_r_squared(predictions, targets):
    ss_tot = np.sum((targets - np.mean(targets))**2)
    ss_res = np.sum((targets - predictions)**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


## Know your ROC curves (KYR)===========================
def plot_roc_curves(fpr, tpr, auc, file_path):
    """Plot ROC curves for all epochs."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, linestyle='-', label=f'AUC = {auc:.4f}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
   #plt.show()


## Know your modules (KYM)================================
class CustomMoleculeDataset(Dataset):
    """A custom dataset class to work DataLoader"""
    def __init__(self, samples, atom_autoencoder, bond_autoencoder, device):
        self.samples = samples
        self.atom_autoencoder = atom_autoencoder
        self.bond_autoencoder = bond_autoencoder
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        processed_data = process_molecule(sample, self.atom_autoencoder, self.bond_autoencoder, self.device)

        if processed_data is None:
            return (None,) * 9

        if len(processed_data) != 9:
            raise ValueError(f"Unexpected number of elements: {len(processed_data)}")


        return processed_data



def process_molecule(molecule, atom_autoencoder, bond_autoencoder, device):
    """Separate input features and target"""
    if len(molecule) < 9:
        return None

    # Extracting features and target (9th element (index 8) is the target)
    target = molecule[8]
    molecule_data = [mol_elem for mol_elem in molecule[:8]]

    atom_features = molecule_data[0]
    bond_features = molecule_data[1]

    if len(atom_features) == 0 or len(bond_features) == 0:
        return None

   #molecule_data = [item.to(device) for item in molecule_data]
    molecule_data = [item.to(device).detach() for item in molecule_data]
    # Applying encoders
    molecule_data[0] = atom_autoencoder.encode(molecule_data[0])
    molecule_data[1] = bond_autoencoder.encode(molecule_data[1])


    # Moving data to the device in the training loop
    return molecule_data[0], molecule_data[1], molecule_data[2], molecule_data[3], molecule_data[4], molecule_data[5], molecule_data[6], molecule_data[7], target





def collate_fn(batch):
    """Custom collate function for batching molecular data."""


    # Filter out None values
    batch = [item for item in batch if item[0] is not None]

    # If all samples were skipped, return empty tensors
    if not batch:
        return (torch.empty(0),) * 8 + (torch.empty(0),)

    # Extract components from batch
    atomic_features = [item[0] for item in batch]
    bond_features = [item[1] for item in batch]
    angle_features = [item[2] for item in batch]
    dihedral_features = [item[3] for item in batch]
    global_molecular_features = [item[4] for item in batch]
    bond_indices = [item[5] for item in batch]
    angle_indices = [item[6] for item in batch]
    dihedral_indices = [item[7] for item in batch]
    targets = [item[8] for item in batch]

    # Determine the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Convert all tensors to detach and ensure they are on the same device
    def detach_and_to_device(tensors, device):
        return [tensor.detach().to(device) for tensor in tensors]

    device = atomic_features[0].device if atomic_features else 'cpu'
    atomic_features = detach_and_to_device(atomic_features, device)
    bond_features = detach_and_to_device(bond_features, device)
    angle_features = detach_and_to_device(angle_features, device)
    dihedral_features = detach_and_to_device(dihedral_features, device)
    global_molecular_features = detach_and_to_device(global_molecular_features, device)
    targets = detach_and_to_device(targets, device)

    # Pad sequences and stack
    def pad_sequences(sequences):
        lengths = [seq.size(0) for seq in sequences]
        max_len = max(lengths, default=0)
        return pad_sequence(sequences, batch_first=True, padding_value=0), max_len

    atomic_features, _ = pad_sequences(atomic_features)
    bond_features, max_bond_len = pad_sequences(bond_features)
    angle_features, max_angle_len = pad_sequences(angle_features)
    dihedral_features, max_dihedral_len = pad_sequences(dihedral_features)
    global_molecular_features = torch.stack(global_molecular_features)
    targets = torch.stack(targets)

    # Pad indices
    def pad_indices(indices_list, max_len):
        padded_indices = torch.full((len(indices_list), max_len, 2), -1, dtype=torch.long)
        for i, indices in enumerate(indices_list):
            if indices.size(0) > 0:
                end = min(indices.size(0), max_len)
                padded_indices[i, :end] = indices[:end]
        return padded_indices

    bond_indices = pad_indices(bond_indices, max_bond_len)
    angle_indices = pad_indices(angle_indices, max_angle_len)
    dihedral_indices = pad_indices(dihedral_indices, max_dihedral_len)

    # Create a batch and move to device
    batch_tensors = (
        atomic_features, bond_features, angle_features, dihedral_features,
        global_molecular_features, bond_indices, angle_indices, dihedral_indices,
        targets
    )

    # Move all tensors to the device
    batch_tensors = tuple(tensor.to(device) for tensor in batch_tensors)

    return batch_tensors




def train_gnn3d_model(gnn3d, train_samples, val_samples, test_samples, atom_autoencoder, bond_autoencoder, n_epochs=10, printstep=10, save_dir="./models/", dataset_name="logs", batch_size=16, prefetch_factor=2, num_workers=0, random_state=None):
    """Train GNN3D with AUC-ROC"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    mse_loss_fn = torch.nn.CrossEntropyLoss()
    gnn_optimizer = torch.optim.Adam(gnn3d.parameters(), lr=1e-3, weight_decay = 0.001)

    dataloader_train = DataLoader(
        CustomMoleculeDataset(train_samples, atom_autoencoder, bond_autoencoder, device=device),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    dataloader_val = DataLoader(
        CustomMoleculeDataset(val_samples, atom_autoencoder, bond_autoencoder, device=device),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    dataloader_test = DataLoader(
        CustomMoleculeDataset(test_samples, atom_autoencoder, bond_autoencoder, device=device),
        batch_size=batch_size,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn
    )

    gnn3d.to(device)
    start_time = time.time()

    avg_rmse_losses_train = []
    epoch_auc_val = []

    for epoch_i in range(n_epochs):
        # Training
        avg_rmse_loss_train = 0
        total_samples_train = 0
        skipped_samples_train = 0

        for batch in tqdm(dataloader_train, desc='Training samples'):

            # Filter out None values
            batch = [item for item in batch if item is not None]
            if not batch:
                skipped_samples_train += len(batch)
                continue

            atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices, targets = batch

            valid_indices = [i for i, target in enumerate(targets) if target is not None]
            if not valid_indices:
                skipped_samples_train += len(targets)
                continue
            # Only retain valid samples
            atomic_features = atomic_features[valid_indices]
            bond_features = bond_features[valid_indices]
            angle_features = angle_features[valid_indices]
            dihedral_features = dihedral_features[valid_indices]
            global_molecular_features = global_molecular_features[valid_indices]
            bond_indices = bond_indices[valid_indices]
            angle_indices = angle_indices[valid_indices]
            dihedral_indices = dihedral_indices[valid_indices]
            targets = targets[valid_indices]

            # Forward pass
            total_loss = 0
            for i in range(len(targets)):
                molecule_data = (atomic_features[i], bond_features[i], angle_features[i], dihedral_features[i], global_molecular_features[i], bond_indices[i], angle_indices[i], dihedral_indices[i])
                target = targets[i]

                prediction = gnn3d(molecule_data)
                target = torch.tensor([1, 0] if target == 0 else [0, 1], dtype=torch.float32).to(device)
                loss = mse_loss_fn(prediction, target)
                total_loss += loss

#           total_loss /= len(targets)

            if len(targets) > 0:
                total_loss /= len(targets)
            else:
                continue  # Safeguard against division by zero


            gnn_optimizer.zero_grad()
            total_loss.backward()
            gnn_optimizer.step()

            avg_rmse_loss_train += torch.sqrt(total_loss).item() * len(targets)
            total_samples_train += len(targets)

        avg_rmse_loss_train /= total_samples_train
        avg_rmse_losses_train.append(avg_rmse_loss_train)

        # Validation
        all_targets_val = []
        all_outputs_val = []
        skipped_samples_val = 0

        with torch.no_grad():
            for batch in tqdm(dataloader_val, desc='Validation samples'):

                # Filter out None values
                batch = [item for item in batch if item is not None]
                if not batch:
                    skipped_samples_val += len(batch)
                    continue

                atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices, targets = batch

                valid_indices = [i for i, target in enumerate(targets) if target is not None]
                if not valid_indices:
                    skipped_samples_val += len(targets)
                    continue
                # Only retain valid samples
                atomic_features = atomic_features[valid_indices]
                bond_features = bond_features[valid_indices]
                angle_features = angle_features[valid_indices]
                dihedral_features = dihedral_features[valid_indices]
                global_molecular_features = global_molecular_features[valid_indices]
                bond_indices = bond_indices[valid_indices]
                angle_indices = angle_indices[valid_indices]
                dihedral_indices = dihedral_indices[valid_indices]
                targets = targets[valid_indices]

                # Forward pass
                for i in range(len(targets)):
                    molecule_data = (atomic_features[i], bond_features[i], angle_features[i], dihedral_features[i], global_molecular_features[i], bond_indices[i], angle_indices[i], dihedral_indices[i])
                    target = targets[i]

                    prediction = gnn3d(molecule_data)
                    all_targets_val.append(target.item())
                    all_outputs_val.append(prediction[1].item())

        all_targets = np.array(all_targets_val)
        all_outputs = np.array(all_outputs_val)

        fpr_val, tpr_val, _ = roc_curve(all_targets, all_outputs)
        auc_score_per_epoch_val = auc(fpr_val, tpr_val)

        epoch_auc_val.append(auc_score_per_epoch_val)

        print(f"Epoch: {epoch_i:>3} | RMSE Train Loss: {avg_rmse_loss_train:.4f} | Validation AUC: {auc_score_per_epoch_val:.4f}")

    # Plot ROC Curves
    roc_curve_file_path = os.path.join(save_dir, f"gnn3d_auc_roc_{dataset_name}_random_state_{random_state}.png")
    plot_roc_curves(fpr_val, tpr_val, auc_score_per_epoch_val, roc_curve_file_path)

    # Saving model state
    torch.save(gnn3d.state_dict(), os.path.join(save_dir, f"gnn3d_{dataset_name}.pth"))
    print(f"Model saved to {os.path.join(save_dir, f'gnn3d_{dataset_name}.pth')}")

    end_time = time.time()
    actual_time = end_time - start_time

    actual_time_td = datetime.timedelta(seconds=actual_time)
    hours = actual_time_td.seconds // 3600
    minutes = (actual_time_td.seconds // 60) % 60
    seconds = actual_time_td.seconds % 60
    actual_time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"

    print(f"Time taken for training: {actual_time_str}")

    # Evaluation on test samples
    all_targets_test = []
    all_outputs_test = []
    skipped_samples_test = 0

    with torch.no_grad():
        for batch in tqdm(dataloader_test, desc='Test samples'):

            # Filter out None values
            batch = [item for item in batch if item is not None]
            if not batch:
                skipped_samples_test += len(batch)
                continue

            atomic_features, bond_features, angle_features, dihedral_features, global_molecular_features, bond_indices, angle_indices, dihedral_indices, targets_batch = batch

            valid_indices = [i for i, target in enumerate(targets_batch) if target is not None]
            if not valid_indices:
                skipped_samples_test += len(targets_batch)
                continue
            # Only retain valid samples
            atomic_features = atomic_features[valid_indices]
            bond_features = bond_features[valid_indices]
            angle_features = angle_features[valid_indices]
            dihedral_features = dihedral_features[valid_indices]
            global_molecular_features = global_molecular_features[valid_indices]
            bond_indices = bond_indices[valid_indices]
            angle_indices = angle_indices[valid_indices]
            dihedral_indices = dihedral_indices[valid_indices]
            targets_batch = targets_batch[valid_indices]

            # Forward pass
            for i in range(len(targets_batch)):
                molecule_data = (atomic_features[i], bond_features[i], angle_features[i], dihedral_features[i], global_molecular_features[i], bond_indices[i], angle_indices[i], dihedral_indices[i])
                target = targets_batch[i]

                prediction = gnn3d(molecule_data)
                all_targets_test.append(target.item())
                all_outputs_test.append(prediction[1].item())


    all_targets = np.array(all_targets_test)
    all_outputs = np.array(all_outputs_test)

    fpr, tpr, _ = roc_curve(all_targets, all_outputs)
    auc_score_test = auc(fpr, tpr)

    print(f"Test Results:")
    print(f"Test AUC: {auc_score_test:.4f}")

   #for target, output in zip(all_targets_test, all_outputs_test):
   #    print(f"Target: {target}, Predicted Probability: {output:.4f}")

    end_time = time.time()
    actual_time = end_time - start_time

    actual_time_td = datetime.timedelta(seconds=actual_time)
    hours = actual_time_td.seconds // 3600
    minutes = (actual_time_td.seconds // 60) % 60
    seconds = actual_time_td.seconds % 60
    actual_time_str = f"{hours} hours, {minutes} minutes, {seconds} seconds"

    print(f"Time taken for training and evaluation: {actual_time_str}")

    return avg_rmse_losses_train, epoch_auc_val, all_outputs_test, all_targets_test, skipped_samples_train, skipped_samples_val, skipped_samples_test


## Know your variables for main()================================
def main():
    """Function for train, validation and test
       Make sure to assign your local variables here too"""
    mp.set_start_method('spawn', force=True)

    datadir="../../../data/classification/hdat/"
    modeldir="./models/"
    dataset_name="hdat"

    train_samples = ToxQDataset(datadir+"train")
    print(train_samples)
    print("===================================")
    test_samples = ToxQDataset(datadir+"test")
    print(test_samples)
    print("===================================")

   #subset_size = 525
   #subset_indices = list(range(min(subset_size, len(train_samples))))
   #subset_samples = Subset(train_samples, subset_indices)

    n_epochs=10
    printstep=2
    batch_size=1
    prefetch_factor=None
    num_workers=0

    rmse_losses_train = []
    rmse_losses_val = []

    random_state_values = [21356, 137420, 949928, 435690, 7435290]

    for random_state in random_state_values:

        print(f"Training for random_state = {random_state}")


        latent_atomic_vector_size = 46
        latent_bond_vector_size = 3
        number_of_molecular_features = 200
        number_of_classes = 2

        train_part_samples, val_samples = train_test_split(train_samples, test_size=0.1, random_state=random_state)
       #train_part_samples, val_samples = train_test_split(subset_samples, test_size=0.1, random_state=random_state)
        atom_autoencoder = AtomAutoencoder(154, latent_atomic_vector_size).to(device)
        bond_autoencoder = BondAutoencoder(10, latent_bond_vector_size).to(device)

        gnn3d_config = GNN3DConfig(latent_atomic_vector_size, latent_bond_vector_size, number_of_molecular_features)
        gnn3d_classifier = GNN3DClassifier(gnn3d_config, number_of_classes)

        rmse_losses_train_per_model, rmse_losses_val_per_model, predictions, targets, skipped_train, skipped_val, skipped_test=train_gnn3d_model(
            gnn3d_classifier,
            train_part_samples,
            val_samples,
            test_samples,
            atom_autoencoder,
            bond_autoencoder,
            n_epochs=n_epochs,
            printstep=printstep,
            save_dir=modeldir,
            dataset_name=dataset_name,
            batch_size=batch_size,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
            random_state=random_state
        )


        rmse_losses_train.append(rmse_losses_train_per_model)
        rmse_losses_val.append(rmse_losses_val_per_model)

    return gnn3d_classifier, atom_autoencoder, bond_autoencoder, n_epochs, rmse_losses_train, rmse_losses_val, predictions, targets


## Training=================================================
## GNN3D training for multiple models; and testing
if __name__ == '__main__':
    gnn3d_classifier, atom_autoencoder, bond_autoencoder, n_epochs, rmse_losses_train, rmse_losses_val, predictions, targets = main()

    print("RMSE Losses Train:", rmse_losses_train)
    print("AUC Losses Val:", rmse_losses_val)

    plt.figure(figsize=(10, 6))
    for i, train_losses in enumerate(rmse_losses_train):
        plt.plot(range(1, n_epochs + 1), train_losses, marker='o', color=f'C{i}', label=f'Model {i+1}: Training Loss RMSE')

    for i, val_losses in enumerate(rmse_losses_val):
        plt.plot(range(1, n_epochs + 1), val_losses, marker='x', color=f'C{i}', label=f'Model {i+1}: Validation Loss AUC')

    # Calculate and plot average losses for training and validation from various models
    avg_train_losses = [sum(losses) / len(losses) for losses in zip(*rmse_losses_train)]
    avg_val_losses = [sum(losses) / len(losses) for losses in zip(*rmse_losses_val)]

    plt.plot(range(1, n_epochs + 1), avg_train_losses, marker='o', color='black', linestyle='--', label='Average Training Loss RMSE')
    plt.plot(range(1, n_epochs + 1), avg_val_losses, marker='x', color='black', linestyle='--', label='Average Validation Loss AUC')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs for Multiple Models')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{modeldir}gnn3d_losses_{dataset_name}_multi_models.png")
    plt.savefig(f"{modeldir}gnn3d_losses_{dataset_name}_multi_models.svg", format='svg')
   #plt.show()

    print(f"Epoch | Average Training Loss | Average Validation Loss")
    for i in range(n_epochs):
        print(f" {i:>4} | {avg_train_losses[i]:.4f}                | {avg_val_losses[i]:.4f}")


## Know your correlations (KYC)=============================

    # Convert lists to NumPy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(predictions, targets)

    print(f"Pearson Correlation Coefficient: {corr:.4f}")

    r_squared = calculate_r_squared(predictions, targets)
    print(f"Coefficient of Determination R^2: {r_squared:.4f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, color='blue', alpha=0.5)
    plt.plot(targets, targets, color='red', linestyle='-', linewidth=2)  # Plot y=x line
    plt.text(np.min(targets), np.max(predictions), f"Pearson Correlation: {corr:.4f}", fontsize=12, verticalalignment='top')
    plt.title('Scatter Plot of Predictions vs Targets')
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{modeldir}gnn3d_scatter_{dataset_name}_predictions_targets.png")
    plt.savefig(f"{modeldir}gnn3d_scatter_{dataset_name}_predictions_targets.svg", format='svg')

## Know your structures (KYS)===============================
    '''Test data WITH SKIP starts
       USE this if you have <translated_smiles.npy> file'''
    molecules = np.load(datadir + 'test/translated_smiles.npy')
    results = pd.DataFrame({
        'y_true': targets,
        'y_pred_prob': predictions,
        'smiles': molecules
    })


    results.to_csv(modeldir + 'results_true_pred_smiles.csv', index=False, float_format='%.4f')

# Determine predictions
    results['y_pred'] = (results['y_pred_prob'] >= 0.5).astype(int)  # Adjust threshold if needed
    results['is_correct'] = results['y_true'] == results['y_pred']

# Select misclassified samples and sort them
    misclassified = results[~results['is_correct']]
    top_misclassified = misclassified.sort_values(by='y_pred_prob').head(10)  # Adjust as needed

# Select correctly classified samples and sort them
    best_classified = results[results['is_correct']]
    top_best_classified = best_classified.sort_values(by='y_pred_prob', ascending=False).head(10)  # Adjust as needed

# Prepare for visualization of misclassified samples
    top_misclassified_smiles = results['smiles'].iloc[top_misclassified.index].apply(lambda x: Chem.MolFromSmiles(x))
    top_misclassified_y_true = top_misclassified['y_true'].tolist()
    top_misclassified_y_pred = top_misclassified['y_pred'].tolist()
    top_misclassified_y_pred_prob = top_misclassified['y_pred_prob'].tolist()
    misclassified_legends = [f"True: {top_misclassified_y_true[i]}, Pred: {top_misclassified_y_pred[i]} (Prob: {top_misclassified_y_pred_prob[i]:.4f})" for i in range(len(top_misclassified))]

# Generate and save image for misclassified samples
    misclassified_img = Draw.MolsToGridImage(top_misclassified_smiles, molsPerRow=5, maxMols=10, legends=misclassified_legends, useSVG=True)
    misclassified_svg = misclassified_img.data
    with open(f"./{modeldir}/{dataset_name}_top_misclassified_mols_to_grid.svg", "w") as f:
        f.write(misclassified_svg)

# Prepare for visualization of best classified samples
    top_best_classified_smiles = results['smiles'].iloc[top_best_classified.index].apply(lambda x: Chem.MolFromSmiles(x))
    top_best_classified_y_true = top_best_classified['y_true'].tolist()
    top_best_classified_y_pred = top_best_classified['y_pred'].tolist()
    top_best_classified_y_pred_prob = top_best_classified['y_pred_prob'].tolist()
    best_classified_legends = [f"True: {top_best_classified_y_true[i]}, Pred: {top_best_classified_y_pred[i]} (Prob: {top_best_classified_y_pred_prob[i]:.4f})" for i in range(len(top_best_classified))]

# Generate and save image for best classified samples
    best_classified_img = Draw.MolsToGridImage(top_best_classified_smiles, molsPerRow=5, maxMols=10, legends=best_classified_legends, useSVG=True)
    best_classified_svg = best_classified_img.data
    with open(f"./{modeldir}/{dataset_name}_top_best_classified_mols_to_grid.svg", "w") as f:
        f.write(best_classified_svg)
