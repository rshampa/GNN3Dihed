from model import Autoencoder
from dataset import LogSDataset, QM9, LogPDataset

import torch

use_gpu = False
n_epochs = 30
printstep = 1000

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

# Loading datasets
print("Loading Log P Dataset.")
logp_dataset = LogPDataset("./data/logp")
print(logp_dataset)
print("\n\nLoading Log S Dataset.")
logs_dataset = LogSDataset("./data/logs")
print(logs_dataset)
# print("\n\nLoading QM9 Dataset.")
# qm9_dataset = QM9("./data/qm9")
# print(qm9_dataset)

molecule = logp_dataset[0]

# Printing out the dimensions of all of these features with a description of what each feature is
print(f"Atomic Features: {molecule[0].shape} - This represents the atomic features of the molecule")
print(f"Bond Features: {molecule[1].shape} - This represents the bond features of the molecule")
print(f"Angle Features: {molecule[2].shape} - This represents the angle features of the molecule")
print(f"Dihedral Features: {molecule[3].shape} - This represents the dihedral features of the molecule")
print(f"Global Molecular Features: {molecule[4].shape} - This represents the global molecular features of the molecule")
print(f"Bond Indices: {molecule[5].shape} - This represents the bond indices of the molecule")
print(f"Angle Indices: {molecule[6].shape} - This represents the angle indices of the molecule")
print(f"Dihedral Indices: {molecule[7].shape} - This represents the dihedral indices of the molecule")
print(f"Target: {molecule[8].shape} - This represents the target of the molecule")



# Training autoencoder
atom_autoencoder = Autoencoder(80, 10)
bond_autoencoder = Autoencoder(10, 3)

# Printing summary of both autoencoders
print("Atom Autoencoder:\n", atom_autoencoder)
print("\nBond Autoencoder:\n", bond_autoencoder)

# Setting up loss function
mse_loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
atom_optimizer = torch.optim.Adam(atom_autoencoder.parameters())
bond_optimizer = torch.optim.Adam(bond_autoencoder.parameters())

##################################### LOG S Data ###################################

print("Training autoencoders on logs")
for epoch_i in range(n_epochs):
  avg_atom_rmse_loss = 0
  avg_bond_rmse_loss = 0
  for i, molecule in enumerate(logs_dataset):
    
    # if i > 1000:
    #   break # Everything else is for training.
    
    # Putting everything onto "device"  
  
    atom_features = molecule[0].to(device)
    bond_features = molecule[1].to(device)
      
    # Forward pass
    reconstructed_atom = atom_autoencoder(atom_features)
    reconstructed_bond = bond_autoencoder(bond_features)
    
    # Calculating loss
    atom_loss = mse_loss_fn(reconstructed_atom, atom_features)
    bond_loss = mse_loss_fn(reconstructed_bond, bond_features)
    
    # Backward pass
    atom_optimizer.zero_grad()
    bond_optimizer.zero_grad()
    
    atom_loss.backward()
    bond_loss.backward()
    
    atom_optimizer.step()
    bond_optimizer.step()
    
    # Calculating average loss
    avg_atom_rmse_loss = (avg_atom_rmse_loss * i + atom_loss.item() ** 0.5) / (i + 1)
    avg_bond_rmse_loss = (avg_bond_rmse_loss * i + bond_loss.item() ** 0.5) / (i + 1)
    
    if i % printstep == 0:
      print(f"LOG S Epoch: {epoch_i}, ex: {i}, Atom RMSE Loss: {avg_atom_rmse_loss}, Bond RMSE Loss: {avg_bond_rmse_loss}")

# Testing models on the remaining data
avg_atom_rmse_loss = 0
avg_bond_rmse_loss = 0

for i, molecule in enumerate(logs_dataset):
    
    # if i <= 1000:
    #   continue # Everything else is for testing.
    
    # Putting everything onto "device"  
  
    atom_features = molecule[0].to(device)
    bond_features = molecule[1].to(device)
      
    # Forward pass
    reconstructed_atom = atom_autoencoder(atom_features)
    reconstructed_bond = bond_autoencoder(bond_features)
    
    # Calculating loss
    atom_loss = mse_loss_fn(reconstructed_atom, atom_features)
    bond_loss = mse_loss_fn(reconstructed_bond, bond_features)
    
    # Calculating average loss
    avg_atom_rmse_loss = (avg_atom_rmse_loss * i + atom_loss.item() ** 0.5) / (i + 1)
    avg_bond_rmse_loss = (avg_bond_rmse_loss * i + bond_loss.item() ** 0.5) / (i + 1)
  
print(f"LOG S Test Atom RMSE Loss: {avg_atom_rmse_loss}, Test Bond RMSE Loss: {avg_bond_rmse_loss}")


##################################### LOG P Data ###################################


print("Training autoencoders on logp")
for epoch_i in range(n_epochs):
  avg_atom_rmse_loss = 0
  avg_bond_rmse_loss = 0
  for i, molecule in enumerate(logp_dataset):
    
    # if i > 1000:
    #   break # Everything else is for training.
    
    # Putting everything onto "device"  
  
    atom_features = molecule[0].to(device)
    bond_features = molecule[1].to(device)
      
    # Forward pass
    reconstructed_atom = atom_autoencoder(atom_features)
    reconstructed_bond = bond_autoencoder(bond_features)
    
    # Calculating loss
    atom_loss = mse_loss_fn(reconstructed_atom, atom_features)
    bond_loss = mse_loss_fn(reconstructed_bond, bond_features)
    
    # Backward pass
    atom_optimizer.zero_grad()
    bond_optimizer.zero_grad()
    
    atom_loss.backward()
    bond_loss.backward()
    
    atom_optimizer.step()
    bond_optimizer.step()
    
    # Calculating average loss
    avg_atom_rmse_loss = (avg_atom_rmse_loss * i + atom_loss.item() ** 0.5) / (i + 1)
    avg_bond_rmse_loss = (avg_bond_rmse_loss * i + bond_loss.item() ** 0.5) / (i + 1)
    
    if i % printstep == 0:
      print(f"LOG P Epoch: {epoch_i}, ex: {i}, Atom RMSE Loss: {avg_atom_rmse_loss}, Bond RMSE Loss: {avg_bond_rmse_loss}")

# Testing models on the remaining data
avg_atom_rmse_loss = 0
avg_bond_rmse_loss = 0

for i, molecule in enumerate(logp_dataset):
    
    # if i <= 1000:
    #   continue # Everything else is for testing.
    
    # Putting everything onto "device"  
  
    atom_features = molecule[0].to(device)
    bond_features = molecule[1].to(device)
      
    # Forward pass
    reconstructed_atom = atom_autoencoder(atom_features)
    reconstructed_bond = bond_autoencoder(bond_features)
    
    # Calculating loss
    atom_loss = mse_loss_fn(reconstructed_atom, atom_features)
    bond_loss = mse_loss_fn(reconstructed_bond, bond_features)
    
    # Calculating average loss
    avg_atom_rmse_loss = (avg_atom_rmse_loss * i + atom_loss.item() ** 0.5) / (i + 1)
    avg_bond_rmse_loss = (avg_bond_rmse_loss * i + bond_loss.item() ** 0.5) / (i + 1)
  
print(f"LOG P Test Atom RMSE Loss: {avg_atom_rmse_loss}, Test Bond RMSE Loss: {avg_bond_rmse_loss}")

# Saving models to disk
torch.save(atom_autoencoder.state_dict(), "./models/atom_autoencoder.pth")
torch.save(bond_autoencoder.state_dict(), "./models/bond_autoencoder.pth")

