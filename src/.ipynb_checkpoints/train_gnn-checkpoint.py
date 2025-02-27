from model import GNN3D, Autoencoder
from dataset import LogSDataset, LogPDataset, QM9

import torch

use_gpu = False
n_epochs = 3
printstep = 50

# Setting up device
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

# Loading datasets
print("Loading Log P Dataset.")
logp_dataset = LogPDataset("./data/logp")
print(logp_dataset)
print("\n\nLoading Log S Dataset.")
logs_dataset = LogSDataset("./data/logs")
print(logs_dataset)
print(logs_dataset[-1][8])
print("\n\nLoading QM9 Dataset.")
qm9_dataset = QM9("./data/qm9")
print(qm9_dataset)

exit()

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

# Loading saved .pth autoencoder models
atom_autoencoder = Autoencoder(80, 10)
bond_autoencoder = Autoencoder(10, 3)
gnn3d = GNN3D(atomic_vector_size=10, bond_vector_size=3, number_of_molecular_features=200, number_of_targets=1)

atom_autoencoder.load_state_dict(torch.load("./models/atom_autoencoder.pth"))
bond_autoencoder.load_state_dict(torch.load("./models/bond_autoencoder.pth"))

# Printing summary of both autoencoders
print("Atom Autoencoder:\n", atom_autoencoder)
print("\nBond Autoencoder:\n", bond_autoencoder)
print("\nGNN3D:\n", gnn3d)

# Setting up loss function
mse_loss_fn = torch.nn.MSELoss()
mae_loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.Adam(list(atom_autoencoder.parameters()) + list(bond_autoencoder.parameters()) + list(gnn3d.parameters()))

##################################### LOG P Data ###################################


print("Training autoencoders on logp")
for epoch_i in range(n_epochs):
  avg_rmse_loss = 0
  avg_mse_loss = 0
  for i, molecule in enumerate(logp_dataset):
    
    if i > 4100:
      break # Everything else is for training.
    
    # Putting everything onto "device"  
    target = molecule[8].to(device)
    molecule = [molecule[0].to(device), molecule[1].to(device), molecule[2].to(device), molecule[3].to(device), molecule[4].to(device), molecule[5].to(device), molecule[6].to(device), molecule[7].to(device)]
    
    # Putting latent atomic and bond features through GNN3D
    molecule[0] = atom_autoencoder.encode(molecule[0])
    molecule[1] = bond_autoencoder.encode(molecule[1])
      
    # Forward pass
    prediction = gnn3d(molecule)
    
    # Calculating loss
    loss = mse_loss_fn(prediction, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculating average loss
    avg_rmse_loss = (avg_rmse_loss * i + (loss.item() ** 0.5)) / (i + 1)
    avg_mse_loss = (avg_mse_loss * i + loss.item()) / (i + 1)
    
    if i % printstep == 0:
      print(f"LOG P Epoch: {epoch_i}, ex: {i}, Avg. RMSE Loss: {avg_rmse_loss}, Avg. MSE Loss: {avg_mse_loss}")

  # Saving checkpoint
  torch.save(atom_autoencoder.state_dict(), "./models/logp/atom_autoencoder_" + str(epoch_i) +".pth")
  torch.save(bond_autoencoder.state_dict(), "./models/logp/bond_autoencoder_" + str(epoch_i) +".pth")
  torch.save(gnn3d.state_dict(), "./models/logp/gnn3d_" + str(epoch_i) +".pth")


# Testing models on the remaining data
avg_rmse_loss = 0
avg_mse_loss = 0


for i, molecule in enumerate(logp_dataset):
    
    if i <= 4100:
      continue # Everything else is for testing.
    
    # Putting everything onto "device"  
    molecule = [molecule[0].to(device), molecule[1].to(device), molecule[2].to(device), molecule[3].to(device), molecule[4].to(device), molecule[5].to(device), molecule[6].to(device), molecule[7].to(device)]
    target = molecule[8].to(device)
    
    # Putting latent atomic and bond features through GNN3D
    molecule[0] = atom_autoencoder(molecule[0])
    molecule[1] = bond_autoencoder(molecule[1])
      
    # Forward pass
    prediction = gnn3d(molecule)
    
    # Calculating loss
    loss = mse_loss_fn(prediction, target)
    
    # Calculating average loss
    avg_rmse_loss = (avg_rmse_loss * i + (loss.item() ** 0.5)) / (i + 1)
    avg_mse_loss = (avg_mse_loss * i + loss.item()) / (i + 1)
    
  
print(f"LOG P Test RMSE Loss: {avg_rmse_loss}, Test MSE Loss: {avg_mse_loss}")

# Saving model weights
torch.save(atom_autoencoder.state_dict(), "./models/logp_atom_autoencoder.pth")
torch.save(bond_autoencoder.state_dict(), "./models/logp_bond_autoencoder.pth")
torch.save(gnn3d.state_dict(), "./models/logp_gnn3d.pth")
