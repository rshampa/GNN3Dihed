# GNN3Dihed: Graph neural network for 3-dimensional structures including dihedral angles for molecular property prediction

  ChemRxiv: https://doi.org/10.26434/chemrxiv-2024-jlwh5

## Directory Structure

  data, train, and notebooks have similar regression and classification directory structures.  

```
root/
  ├── LICENSE
  ├── README.md
  ├── requirements.txt
  ├── src/
  │   ├── __init__.py
  │   ├── config.py
  │   ├── dataset.py
  │   ├── model.py
  │   ├── molecular_representation.py
  │   └── utils.py
  ├── data/
  │   ├── regression/
  │   │   └── datasetname/
  │   │       ├── train.csv
  │   │       ├── test.csv
  │   │       ├── train/
  │   │       │   ├── *_features.npy
  │   │       │   └── *_indices.npy
  │   │       ├── test/
  │   │       │   ├── *_features.npy
  │   │       │   └── *_indices.npy
  ├── train/
  │   ├── regression/
  │   │   └── datasetname/
  │   │       ├── molrep.py
  │   │       ├── train.py
  │   │       └── runme.sh
  ├── notebooks/
  │   ├── regression/
  │   │   └── datasetname/
  │   │       ├── train_autoencoder_only.ipynb
  │   │       └── train.ipynb
```

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

---

## Installation

### Prerequisites

Make sure you have the following software installed:

- Conda (or pip if using an environment manager other than conda)

### Steps to install

1. Clone the repository:

```
   git clone https://github.com/rshampa/GNN3Dihed.git
```

2. Navigate to the root folder:

```
   cd GNN3Dihed
```

3. Create a virtual environment (recommended):

```
   conda create -n gnn3dihed python=3.12
   conda activate gnn3dihed
   pip install -r requirements.txt
```

## Usage

  How to use?

  - Regression task: An example for FreeSolv dataset

### Steps to generate features and train a model

1. Go to freesolv folder:

```
   cd train/regression/freesolv 
```

2. Run runme.sh:

```
   ./runme.sh
```

  - Classification  task: An example for hDAT dataset

### Steps to generate features and train a model

1. Go to hdat folder:

```
   cd train/classification/hdat
```

2. Run runme.sh:

```
   ./runme.sh
```

## License

   This work is licensed under the MIT License.

## Citation

   If you find this repository useful, please cite our paper:

   Reddy Sangala SA, Raghunathan S. Graph neural network for 3-dimensional structures including dihedral angles for molecular property prediction. ChemRxiv. 2024; https://doi.org/10.26434/chemrxiv-2024-jlwh5

## Contact

   If you have any questions or feedback, feel free to reach out! 

   Shampa Raghunathan: shampa.raghunathan@mahindrauniversity.edu.in
