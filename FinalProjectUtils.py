import sys
import os
import pandas as pd
import torch
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolToSmiles
import torch_geometric
from torch_geometric.data import Data
from collections import defaultdict
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image
from IPython.display import display
from torch_geometric.data import Dataset
from rdkit import DataStructs
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import torch.optim




def strip_metadata(smiles):
    return smiles.split('|')[0] if isinstance(smiles, str) else smiles


# --- 2. Standardize molecule SMILES ---
def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
          for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

          return MolToSmiles(mol, canonical=True)
    except:
        return None

def standardize_multi_smiles(smiles_string):
    components = smiles_string.split('.')
    standardized = [standardize_smiles(sm) for sm in components]
    return '.'.join(sorted(filter(None, standardized)))

def build_fingerprint_lookup_from_dataframe(df, nBits=2048):
    """
    Given a DataFrame with 'reactants' and 'products' columns,
    builds a fingerprint lookup table {SMILES: RDKit fingerprint}.
    """
    all_molecules = set()

    for reactant_str in df['reactants']:
        for mol in reactant_str.split('.'):
            all_molecules.add(mol)

    for product_str in df['products']:
        all_molecules.add(product_str)

    all_molecules = {mol for mol in all_molecules if mol is not None}

    mol_to_id = {mol: idx for idx, mol in enumerate(sorted(all_molecules))}
    id_to_mol = {idx: mol for mol, idx in mol_to_id.items()}

    fingerprint_lookup = {}
    for i in tqdm(range(len(id_to_mol)), desc="Generating Morgan Fingerprints"):
        smi = id_to_mol[i]
        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            continue  # skip invalid
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=nBits)
        fingerprint_lookup[smi] = fp

    return fingerprint_lookup



def pooled_reactant_fingerprint(reactants, lookup, nBits=2048):
    """
    Pool RDKit fingerprints (ExplicitBitVect) for all components in a dot-separated reactant string.
    Returns a NumPy array representing the mean fingerprint.
    """
    fps = []
    for sm in reactants.split('.'):
        fp = lookup.get(sm)
        if fp is not None:
            arr = np.zeros((nBits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
        else:
            fps.append(np.zeros(nBits, dtype=np.float32))  # fallback for missing

    return np.mean(fps, axis=0) if fps else np.zeros(nBits, dtype=np.float32)




''' First Set of Helper Functions '''

from rdkit import Chem

def parse_reaction(smiles):   # This needs to be the full reaction smiles
    reactants, _, products = smiles.split(">")  # Assumes the presence of reagents
    reactant_mols = [Chem.MolFromSmiles(sm) for sm in reactants.split('.')]
    product_mols = [Chem.MolFromSmiles(sm) for sm in products.split('.')]
    return reactant_mols, product_mols


def get_mapped_atoms(mols):
    atom_info = {}
    for mol in mols:
        if mol is None:
            continue
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num == 0:
                continue  # skip unmapped atoms
            atom_info[map_num] = {
                'symbol': atom.GetSymbol(),
                'neighbors': sorted([nbr.GetSymbol() for nbr in atom.GetNeighbors()])
            }
    return atom_info

def detect_reaction_centers(reactant_info, product_info):
    centers = []
    for map_num in reactant_info:
        if map_num not in product_info:
            centers.append(map_num)
        else:
            r = reactant_info[map_num]
            p = product_info[map_num]
            if r['symbol'] != p['symbol'] or r['neighbors'] != p['neighbors']:
                centers.append(map_num)
    return centers


def find_reaction_centers(atom_mapped_smiles):
    reactants, products = parse_reaction(atom_mapped_smiles)
    reactant_info = get_mapped_atoms(reactants)
    product_info = get_mapped_atoms(products)
    return detect_reaction_centers(reactant_info, product_info)

def get_broken_bonds(product_mol, reaction_center_atom_maps):
    broken_bonds = set()
    atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }

    for atom in product_mol.GetAtoms():
        idx_i = atom.GetIdx()
        map_i = atom.GetAtomMapNum()

        for nbr in atom.GetNeighbors():
            idx_j = nbr.GetIdx()
            map_j = nbr.GetAtomMapNum()

            if map_i in reaction_center_atom_maps or map_j in reaction_center_atom_maps:
                bond = product_mol.GetBondBetweenAtoms(idx_i, idx_j)
                if bond:
                    broken_bonds.add((min(idx_i, idx_j), max(idx_i, idx_j)))

    return list(broken_bonds)


def extract_reaction_center_bonds(row):
    try:
        reaction_smiles = row["ReactionSmiles_AtomMap"]
        center_atoms = row["reaction_center"]

        # Extract product part from full reaction SMILES
        product_smiles = reaction_smiles.split(">")[2]

        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None or not center_atoms:
            return []

        return get_broken_bonds(mol, center_atoms)
    except Exception as e:
        return []
    

''' Second Set of Helper Functions '''

def reaction_center_to_bond_indices(product_smiles_mapped, reaction_center_mapnums):
    mol = Chem.MolFromSmiles(product_smiles_mapped)
    if mol is None:
        return []

    # Build map number → atom index lookup
    mapnum_to_idx = {}
    for atom in mol.GetAtoms():
        map_num = atom.GetAtomMapNum()
        if map_num != 0:
            mapnum_to_idx[map_num] = atom.GetIdx()

    bond_indices = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        begin_map = begin.GetAtomMapNum()
        end_map = end.GetAtomMapNum()

        # If either atom in the bond is in the reaction center
        if begin_map in reaction_center_mapnums or end_map in reaction_center_mapnums:
            bond_indices.append(bond.GetIdx())

    return bond_indices

def get_broken_bonds(product_mol, reaction_center_atom_maps):
    broken_bonds = set()
    atom_map_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in product_mol.GetAtoms()
        if atom.GetAtomMapNum() != 0
    }

    for atom in product_mol.GetAtoms():
        idx_i = atom.GetIdx()
        map_i = atom.GetAtomMapNum()

        for nbr in atom.GetNeighbors():
            idx_j = nbr.GetIdx()
            map_j = nbr.GetAtomMapNum()

            if map_i in reaction_center_atom_maps or map_j in reaction_center_atom_maps:
                bond = product_mol.GetBondBetweenAtoms(idx_i, idx_j)
                if bond:
                    broken_bonds.add((min(idx_i, idx_j), max(idx_i, idx_j)))

    return list(broken_bonds)


def extract_reaction_center_bonds(row):
    try:
        reaction_smiles = row["ReactionSmiles_AtomMap"]
        center_atoms = row["reaction_center"]

        # Extract product part from full reaction SMILES
        product_smiles = reaction_smiles.split(">")[2]

        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None or not center_atoms:
            return []

        return get_broken_bonds(mol, center_atoms)
    except Exception as e:
        return []

def extract_reaction_center_bond_indices(row):
    try:
        product_smiles = row["ReactionSmiles_AtomMap"].split(">")[2]
        return reaction_center_to_bond_indices(product_smiles, row["reaction_center"])
    except:
        return []


''' Third Set of Helper Functions '''

import torch
from rdkit import Chem
from torch_geometric.data import Data

# ─── Helper: one‑hot for atomic numbers ───────────────────────────────────────
# H, B, C, N, O, F, Mg, Si, P, S, Cl, Cu, Zn, Br, Pd, Sn, I + “other”
ALLOWED_ATOM_NUMS = [
    1,   # H
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    12,  # Mg
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    29,  # Cu
    30,  # Zn
    35,  # Br
    46,  # Pd
    50,  # Sn
    53   # I
]


def one_hot(val, allowed):
    vec = [int(val == a) for a in allowed]
    # if it wasn’t in the allowed list, flip the final “other” bit
    vec.append(int(sum(vec) == 0))
    return vec


featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)


def smile_to_graph(smiles: str):
    if not isinstance(smiles, str):
        return None

    parts = smiles.split('.')
    graphs = []

    for smi in parts:
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumAtoms() <= 1:
            continue  # skip invalid or single-atom molecules

        dc_graph = featurizer.featurize([mol])[0]
        if not isinstance(dc_graph, GraphData):
            continue

        x = torch.tensor(dc_graph.node_features, dtype=torch.float)
        edge_index = torch.tensor(dc_graph.edge_index, dtype=torch.long)
        edge_attr = torch.tensor(dc_graph.edge_features, dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    # return graphs if len(graphs) > 1 else graphs[0] if graphs else None
    return graphs



import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_reaction_subgraph(reaction_graph, num_nodes=30, seed=42):
    """
    Visualize a subgraph of a reaction graph using NetworkX and matplotlib.
    
    Parameters:
        reaction_graph (torch_geometric.data.Data): PyG Data object representing the reaction graph.
        num_nodes (int): Number of nodes to include in the subgraph (starting from index 0).
        seed (int): Random seed for layout consistency.

    Returns:
        None. Displays the subgraph with SMILES as node labels and yields as edge labels.
    """
    # Step 1: Convert to NetworkX DiGraph
    G = to_networkx(reaction_graph, to_undirected=False)

    # Step 2: Select a subgraph of the first `num_nodes` nodes
    subset_nodes = list(G.nodes)[:num_nodes]
    subgraph = G.subgraph(subset_nodes)

    # Step 3: Map nodes to SMILES labels
    node_labels = {i: reaction_graph.smiles[i] for i in subgraph.nodes}

    # Step 4: Extract yield values for edges in the subgraph
    edge_labels = {}
    for i, (src, dst) in enumerate(reaction_graph.edge_index.t().tolist()):
        if src in subgraph.nodes and dst in subgraph.nodes:
            yield_val = reaction_graph.edge_weight[i].item()
            edge_labels[(src, dst)] = f"{yield_val:.1f}%"

    # Step 5: Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(subgraph, seed=seed)

    nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(subgraph, pos, labels=node_labels, font_size=7)
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)

    plt.title("Reaction Subgraph with SMILES Node Labels and Yield Edge Labels", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_reaction_center_bonds(smiles, bond_indices):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES")
        return

    Chem.rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DCairo(500, 300)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=bond_indices
    )
    drawer.FinishDrawing()
    return Image(drawer.GetDrawingText())

def draw_reactants_and_products(reactants_smiles, products_smiles):
    def clean(smiles):
        return smiles.split("|")[0].strip()  # strip any remaining metadata just in case

    reactants_smiles = clean(reactants_smiles)
    products_smiles = clean(products_smiles)

    reactant_mols = []
    for sm in reactants_smiles.split("."):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            Chem.rdDepictor.Compute2DCoords(mol)
            reactant_mols.append(mol)
        else:
            print(f"⚠️ Failed to parse reactant: {sm}")

    product_mols = []
    for sm in products_smiles.split("."):
        mol = Chem.MolFromSmiles(sm)
        if mol:
            Chem.rdDepictor.Compute2DCoords(mol)
            product_mols.append(mol)
        else:
            print(f"⚠️ Failed to parse product: {sm}")

    print("🔹 Reactants:")
    display(Draw.MolsToGridImage(reactant_mols, molsPerRow=4, subImgSize=(300, 300)))

    print("🔸 Products:")
    if product_mols:
        display(Draw.MolsToGridImage(product_mols, molsPerRow=4, subImgSize=(300, 300)))
    else:
        print("⚠️ No valid products to display.")


class ReactionCenterDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        product_smiles = row["ReactionSmiles_AtomMap"].split(">")[2]
        mol = Chem.MolFromSmiles(product_smiles)

        if mol is None:
            return None  # Optionally skip or raise

        # --- 1. Nodes ---
        x = []
        for atom in mol.GetAtoms():
            # You can add more atom features later
            x.append([atom.GetAtomicNum()])
        x = torch.tensor(x, dtype=torch.float)

        # --- 2. Edges ---
        edge_index = []
        bond_idx_to_edge_idx = {}  # map RDKit bond idx → position in edge list
        for i, bond in enumerate(mol.GetBonds()):
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            edge_index.append([a1, a2])
            edge_index.append([a2, a1])  # undirected graph
            bond_idx_to_edge_idx[bond.GetIdx()] = len(edge_index) - 2  # one direction only

        edge_index = torch.tensor(edge_index, dtype=torch.long).T

        # --- 3. Edge Labels ---
        bond_indices = row["reaction_center_bond_indices"]
        num_bonds = mol.GetNumBonds()
        edge_label = torch.zeros(edge_index.size(1), dtype=torch.float)

        for bond_idx in bond_indices:
            if bond_idx in bond_idx_to_edge_idx:
                # Set both directions (i→j and j→i) as 1
                forward = bond_idx_to_edge_idx[bond_idx]
                edge_label[forward] = 1
                edge_label[forward + 1] = 1

        return Data(x=x, edge_index=edge_index, edge_label=edge_label)


class NewGAT(nn.Module):
  def __init__(self, feature_size, edge_dim, embedding_size=128):
    super(NewGAT, self).__init__()
    self.edge_emb = nn.Linear(edge_dim, embedding_size)

    # First Layer
    self.conv1 = GATConv(feature_size, embedding_size, heads=3, concat=True, edge_dim=edge_dim)

    # Linear refocus
    self.focus_layer1 = nn.Linear(embedding_size * 3, embedding_size)

    # Second Layer
    self.conv2 = GATConv(embedding_size , embedding_size,  heads=3, concat=True, edge_dim=edge_dim)

    # Linear refocus
    self.focus_layer2 = nn.Linear(embedding_size * 3, embedding_size)

    # Output Layer
    self.readout = nn.Linear(embedding_size , embedding_size)


  def forward(self,x, edge_index, edge_attr, batch):

    # First layer forward pass
    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)

    # Refocus
    x = self.focus_layer1(x)
    x = F.relu(x)

    # Second layer forward pass
    x = self.conv2(x, edge_index, edge_attr)
    x = F.relu(x)

    # Refocus
    x = self.focus_layer2(x)
    x = F.relu(x)
   
    # Pooling and output
    x = global_add_pool(x, batch)
    x = self.readout(x)

    return F.relu(x)

class Predictor(nn.Module):
    def __init__(self, embedding_size):
        super(Predictor, self).__init__()

        # Multi-layer Perceptron
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LeakyReLU(0.01),
            nn.Linear(embedding_size, embedding_size//2),
            nn.ReLU(),
            nn.Linear(embedding_size//2, 1)
        )


    def forward(self, react_emb, prod_emb):
        x = torch.cat([react_emb, prod_emb], dim=-1)  # shape: [1, 256]
        out = self.mlp(x).squeeze(-1)
        return out




