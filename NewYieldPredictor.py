import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from rdkit import Chem
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat.graph_data import GraphData


featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)

class PredictionData(Dataset):
  def __init__(self, dataframe):
    self.df = dataframe.reset_index(drop = True)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    rgraph = self.df.loc[idx, 'reactants']
    pgraph = self.df.loc[idx, 'product']
    return rgraph, pgraph


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
    

def Ypredict(synthons, product, gnn, mlp, DataLoad):
  ''' 
  gnn is the graph embedding generator, mlp is the
  yield predictor, and DataLoad is the PyTorch DataLoader.
  Function returns the yield prediction. 
  '''
  device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  gnn.eval()  # Must be a trained GNN
  gnn.to(device)
  mlp.eval()  # Must be a trained MLP
  mlp.to(device)


  # Build the graphs from the inputs
  reactants = []
  product = smile_to_graph(product)
  for syn in synthons:
    syn_graph = smile_to_graph(syn)
    reactants.append(syn_graph[0])
  
  # Build the DataFrame
  data = {
      'reactants': [reactants],
      'product': [product]
  }

  df = pd.DataFrame(data)

  # Build the DataLoader
  OurDataSet = PredictionData(df)
  OurDataLoad = DataLoad(OurDataSet)

  for react, prod in OurDataLoad:
    if isinstance(react, list):
      react = Batch.from_data_list(react)
    react.to(device)

    if isinstance(prod, list):
      prod = Batch.from_data_list(prod)
    prod.to(device)
    
    with torch.no_grad():
          react_x = gnn(
              react.x,
              react.edge_index,
              react.edge_attr,
              react.batch
          )
          prod_x = gnn(
              prod.x,
              prod.edge_index,
              prod.edge_attr,
              prod.batch
          )

          react_x = react_x.mean(dim=0, keepdim=True)
          pred_yield = mlp(react_x,prod_x).squeeze().item()

    return pred_yield
