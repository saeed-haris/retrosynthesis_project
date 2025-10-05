from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Draw
import random
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit.Chem import rdmolops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display

class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, latent_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.embed(x)
        _, h = self.encoder_rnn(emb)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        h0 = self.latent_to_hidden(z).unsqueeze(0)
        start_input = torch.zeros((z.size(0), seq_len, self.embed.embedding_dim), device=z.device)
        dec_out, _ = self.decoder_rnn(start_input, h0)
        return self.output(dec_out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        logits = self.decode(z, x.size(1))
        return logits, mu, logvar


def vae_loss_fn(logits, targets, mu, logvar):
    recon_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl
    


def canonicalize(smiles):
    if smiles == None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

#gir : Updated may13 1:12pm
def clean(smiles):
    if smiles is None:
        return None
    if isinstance(smiles, (int, float)):
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.SanitizeMol(mol)
    mol_no_H = Chem.RemoveHs(mol)
    canon_smiles = Chem.MolToSmiles(mol_no_H, canonical=True)
    return canon_smiles

def syn_pred(prod_sm):
    """
    Decomposes a molecule using BRICS fragmentation into synthons.

    Args:
        prod_sm (str): SMILES string of the product molecule.

    Returns:
        List[str]: List of BRICS fragment SMILES strings.
    """

    cp_sm = clean(prod_sm)
    mol = Chem.MolFromSmiles(cp_sm)
    frags = list(Chem.BRICS.BRICSDecompose(mol))
    synths = []
    for f in frags:
      cf = clean(f)
      canf = canonicalize(f)
      synths.append(canf)

    return synths

def make_search_alg_df(dfFull, num_rxnt_lst):
    """
    Reads a reaction dataset and processes it for reactant-product modeling.

    Args:
        path_to_data (str): Path to the tab-separated .csv data file
        num_rxnt_lst (list): List of acceptable numbers of reactants (e.g., [2, 3])

    Returns:
        pd.DataFrame: Cleaned dataframe with split reactants and product columns
    """

    # Read the reaction dataset
    yield_data = dfFull

    # Extract only the ReactionSmiles and Yield columns
    y_df = yield_data[['ReactionSmiles', 'Yield']]

    # Split ReactionSmiles into Reactants, Solvent/Catalyst (ignored), and Product
    y_df[['Reactants', 'Solvent/Catalyst', 'Product']] = y_df['ReactionSmiles'].str.split('>', expand=True)

    # Clean up whitespace
    y_df['Reactants'] = y_df['Reactants'].str.strip()
    y_df['Product'] = y_df['Product'].str.strip()

    # Keep only Reactants, Product, and Yield columns
    y_df = y_df[['Reactants', 'Product', 'Yield']]

    # Calculate the number of reactants (counted by '.')
    y_df['num_reactants'] = y_df['Reactants'].str.split('.').apply(len)

    # Filter dataset to reactions with allowed numbers of reactants
    sdata = y_df[y_df['num_reactants'].isin(num_rxnt_lst)].reset_index(drop=True)

    # Limit the number of samples per reactant count (max 7000 per group)
    sdata = sdata.groupby('num_reactants').head(7000)

    # Calculate how many padded columns are needed (for alignment)
    sdata['needed_padd'] = max(num_rxnt_lst) - sdata['num_reactants']

    # Split reactants into separate columns (up to the max number of reactants)
    split_r_df = sdata['Reactants'].str.split('.', n=max(num_rxnt_lst)-1, expand=True)

    # Add the Product and Yield columns back
    split_r_df['Product'] = sdata['Product']
    split_r_df['Yield'] = sdata['Yield']

    # Rename reactant columns: rxnt 0, rxnt 1, rxnt 2, etc.
    rxtnt_col_names = [f'rxnt {i}' for i in range(max(num_rxnt_lst))]
    split_r_df.columns = rxtnt_col_names + ['Product', 'Yield']

    # Clean each SMILES string (assuming you have a clean() function)
    df = split_r_df.map(lambda x: clean(x))

    # Drop the Yield column — keeping only reactants and product
    dfrp = df.drop(columns=['Yield'])

    return dfrp


def get_unique_sets(dfrp):
    """
    Given a dataframe of split reactants and products,
    returns sets of unique SMILES strings.

    Args:
        dfrp (pd.DataFrame): DataFrame with columns ['rxnt 0', 'rxnt 1', ..., 'Product']

    Returns:
        set_all (set): Set of all unique SMILES strings (reactants + products)
        set_p (set): Set of unique product SMILES strings
        set_r (set): Set of unique reactant SMILES strings
    """

    # Unique SMILES across the whole dataframe
    unique_df = pd.unique(dfrp.values.ravel())
    unique_df = [s for s in unique_df if pd.notnull(s)]
    num_unique = len(unique_df)
    print(f"Number of unique SMILES strings (overall): {num_unique}")

    # Unique reactants
    rxtnts = dfrp.drop('Product', axis=1)
    unique_rxtnts = pd.unique(rxtnts.values.ravel())
    unique_rxtnts = [s for s in unique_rxtnts if pd.notnull(s)]
    num_runique = len(unique_rxtnts)
    print(f"Number of unique reactant SMILES strings: {num_runique}")

    # Unique products
    products = dfrp['Product']
    unique_prods = pd.unique(products.values.ravel())
    unique_prods = [s for s in unique_prods if pd.notnull(s)]
    num_punique = len(unique_prods)
    print(f"Number of unique product SMILES strings: {num_punique}")

    # Intersection counts
    count = sum(1 for mol in unique_prods if mol in unique_rxtnts)
    count_unique = len(set(unique_prods) & set(unique_rxtnts))
    print("Count (including duplicates):", count)
    print("Count (unique items only):", count_unique)

    # Create the sets
    set_all = set(unique_prods) | set(unique_rxtnts)  # union of products and reactants
    set_p = set(unique_prods)
    set_r = set(unique_rxtnts)

    return set_all, set_p, set_r


vocab_size=1193

SMI_REGEX_PATTERN = (
    r"(\[[^\]]+\]"            
    r"|Br?|Cl?"               
    r"|N|O|S|P|F|I"           
    r"|b|c|n|o|s|p"           
    r"|\(|\)|\."              
    r"|=|#|\+|\\|/"           
    r"|:|@|\?|>|~|\*|\$"      
    r"|\%\d{2}"               
    r"|[0-9])"                
)

token_to_idx={"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "#": 4, "(": 5, ")": 6, "1": 7, "2": 8, "3": 9, "4": 10, "=": 11, "Br": 12, "C": 13, "Cl": 14, "F": 15, "N": 16, "O": 17, "S": 18, "[1*]": 19, "[10*]": 20, "[11*]": 21, "[12*]": 22, "[13*]": 23, "[15*]": 24, "[3*]": 25, "[4*]": 26, "[5*]": 27, "[6*]": 28, "[7*]": 29, "[8*]": 30, "[Br:10]": 31, "[Br:11]": 32, "[Br:12]": 33, "[Br:13]": 34, "[Br:14]": 35, "[Br:15]": 36, "[Br:16]": 37, "[Br:17]": 38, "[Br:18]": 39, "[Br:19]": 40, "[Br:1]": 41, "[Br:20]": 42, "[Br:21]": 43, "[Br:22]": 44, "[Br:23]": 45, "[Br:24]": 46, "[Br:25]": 47, "[Br:26]": 48, "[Br:27]": 49, "[Br:28]": 50, "[Br:2]": 51, "[Br:30]": 52, "[Br:31]": 53, "[Br:32]": 54, "[Br:34]": 55, "[Br:35]": 56, "[Br:36]": 57, "[Br:37]": 58, "[Br:38]": 59, "[Br:40]": 60, "[Br:42]": 61, "[Br:43]": 62, "[Br:44]": 63, "[Br:45]": 64, "[Br:47]": 65, "[Br:48]": 66, "[Br:4]": 67, "[Br:50]": 68, "[Br:52]": 69, "[Br:53]": 70, "[Br:57]": 71, "[Br:58]": 72, "[Br:5]": 73, "[Br:61]": 74, "[Br:62]": 75, "[Br:6]": 76, "[Br:7]": 77, "[Br:8]": 78, "[Br:9]": 79, "[BrH:12]": 80, "[BrH:23]": 81, "[BrH:31]": 82, "[BrH:38]": 83, "[BrH:42]": 84, "[BrH:56]": 85, "[BrH:73]": 86, "[Br]": 87, "[C:106]": 88, "[C:108]": 89, "[C:109]": 90, "[C:10]": 91, "[C:113]": 92, "[C:11]": 93, "[C:12]": 94, "[C:13]": 95, "[C:14]": 96, "[C:15]": 97, "[C:16]": 98, "[C:17]": 99, "[C:18]": 100, "[C:19]": 101, "[C:1]": 102, "[C:20]": 103, "[C:21]": 104, "[C:22]": 105, "[C:23]": 106, "[C:24]": 107, "[C:25]": 108, "[C:26]": 109, "[C:27]": 110, "[C:28]": 111, "[C:29]": 112, "[C:2]": 113, "[C:30]": 114, "[C:31]": 115, "[C:32]": 116, "[C:33]": 117, "[C:34]": 118, "[C:35]": 119, "[C:36]": 120, "[C:37]": 121, "[C:38]": 122, "[C:39]": 123, "[C:3]": 124, "[C:40]": 125, "[C:41]": 126, "[C:42]": 127, "[C:43]": 128, "[C:44]": 129, "[C:45]": 130, "[C:46]": 131, "[C:47]": 132, "[C:48]": 133, "[C:49]": 134, "[C:4]": 135, "[C:50]": 136, "[C:51]": 137, "[C:52]": 138, "[C:53]": 139, "[C:54]": 140, "[C:55]": 141, "[C:56]": 142, "[C:57]": 143, "[C:58]": 144, "[C:59]": 145, "[C:5]": 146, "[C:60]": 147, "[C:61]": 148, "[C:62]": 149, "[C:63]": 150, "[C:64]": 151, "[C:65]": 152, "[C:66]": 153, "[C:67]": 154, "[C:68]": 155, "[C:69]": 156, "[C:6]": 157, "[C:71]": 158, "[C:72]": 159, "[C:73]": 160, "[C:74]": 161, "[C:75]": 162, "[C:76]": 163, "[C:77]": 164, "[C:78]": 165, "[C:79]": 166, "[C:7]": 167, "[C:83]": 168, "[C:86]": 169, "[C:87]": 170, "[C:89]": 171, "[C:8]": 172, "[C:92]": 173, "[C:93]": 174, "[C:94]": 175, "[C:9]": 176, "[CH2:101]": 177, "[CH2:105]": 178, "[CH2:106]": 179, "[CH2:107]": 180, "[CH2:10]": 181, "[CH2:111]": 182, "[CH2:112]": 183, "[CH2:115]": 184, "[CH2:116]": 185, "[CH2:11]": 186, "[CH2:12]": 187, "[CH2:13]": 188, "[CH2:14]": 189, "[CH2:15]": 190, "[CH2:16]": 191, "[CH2:17]": 192, "[CH2:18]": 193, "[CH2:19]": 194, "[CH2:1]": 195, "[CH2:20]": 196, "[CH2:21]": 197, "[CH2:22]": 198, "[CH2:23]": 199, "[CH2:24]": 200, "[CH2:25]": 201, "[CH2:26]": 202, "[CH2:27]": 203, "[CH2:284]": 204, "[CH2:28]": 205, "[CH2:29]": 206, "[CH2:2]": 207, "[CH2:30]": 208, "[CH2:31]": 209, "[CH2:32]": 210, "[CH2:33]": 211, "[CH2:34]": 212, "[CH2:35]": 213, "[CH2:36]": 214, "[CH2:37]": 215, "[CH2:38]": 216, "[CH2:39]": 217, "[CH2:3]": 218, "[CH2:40]": 219, "[CH2:41]": 220, "[CH2:42]": 221, "[CH2:43]": 222, "[CH2:44]": 223, "[CH2:45]": 224, "[CH2:46]": 225, "[CH2:47]": 226, "[CH2:48]": 227, "[CH2:49]": 228, "[CH2:4]": 229, "[CH2:50]": 230, "[CH2:51]": 231, "[CH2:52]": 232, "[CH2:53]": 233, "[CH2:54]": 234, "[CH2:55]": 235, "[CH2:56]": 236, "[CH2:57]": 237, "[CH2:58]": 238, "[CH2:59]": 239, "[CH2:5]": 240, "[CH2:60]": 241, "[CH2:61]": 242, "[CH2:62]": 243, "[CH2:63]": 244, "[CH2:64]": 245, "[CH2:65]": 246, "[CH2:66]": 247, "[CH2:67]": 248, "[CH2:68]": 249, "[CH2:69]": 250, "[CH2:6]": 251, "[CH2:70]": 252, "[CH2:71]": 253, "[CH2:72]": 254, "[CH2:73]": 255, "[CH2:74]": 256, "[CH2:75]": 257, "[CH2:76]": 258, "[CH2:77]": 259, "[CH2:78]": 260, "[CH2:79]": 261, "[CH2:7]": 262, "[CH2:80]": 263, "[CH2:81]": 264, "[CH2:82]": 265, "[CH2:83]": 266, "[CH2:84]": 267, "[CH2:85]": 268, "[CH2:86]": 269, "[CH2:87]": 270, "[CH2:88]": 271, "[CH2:89]": 272, "[CH2:8]": 273, "[CH2:90]": 274, "[CH2:91]": 275, "[CH2:92]": 276, "[CH2:94]": 277, "[CH2:95]": 278, "[CH2:96]": 279, "[CH2:97]": 280, "[CH2:98]": 281, "[CH2:99]": 282, "[CH2:9]": 283, "[CH3:100]": 284, "[CH3:101]": 285, "[CH3:104]": 286, "[CH3:105]": 287, "[CH3:106]": 288, "[CH3:10]": 289, "[CH3:11]": 290, "[CH3:12]": 291, "[CH3:13]": 292, "[CH3:14]": 293, "[CH3:15]": 294, "[CH3:16]": 295, "[CH3:17]": 296, "[CH3:18]": 297, "[CH3:19]": 298, "[CH3:1]": 299, "[CH3:20]": 300, "[CH3:21]": 301, "[CH3:22]": 302, "[CH3:23]": 303, "[CH3:247]": 304, "[CH3:24]": 305, "[CH3:25]": 306, "[CH3:26]": 307, "[CH3:27]": 308, "[CH3:28]": 309, "[CH3:29]": 310, "[CH3:2]": 311, "[CH3:30]": 312, "[CH3:31]": 313, "[CH3:32]": 314, "[CH3:33]": 315, "[CH3:34]": 316, "[CH3:35]": 317, "[CH3:36]": 318, "[CH3:37]": 319, "[CH3:38]": 320, "[CH3:39]": 321, "[CH3:3]": 322, "[CH3:40]": 323, "[CH3:41]": 324, "[CH3:42]": 325, "[CH3:43]": 326, "[CH3:44]": 327, "[CH3:45]": 328, "[CH3:46]": 329, "[CH3:47]": 330, "[CH3:48]": 331, "[CH3:49]": 332, "[CH3:4]": 333, "[CH3:50]": 334, "[CH3:51]": 335, "[CH3:52]": 336, "[CH3:53]": 337, "[CH3:54]": 338, "[CH3:55]": 339, "[CH3:56]": 340, "[CH3:57]": 341, "[CH3:58]": 342, "[CH3:59]": 343, "[CH3:5]": 344, "[CH3:60]": 345, "[CH3:61]": 346, "[CH3:62]": 347, "[CH3:63]": 348, "[CH3:64]": 349, "[CH3:65]": 350, "[CH3:66]": 351, "[CH3:67]": 352, "[CH3:68]": 353, "[CH3:69]": 354, "[CH3:6]": 355, "[CH3:70]": 356, "[CH3:71]": 357, "[CH3:72]": 358, "[CH3:73]": 359, "[CH3:74]": 360, "[CH3:75]": 361, "[CH3:76]": 362, "[CH3:77]": 363, "[CH3:78]": 364, "[CH3:79]": 365, "[CH3:7]": 366, "[CH3:81]": 367, "[CH3:82]": 368, "[CH3:83]": 369, "[CH3:84]": 370, "[CH3:85]": 371, "[CH3:86]": 372, "[CH3:87]": 373, "[CH3:88]": 374, "[CH3:89]": 375, "[CH3:8]": 376, "[CH3:91]": 377, "[CH3:92]": 378, "[CH3:93]": 379, "[CH3:95]": 380, "[CH3:96]": 381, "[CH3:97]": 382, "[CH3:98]": 383, "[CH3:99]": 384, "[CH3:9]": 385, "[CH4:12]": 386, "[CH:104]": 387, "[CH:109]": 388, "[CH:10]": 389, "[CH:111]": 390, "[CH:11]": 391, "[CH:12]": 392, "[CH:13]": 393, "[CH:14]": 394, "[CH:15]": 395, "[CH:16]": 396, "[CH:17]": 397, "[CH:18]": 398, "[CH:19]": 399, "[CH:1]": 400, "[CH:20]": 401, "[CH:21]": 402, "[CH:22]": 403, "[CH:23]": 404, "[CH:24]": 405, "[CH:25]": 406, "[CH:26]": 407, "[CH:275]": 408, "[CH:276]": 409, "[CH:278]": 410, "[CH:27]": 411, "[CH:280]": 412, "[CH:282]": 413, "[CH:28]": 414, "[CH:29]": 415, "[CH:2]": 416, "[CH:30]": 417, "[CH:31]": 418, "[CH:32]": 419, "[CH:33]": 420, "[CH:34]": 421, "[CH:35]": 422, "[CH:36]": 423, "[CH:37]": 424, "[CH:38]": 425, "[CH:39]": 426, "[CH:3]": 427, "[CH:40]": 428, "[CH:41]": 429, "[CH:42]": 430, "[CH:43]": 431, "[CH:44]": 432, "[CH:45]": 433, "[CH:46]": 434, "[CH:47]": 435, "[CH:48]": 436, "[CH:49]": 437, "[CH:4]": 438, "[CH:50]": 439, "[CH:51]": 440, "[CH:52]": 441, "[CH:53]": 442, "[CH:54]": 443, "[CH:55]": 444, "[CH:56]": 445, "[CH:57]": 446, "[CH:58]": 447, "[CH:59]": 448, "[CH:5]": 449, "[CH:60]": 450, "[CH:61]": 451, "[CH:62]": 452, "[CH:63]": 453, "[CH:65]": 454, "[CH:67]": 455, "[CH:68]": 456, "[CH:69]": 457, "[CH:6]": 458, "[CH:70]": 459, "[CH:72]": 460, "[CH:74]": 461, "[CH:75]": 462, "[CH:76]": 463, "[CH:79]": 464, "[CH:7]": 465, "[CH:80]": 466, "[CH:82]": 467, "[CH:83]": 468, "[CH:84]": 469, "[CH:87]": 470, "[CH:89]": 471, "[CH:8]": 472, "[CH:94]": 473, "[CH:9]": 474, "[CH]": 475, "[C]": 476, "[Cl:107]": 477, "[Cl:10]": 478, "[Cl:11]": 479, "[Cl:12]": 480, "[Cl:131]": 481, "[Cl:13]": 482, "[Cl:14]": 483, "[Cl:15]": 484, "[Cl:16]": 485, "[Cl:17]": 486, "[Cl:18]": 487, "[Cl:19]": 488, "[Cl:1]": 489, "[Cl:20]": 490, "[Cl:21]": 491, "[Cl:22]": 492, "[Cl:23]": 493, "[Cl:24]": 494, "[Cl:25]": 495, "[Cl:26]": 496, "[Cl:27]": 497, "[Cl:28]": 498, "[Cl:29]": 499, "[Cl:2]": 500, "[Cl:30]": 501, "[Cl:31]": 502, "[Cl:32]": 503, "[Cl:33]": 504, "[Cl:34]": 505, "[Cl:35]": 506, "[Cl:36]": 507, "[Cl:37]": 508, "[Cl:38]": 509, "[Cl:39]": 510, "[Cl:3]": 511, "[Cl:40]": 512, "[Cl:41]": 513, "[Cl:42]": 514, "[Cl:43]": 515, "[Cl:44]": 516, "[Cl:45]": 517, "[Cl:46]": 518, "[Cl:47]": 519, "[Cl:48]": 520, "[Cl:49]": 521, "[Cl:4]": 522, "[Cl:50]": 523, "[Cl:52]": 524, "[Cl:53]": 525, "[Cl:56]": 526, "[Cl:57]": 527, "[Cl:58]": 528, "[Cl:59]": 529, "[Cl:5]": 530, "[Cl:60]": 531, "[Cl:62]": 532, "[Cl:63]": 533, "[Cl:64]": 534, "[Cl:65]": 535, "[Cl:69]": 536, "[Cl:6]": 537, "[Cl:70]": 538, "[Cl:75]": 539, "[Cl:77]": 540, "[Cl:7]": 541, "[Cl:80]": 542, "[Cl:8]": 543, "[Cl:9]": 544, "[ClH:104]": 545, "[ClH:120]": 546, "[ClH:123]": 547, "[ClH:12]": 548, "[ClH:26]": 549, "[ClH:27]": 550, "[ClH:37]": 551, "[ClH:52]": 552, "[ClH:65]": 553, "[ClH:76]": 554, "[ClH:85]": 555, "[F:10]": 556, "[F:11]": 557, "[F:12]": 558, "[F:13]": 559, "[F:14]": 560, "[F:15]": 561, "[F:16]": 562, "[F:17]": 563, "[F:18]": 564, "[F:19]": 565, "[F:1]": 566, "[F:20]": 567, "[F:21]": 568, "[F:22]": 569, "[F:23]": 570, "[F:24]": 571, "[F:25]": 572, "[F:26]": 573, "[F:27]": 574, "[F:28]": 575, "[F:29]": 576, "[F:2]": 577, "[F:30]": 578, "[F:31]": 579, "[F:32]": 580, "[F:33]": 581, "[F:34]": 582, "[F:35]": 583, "[F:36]": 584, "[F:37]": 585, "[F:38]": 586, "[F:39]": 587, "[F:3]": 588, "[F:40]": 589, "[F:41]": 590, "[F:42]": 591, "[F:43]": 592, "[F:44]": 593, "[F:45]": 594, "[F:46]": 595, "[F:47]": 596, "[F:48]": 597, "[F:49]": 598, "[F:4]": 599, "[F:50]": 600, "[F:51]": 601, "[F:52]": 602, "[F:53]": 603, "[F:54]": 604, "[F:55]": 605, "[F:56]": 606, "[F:57]": 607, "[F:58]": 608, "[F:59]": 609, "[F:5]": 610, "[F:60]": 611, "[F:61]": 612, "[F:62]": 613, "[F:63]": 614, "[F:64]": 615, "[F:65]": 616, "[F:66]": 617, "[F:67]": 618, "[F:68]": 619, "[F:69]": 620, "[F:6]": 621, "[F:70]": 622, "[F:71]": 623, "[F:72]": 624, "[F:73]": 625, "[F:74]": 626, "[F:78]": 627, "[F:79]": 628, "[F:7]": 629, "[F:80]": 630, "[F:81]": 631, "[F:82]": 632, "[F:87]": 633, "[F:88]": 634, "[F:8]": 635, "[F:9]": 636, "[FH:10]": 637, "[FH:35]": 638, "[FH:51]": 639, "[N:107]": 640, "[N:10]": 641, "[N:11]": 642, "[N:12]": 643, "[N:13]": 644, "[N:14]": 645, "[N:15]": 646, "[N:16]": 647, "[N:17]": 648, "[N:18]": 649, "[N:19]": 650, "[N:1]": 651, "[N:20]": 652, "[N:21]": 653, "[N:22]": 654, "[N:23]": 655, "[N:24]": 656, "[N:25]": 657, "[N:26]": 658, "[N:27]": 659, "[N:28]": 660, "[N:29]": 661, "[N:2]": 662, "[N:30]": 663, "[N:31]": 664, "[N:32]": 665, "[N:33]": 666, "[N:34]": 667, "[N:35]": 668, "[N:36]": 669, "[N:37]": 670, "[N:38]": 671, "[N:39]": 672, "[N:3]": 673, "[N:40]": 674, "[N:41]": 675, "[N:42]": 676, "[N:43]": 677, "[N:44]": 678, "[N:45]": 679, "[N:46]": 680, "[N:47]": 681, "[N:48]": 682, "[N:49]": 683, "[N:4]": 684, "[N:50]": 685, "[N:51]": 686, "[N:52]": 687, "[N:53]": 688, "[N:54]": 689, "[N:55]": 690, "[N:57]": 691, "[N:58]": 692, "[N:59]": 693, "[N:5]": 694, "[N:60]": 695, "[N:61]": 696, "[N:62]": 697, "[N:63]": 698, "[N:64]": 699, "[N:65]": 700, "[N:66]": 701, "[N:69]": 702, "[N:6]": 703, "[N:71]": 704, "[N:73]": 705, "[N:76]": 706, "[N:77]": 707, "[N:78]": 708, "[N:7]": 709, "[N:84]": 710, "[N:8]": 711, "[N:9]": 712, "[NH2:103]": 713, "[NH2:105]": 714, "[NH2:10]": 715, "[NH2:113]": 716, "[NH2:117]": 717, "[NH2:11]": 718, "[NH2:12]": 719, "[NH2:13]": 720, "[NH2:14]": 721, "[NH2:15]": 722, "[NH2:16]": 723, "[NH2:17]": 724, "[NH2:18]": 725, "[NH2:19]": 726, "[NH2:1]": 727, "[NH2:20]": 728, "[NH2:21]": 729, "[NH2:22]": 730, "[NH2:23]": 731, "[NH2:24]": 732, "[NH2:25]": 733, "[NH2:26]": 734, "[NH2:27]": 735, "[NH2:28]": 736, "[NH2:29]": 737, "[NH2:2]": 738, "[NH2:30]": 739, "[NH2:31]": 740, "[NH2:32]": 741, "[NH2:33]": 742, "[NH2:34]": 743, "[NH2:35]": 744, "[NH2:36]": 745, "[NH2:37]": 746, "[NH2:38]": 747, "[NH2:39]": 748, "[NH2:3]": 749, "[NH2:40]": 750, "[NH2:41]": 751, "[NH2:42]": 752, "[NH2:43]": 753, "[NH2:44]": 754, "[NH2:45]": 755, "[NH2:46]": 756, "[NH2:47]": 757, "[NH2:48]": 758, "[NH2:49]": 759, "[NH2:4]": 760, "[NH2:50]": 761, "[NH2:51]": 762, "[NH2:52]": 763, "[NH2:53]": 764, "[NH2:54]": 765, "[NH2:55]": 766, "[NH2:56]": 767, "[NH2:57]": 768, "[NH2:58]": 769, "[NH2:59]": 770, "[NH2:5]": 771, "[NH2:60]": 772, "[NH2:61]": 773, "[NH2:62]": 774, "[NH2:63]": 775, "[NH2:64]": 776, "[NH2:65]": 777, "[NH2:66]": 778, "[NH2:67]": 779, "[NH2:68]": 780, "[NH2:69]": 781, "[NH2:6]": 782, "[NH2:70]": 783, "[NH2:71]": 784, "[NH2:72]": 785, "[NH2:73]": 786, "[NH2:74]": 787, "[NH2:75]": 788, "[NH2:76]": 789, "[NH2:77]": 790, "[NH2:78]": 791, "[NH2:7]": 792, "[NH2:8]": 793, "[NH2:9]": 794, "[NH3:25]": 795, "[NH3:27]": 796, "[NH3:32]": 797, "[NH3:33]": 798, "[NH3:38]": 799, "[NH3:40]": 800, "[NH3:56]": 801, "[NH3:6]": 802, "[NH3:70]": 803, "[NH:10]": 804, "[NH:11]": 805, "[NH:12]": 806, "[NH:13]": 807, "[NH:14]": 808, "[NH:15]": 809, "[NH:16]": 810, "[NH:17]": 811, "[NH:18]": 812, "[NH:19]": 813, "[NH:1]": 814, "[NH:20]": 815, "[NH:21]": 816, "[NH:22]": 817, "[NH:23]": 818, "[NH:24]": 819, "[NH:25]": 820, "[NH:26]": 821, "[NH:27]": 822, "[NH:28]": 823, "[NH:29]": 824, "[NH:2]": 825, "[NH:30]": 826, "[NH:31]": 827, "[NH:32]": 828, "[NH:33]": 829, "[NH:34]": 830, "[NH:35]": 831, "[NH:36]": 832, "[NH:37]": 833, "[NH:38]": 834, "[NH:39]": 835, "[NH:3]": 836, "[NH:40]": 837, "[NH:41]": 838, "[NH:42]": 839, "[NH:43]": 840, "[NH:44]": 841, "[NH:45]": 842, "[NH:46]": 843, "[NH:47]": 844, "[NH:48]": 845, "[NH:49]": 846, "[NH:4]": 847, "[NH:50]": 848, "[NH:51]": 849, "[NH:52]": 850, "[NH:53]": 851, "[NH:54]": 852, "[NH:55]": 853, "[NH:56]": 854, "[NH:57]": 855, "[NH:58]": 856, "[NH:59]": 857, "[NH:5]": 858, "[NH:60]": 859, "[NH:61]": 860, "[NH:62]": 861, "[NH:63]": 862, "[NH:64]": 863, "[NH:66]": 864, "[NH:67]": 865, "[NH:68]": 866, "[NH:69]": 867, "[NH:6]": 868, "[NH:70]": 869, "[NH:72]": 870, "[NH:73]": 871, "[NH:74]": 872, "[NH:76]": 873, "[NH:77]": 874, "[NH:7]": 875, "[NH:81]": 876, "[NH:82]": 877, "[NH:84]": 878, "[NH:85]": 879, "[NH:86]": 880, "[NH:8]": 881, "[NH:9]": 882, "[O:100]": 883, "[O:101]": 884, "[O:108]": 885, "[O:10]": 886, "[O:110]": 887, "[O:114]": 888, "[O:116]": 889, "[O:11]": 890, "[O:12]": 891, "[O:13]": 892, "[O:14]": 893, "[O:15]": 894, "[O:16]": 895, "[O:17]": 896, "[O:18]": 897, "[O:19]": 898, "[O:1]": 899, "[O:20]": 900, "[O:21]": 901, "[O:22]": 902, "[O:23]": 903, "[O:24]": 904, "[O:257]": 905, "[O:25]": 906, "[O:26]": 907, "[O:274]": 908, "[O:27]": 909, "[O:28]": 910, "[O:29]": 911, "[O:2]": 912, "[O:30]": 913, "[O:31]": 914, "[O:32]": 915, "[O:33]": 916, "[O:34]": 917, "[O:35]": 918, "[O:36]": 919, "[O:37]": 920, "[O:38]": 921, "[O:39]": 922, "[O:3]": 923, "[O:40]": 924, "[O:41]": 925, "[O:42]": 926, "[O:43]": 927, "[O:44]": 928, "[O:45]": 929, "[O:46]": 930, "[O:47]": 931, "[O:48]": 932, "[O:49]": 933, "[O:4]": 934, "[O:50]": 935, "[O:51]": 936, "[O:52]": 937, "[O:53]": 938, "[O:54]": 939, "[O:55]": 940, "[O:56]": 941, "[O:57]": 942, "[O:58]": 943, "[O:59]": 944, "[O:5]": 945, "[O:60]": 946, "[O:61]": 947, "[O:62]": 948, "[O:63]": 949, "[O:64]": 950, "[O:65]": 951, "[O:66]": 952, "[O:67]": 953, "[O:68]": 954, "[O:69]": 955, "[O:6]": 956, "[O:70]": 957, "[O:71]": 958, "[O:72]": 959, "[O:73]": 960, "[O:74]": 961, "[O:75]": 962, "[O:76]": 963, "[O:77]": 964, "[O:78]": 965, "[O:79]": 966, "[O:7]": 967, "[O:80]": 968, "[O:81]": 969, "[O:82]": 970, "[O:84]": 971, "[O:86]": 972, "[O:87]": 973, "[O:88]": 974, "[O:8]": 975, "[O:90]": 976, "[O:92]": 977, "[O:93]": 978, "[O:94]": 979, "[O:95]": 980, "[O:99]": 981, "[O:9]": 982, "[OH2:118]": 983, "[OH2:24]": 984, "[OH2:26]": 985, "[OH2:2]": 986, "[OH2:53]": 987, "[OH2:55]": 988, "[OH2:67]": 989, "[OH2:73]": 990, "[OH2:78]": 991, "[OH2:9]": 992, "[OH:107]": 993, "[OH:10]": 994, "[OH:111]": 995, "[OH:112]": 996, "[OH:115]": 997, "[OH:117]": 998, "[OH:11]": 999, "[OH:12]": 1000, "[OH:13]": 1001, "[OH:14]": 1002, "[OH:15]": 1003, "[OH:16]": 1004, "[OH:17]": 1005, "[OH:18]": 1006, "[OH:19]": 1007, "[OH:1]": 1008, "[OH:20]": 1009, "[OH:21]": 1010, "[OH:22]": 1011, "[OH:23]": 1012, "[OH:24]": 1013, "[OH:25]": 1014, "[OH:26]": 1015, "[OH:277]": 1016, "[OH:279]": 1017, "[OH:27]": 1018, "[OH:281]": 1019, "[OH:283]": 1020, "[OH:285]": 1021, "[OH:28]": 1022, "[OH:29]": 1023, "[OH:2]": 1024, "[OH:30]": 1025, "[OH:31]": 1026, "[OH:32]": 1027, "[OH:33]": 1028, "[OH:34]": 1029, "[OH:35]": 1030, "[OH:36]": 1031, "[OH:37]": 1032, "[OH:38]": 1033, "[OH:39]": 1034, "[OH:3]": 1035, "[OH:40]": 1036, "[OH:41]": 1037, "[OH:42]": 1038, "[OH:43]": 1039, "[OH:44]": 1040, "[OH:45]": 1041, "[OH:46]": 1042, "[OH:47]": 1043, "[OH:48]": 1044, "[OH:49]": 1045, "[OH:4]": 1046, "[OH:50]": 1047, "[OH:51]": 1048, "[OH:52]": 1049, "[OH:53]": 1050, "[OH:54]": 1051, "[OH:55]": 1052, "[OH:56]": 1053, "[OH:57]": 1054, "[OH:58]": 1055, "[OH:59]": 1056, "[OH:5]": 1057, "[OH:60]": 1058, "[OH:61]": 1059, "[OH:62]": 1060, "[OH:63]": 1061, "[OH:64]": 1062, "[OH:65]": 1063, "[OH:66]": 1064, "[OH:67]": 1065, "[OH:68]": 1066, "[OH:69]": 1067, "[OH:6]": 1068, "[OH:71]": 1069, "[OH:72]": 1070, "[OH:73]": 1071, "[OH:75]": 1072, "[OH:79]": 1073, "[OH:7]": 1074, "[OH:80]": 1075, "[OH:81]": 1076, "[OH:82]": 1077, "[OH:83]": 1078, "[OH:84]": 1079, "[OH:88]": 1080, "[OH:89]": 1081, "[OH:8]": 1082, "[OH:91]": 1083, "[OH:94]": 1084, "[OH:96]": 1085, "[OH:9]": 1086, "[O]": 1087, "[S:10]": 1088, "[S:11]": 1089, "[S:12]": 1090, "[S:13]": 1091, "[S:14]": 1092, "[S:15]": 1093, "[S:16]": 1094, "[S:17]": 1095, "[S:18]": 1096, "[S:19]": 1097, "[S:1]": 1098, "[S:20]": 1099, "[S:21]": 1100, "[S:22]": 1101, "[S:23]": 1102, "[S:24]": 1103, "[S:25]": 1104, "[S:26]": 1105, "[S:27]": 1106, "[S:28]": 1107, "[S:29]": 1108, "[S:2]": 1109, "[S:30]": 1110, "[S:31]": 1111, "[S:32]": 1112, "[S:33]": 1113, "[S:34]": 1114, "[S:35]": 1115, "[S:36]": 1116, "[S:37]": 1117, "[S:38]": 1118, "[S:39]": 1119, "[S:3]": 1120, "[S:40]": 1121, "[S:41]": 1122, "[S:42]": 1123, "[S:43]": 1124, "[S:44]": 1125, "[S:45]": 1126, "[S:46]": 1127, "[S:47]": 1128, "[S:48]": 1129, "[S:49]": 1130, "[S:4]": 1131, "[S:50]": 1132, "[S:51]": 1133, "[S:52]": 1134, "[S:53]": 1135, "[S:54]": 1136, "[S:56]": 1137, "[S:59]": 1138, "[S:5]": 1139, "[S:61]": 1140, "[S:63]": 1141, "[S:64]": 1142, "[S:65]": 1143, "[S:67]": 1144, "[S:6]": 1145, "[S:73]": 1146, "[S:7]": 1147, "[S:8]": 1148, "[S:9]": 1149, "[SH2:32]": 1150, "[SH:10]": 1151, "[SH:11]": 1152, "[SH:12]": 1153, "[SH:13]": 1154, "[SH:14]": 1155, "[SH:15]": 1156, "[SH:16]": 1157, "[SH:17]": 1158, "[SH:19]": 1159, "[SH:1]": 1160, "[SH:21]": 1161, "[SH:22]": 1162, "[SH:23]": 1163, "[SH:24]": 1164, "[SH:25]": 1165, "[SH:26]": 1166, "[SH:27]": 1167, "[SH:28]": 1168, "[SH:2]": 1169, "[SH:30]": 1170, "[SH:31]": 1171, "[SH:32]": 1172, "[SH:33]": 1173, "[SH:34]": 1174, "[SH:35]": 1175, "[SH:36]": 1176, "[SH:37]": 1177, "[SH:38]": 1178, "[SH:39]": 1179, "[SH:3]": 1180, "[SH:40]": 1181, "[SH:41]": 1182, "[SH:43]": 1183, "[SH:45]": 1184, "[SH:49]": 1185, "[SH:5]": 1186, "[SH:64]": 1187, "[SH:6]": 1188, "[SH:7]": 1189, "[SH:8]": 1190, "[SH:9]": 1191, "[SH]": 1192}

idx_to_token = {i: tok for tok, i in token_to_idx.items()}

vocab_size=1193

def tokenize_smiles(smi):
    tokens = re.findall(SMI_REGEX_PATTERN, smi)
    return tokens
    
def detokenize_smiles(tokens):
    return ''.join(tokens)

def sample_from_latent(model, z, max_len, k=10):
    model.eval()
    with torch.no_grad():
        batch_size = z.size(0)
        h = model.latent_to_hidden(z).unsqueeze(0) 
        
        input_char = torch.full((batch_size,), token_to_idx['<bos>'], dtype=torch.long, device=z.device)  # [batch]
        generated = []

        for _ in range(max_len):
            emb = model.embed(input_char).unsqueeze(1)  
            output, h = model.decoder_rnn(emb, h)       
            logits = model.output(output.squeeze(1))   
            next_char = top_k_sampling(logits, k=k).squeeze(1) 
            generated.append(next_char.unsqueeze(1))
            input_char = next_char

        generated = torch.cat(generated, dim=1)  
        return generated
        
def decode_smiles(tensor):
    tokens = []
    for idx in tensor:
        tok = idx_to_token[int(idx)]
        if tok == '<eos>':
            break
        tokens.append(tok)
    return detokenize_smiles(tokens)



def top_k_sampling(logits, k=10):
  values, indices = torch.topk(logits, k)
  probs = F.softmax(values, dim=-1)
  sample = torch.multinomial(probs, num_samples=1)
  return indices.gather(-1, sample)

def generate_analogs(model, query_smiles, num_samples=5, noise_scale=0.3, k=10):
  from rdkit import Chem

  model.eval()
  tokens = [token_to_idx['<bos>']] + [token_to_idx.get(c, token_to_idx['<unk>']) for c in query_smiles] + [token_to_idx['<eos>']]
  tokens += [token_to_idx['<pad>']] * (50 - len(tokens))
  tokens = torch.tensor(tokens[:-1]).unsqueeze(0).to(next(model.parameters()).device)

  valid_smiles = []
  attempts = 0
  max_attempts = num_samples * 10  # Allow retries in case of invalid outputs

  with torch.no_grad():
      mu, logvar = model.encode(tokens)
      z = model.reparam(mu, logvar)

      while len(valid_smiles) < num_samples and attempts < max_attempts:
          noise = torch.randn((1, z.size(1)), device=z.device) * noise_scale
          z_sample = z + noise
          sampled = sample_from_latent(model, z_sample, max_len=50, k=k)
          smi = decode_smiles(sampled[0])
          if Chem.MolFromSmiles(smi):
              valid_smiles.append(smi)
          attempts += 1

  if len(valid_smiles) > 0:
      return valid_smiles
  else:
      return ["No valid SMILES generated"]



def visualize_smiles_list(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    mols = [m for m in mols if m is not None]
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 200))
        display(img)


def find_best_analog(
    query,
    model,
    num_samples=100,
    noise_scale=0.5,
    threshold=0.1,
    max_nodes=32,
    vocab_size=20,
    device= device
):
    # Step 3: Compare using Tanimoto similarity
    analogs = generate_analogs(model,query, num_samples=num_samples, noise_scale=noise_scale, k=10)

    query_mol = Chem.MolFromSmiles(query)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2)


    similar_mols = []
    for mol in analogs:
        mol = Chem.MolFromSmiles(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2)
        sim = DataStructs.TanimotoSimilarity(query_fp, mol_fp)
        if sim >= threshold and sim != 1:
            similar_mols.append((Chem.MolToSmiles(mol), sim))

    ana = sorted(similar_mols, key=lambda x: x[1], reverse=True)
    return ana[0][0]


def sim_search(synthon_sm, model,set_mols, tanimoto_threshold=0.3):

    """
    find sunthon containing molecules returns highest tanimoto in dataset
    or reasonable analog, or returns given synthon.

    arguments:
        synthon_sm (str): Synthon SMILES.
        set_mols (set): Set of molecule SMILES to search.
        tanimoto_threshold (float): Minimum Tanimoto score to accept.

    returns:
        best_match (str or None): SMILES of best matching molecule, or None.
    """

    gen = GetMorganGenerator(radius=2, fpSize=2048)
    synthon_mol = Chem.MolFromSmiles(synthon_sm)
    fp_s = gen.GetFingerprint(synthon_mol)

    # build list of substructure matches
    substr_matches = []
    for mol_sm in set_mols:
        mol = Chem.MolFromSmiles(mol_sm)
        if mol.HasSubstructMatch(synthon_mol):
            substr_matches.append(mol_sm)

    if not substr_matches:
      #print("No substructure matches found. Resorting to Full Tanimoto search.")

      # tanimot with entire datafraem
      best_score = -1
      best_match = None

      for mol_sm in set_mols:
          mol = Chem.MolFromSmiles(mol_sm)
          fp_c = gen.GetFingerprint(mol)
          ts = DataStructs.TanimotoSimilarity(fp_s, fp_c)
          if ts > best_score:
              best_score = ts
              best_match = mol_sm

      if best_score >= tanimoto_threshold:
          print(f"~Returning Tanimoto match~: score of: {best_score:.6f}")
          return best_match
      else:
          #print(f"No Tanimoto, best was {best_score:.6f} Resorting to Analog Generator")
          ana =  find_best_analog(query=synthon_sm,model=model,num_samples=100,
                                  noise_scale=1.0,threshold=0.01,
                                  device=device)
          if ana is None:
            print("~Returning Synthon~: no reasonable analog")
            return synthon_sm
          else:
            print(f"~Returning Analog~:")
            return ana

    # only runs in substructure matches are found
    # Substructre & Tanimoto Ranking
    best_score = -1
    best_match = None

    for match_sm in substr_matches:
        match_mol = Chem.MolFromSmiles(match_sm)
        fp_match = gen.GetFingerprint(match_mol)

        ts = DataStructs.TanimotoSimilarity(fp_s, fp_match)

        if ts > best_score: # if better found then update
            best_score = ts
            best_match = match_sm

    if best_score >= tanimoto_threshold:
        print(f"~Returning Substructre Match~: Tanimoto: {best_score:.6f}")
        return best_match
    else:
        print("~Returning Synthon~")
        return synthon_sm

        
        
        

def path_viz(pathway):
    overall_yield = 1.00

    for elem in pathway:
      if elem == pathway[0]:
        print("Target")
      if type(elem) == str:
        display(Draw.MolToImage(Chem.MolFromSmiles(elem), size=(300, 300)))
      if type(elem) == float:
        overall_yield *= (elem/100)
        print(elem)
      if type(elem) == list:
        for j in elem:
          display(Draw.MolToImage(Chem.MolFromSmiles(j), size=(300, 300)))
    print("overall yield: ", overall_yield)