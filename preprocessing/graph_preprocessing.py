from . import preprocessing_utils
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
import torch


def get_atom_features(atom):
    #     result = []
    torch_result = torch.tensor([])

    atomic_number = torch.tensor([atom.GetAtomicNum()]) / 100.0
    torch_result = torch.cat((torch_result, atomic_number), 0)

    PERMITTED_LIST_OF_ATOMAS = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Unknown']
    atom_dict = {elem: index for index, elem in enumerate(PERMITTED_LIST_OF_ATOMAS)}

    atom_type_hot = preprocessing_utils.one_hot_encoding(atom_dict.get(atom.GetSymbol(), len(atom_dict)),
                                                         len(PERMITTED_LIST_OF_ATOMAS))

    torch_result = torch.cat((torch_result, atom_type_hot), 0)

    total_valence = atom.GetTotalValence()
    total_valence_hot = preprocessing_utils.one_hot_encoding(total_valence, 8)
    torch_result = torch.cat((torch_result, total_valence_hot), 0)

    is_aromatic_hot = preprocessing_utils.one_hot_encoding(atom.GetIsAromatic(), 1)
    torch_result = torch.cat((torch_result, is_aromatic_hot), 0)

    HYBRIDIZATIONS = [Chem.HybridizationType.UNSPECIFIED,
                      Chem.HybridizationType.S,
                      Chem.HybridizationType.SP,
                      Chem.HybridizationType.SP2,
                      Chem.HybridizationType.SP3,
                      Chem.HybridizationType.SP3D,
                      Chem.HybridizationType.SP3D2,
                      Chem.HybridizationType.OTHER]
    hybridization_dict = {elem: index for index, elem in enumerate(HYBRIDIZATIONS)}
    hybridization = atom.GetHybridization()
    hybridization_hot = preprocessing_utils.one_hot_encoding(
        hybridization_dict.get(hybridization, len(hybridization_dict)), 8)
    torch_result = torch.cat((torch_result, hybridization_hot), 0)

    # we adapt scale, the output of method GetFormalCharge is [-2, -1, 0, 1, 2]
    formal_charge = atom.GetFormalCharge()

    formal_charge_hot = preprocessing_utils.one_hot_encoding(formal_charge + 2, 5)
    torch_result = torch.cat((torch_result, formal_charge_hot), 0)

    default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())

    default_valence_hot = preprocessing_utils.one_hot_encoding(default_valence, 8)
    torch_result = torch.cat((torch_result, default_valence_hot), 0)

    ring_size = [atom.IsInRingSize(r) for r in range(3, 8)]

    ring_size_hot = torch.tensor(ring_size).type(torch.float)
    torch_result = torch.cat((torch_result, ring_size_hot), 0)

    attached_H = np.sum([neighbour.GetAtomicNum() == 1 for neighbour in atom.GetNeighbors()], dtype=np.uint8)
    explicit = atom.GetNumExplicitHs()
    implicit = atom.GetNumImplicitHs()
    H_num = attached_H + explicit + implicit

    try:
        H_hot = preprocessing_utils.one_hot_encoding(H_num, 6)
    except:
        print(H_num)
        print(attached_H, explicit, implicit)
        raise Exception("Sorry, no numbers below zero")

    torch_result = torch.cat((torch_result, H_hot), 0)

    return torch_result


def get_bond_features(bond, use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    torch_result = torch.tensor([])

    BOND_TYPE = [1.0, 1.5, 2.0, 3.0]
    bond_dict = {elem: index for index, elem in enumerate(BOND_TYPE)}
    bond_type_hot = preprocessing_utils.one_hot_encoding(bond_dict.get(bond.GetBondTypeAsDouble(), len(bond_dict)),
                                                         len(BOND_TYPE))
    torch_result = torch.cat((torch_result, bond_type_hot), 0)

    bond_is_conj_hot = preprocessing_utils.one_hot_encoding(bond.GetIsConjugated(), 1)
    #     bond_is_conj_enc = [int(bond.GetIsConjugated())]
    torch_result = torch.cat((torch_result, bond_is_conj_hot), 0)

    bond_is_in_ring_hot = preprocessing_utils.one_hot_encoding(bond.IsInRing(), 1)
    #     bond_is_in_ring_enc = [int(bond.IsInRing())]
    torch_result = torch.cat((torch_result, bond_is_in_ring_hot), 0)

    if use_stereochemistry == True:
        STEREO_TYPE = ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        stereo_dict = {elem: index for index, elem in enumerate(STEREO_TYPE)}
        stereo_type_hot = preprocessing_utils.one_hot_encoding(stereo_dict.get(str(bond.GetStereo()), len(stereo_dict)),
                                                               len(STEREO_TYPE))
        torch_result = torch.cat((torch_result, stereo_type_hot), 0)
    return torch_result


def create_graph_data(nist_data, intensity_power, output_size, operation, input_source="df"):
    """
    Create pytorch geometric graph data from parquet and others
    Inputs:

    Pandas dataframe with columns:
    rdkit mol
    spectrum: tuple[tuple[2]]
    smiles

    Outputs:

    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular
    graphs that can readily be used for machine learning
    """

    data_list = []
    # if input is pandas dataframe
    if input_source == "df":
        for _, nist_obj in nist_data.iterrows():

            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(nist_obj['smiles'])

            if mol == None:
                continue

            molecules_features = mol_to_graph(mol)
            X = molecules_features["atoms_features"]
            E = molecules_features["edge_features"]
            EF = molecules_features["edge_index"]
            MW = molecules_features["molecular_weight"]

            # construct label tensor
            y_tensor = preprocessing_utils.spectrum_preparation_double(nist_obj["spect"], intensity_power, output_size,
                                                                       operation)

            # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x=X, edge_index=E, edge_attr=EF, molecular_weight=MW, y=y_tensor))

    # if we have to create dataset from homo lumo prediction
    elif input_source == "homo-lumo":

        for obj in nist_data:

            # convert SMILES to RDKit mol object
            mol = Chem.MolFromSmiles(obj[0])

            if mol is None:
                continue

            # get feature dimensions
            molecules_features = mol_to_graph(mol)
            X = molecules_features["atoms_features"]
            E = molecules_features["edge_features"]
            EF = molecules_features["edge_index"]
            MW = molecules_features["molecular_weight"]
            smiles = molecules_features["smiles"]

            # construct label tensor
            y_tensor = torch.tensor([obj[1]])

            # construct Pytorch Geometric data object and append to data list
            data_list.append(Data(x=X, edge_index=E, edge_attr=EF, molecular_weight=MW, y=y_tensor, smiles=smiles))

    return data_list


def mol_to_graph(mol):
    """
    Creates graph representation from  mol object
    """
    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()

    # the purpose is to find out one hot emb dimension
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    # construct node feature matrix X of shape (n_nodes, n_node_features)
    X = np.zeros((n_nodes, n_node_features))

    for atom in mol.GetAtoms():
        X[atom.GetIdx(), :] = get_atom_features(atom)

    X = torch.tensor(X, dtype=torch.float64)

    # construct edge index array E of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))

    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    E = torch.stack([torch_rows, torch_cols], dim=0)

    # construct edge feature array EF of shape (n_edges, n_edge_features)
    EF = np.zeros((n_edges, n_edge_features))

    for (k, (i, j)) in enumerate(zip(rows, cols)):
        EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

    EF = torch.tensor(EF, dtype=torch.float)

    # weight of molecular
    MW = Descriptors.ExactMolWt(mol)
    MW = torch.tensor(int(round(float(MW))))

    smiles = Chem.MolToSmiles(mol)

    return {"atoms_features": X,
            "edge_features": E,
            "edge_index": EF,
            "molecular_weight": MW,
            "smiles": smiles}
