from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
import torch
from torch.utils.data import Dataset
from preprocessing import preprocessing_utils


# define function that transforms SMILES strings into ECFPs
def ecfp_from_smiles(smiles, R=2, L=4096, use_features=False, use_chirality=False):
    """
    Inputs:
    - smiles: SMILES string of input compound
    - R: maximum radius of circular substructures
    - L: fingerprint-length
    - use_features: if false then use standard DAYLIGHT atom features, if true then use pharmacophoric atom features
    - use_chirality: if true then append tetrahedral chirality flags to atom features

    Outputs:
    - np.array(feature_list): ECFP with length L and maximum radius R
    """

    molecule = AllChem.MolFromSmiles(smiles)
    feature_list = AllChem.GetMorganFingerprintAsBitVect(molecule,
                                                         radius=R,
                                                         nBits=L,
                                                         useFeatures=use_features,
                                                         useChirality=use_chirality)
    return torch.tensor(feature_list)

def create_ecfp_dataset_parquet(nist_data, model_config):
    """
    Inputs:

    Pandas dataframe with columns:
    rdkit mol
    spectrum: tuple[tuple[2]]
    smiles

    Outputs:

    list of tuples

    """

    data_list = []

    for _, nist_obj in nist_data.iterrows():

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(nist_obj['smiles'])

        if mol is None:
            continue

        input_ecp = ecfp_from_smiles(nist_obj['smiles'], R=model_config["preprocessing"]["radius"],
                                     L=model_config["preprocessing"]["fingerprint_length"],
                                     use_features=model_config["preprocessing"]["use_features"],
                                     use_chirality=model_config["preprocessing"]["use_chirality"])

        # construct label tensor
        y_tensor = preprocessing_utils.spectrum_preparation_double(nist_obj["spect"],
                                            model_config["preprocessing"]["intensity_power"],
                                            model_config["preprocessing"]["output_size"],
                                            model_config["preprocessing"]["operation"])

        # Molecular weight
        MW = Descriptors.ExactMolWt(mol)
        MW = torch.tensor(int(round(float(MW))))

        # construct Pytorch data object and append to data list
        data_list.append((input_ecp, MW, y_tensor))

    return data_list


class FingerprintDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor, molecular_weight, output_tensor = self.data[index]
        return {
            'input_tensor': input_tensor.float(),  # fingerprint
            'molecular_weight': molecular_weight,
            'output_tensor': output_tensor  # spectrum
        }