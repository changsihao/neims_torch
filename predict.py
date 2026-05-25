import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdMD, Descriptors
from model import MInterface


_MODEL_CACHE = {}


def load_model(
    ckpt_path="logs/neims_run1/version_5/checkpoints/epoch120-valwMSE51.81711.ckpt",
    device="cuda",
):
    key = (ckpt_path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = MInterface.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    _MODEL_CACHE[key] = model
    return model

def smiles_to_ecfp_counts(smiles: str, n_bits=4096, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = rdMD.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx % n_bits] = float(count)
    return arr

def smiles_to_mass(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return int(round(Descriptors.ExactMolWt(mol))) if mol else 0


def predict_from_smiles(smiles_list, device="cuda", min_intensity=0.5, normalize_to=100.0, decimals=2):
    
    single_input = False
    if isinstance(smiles_list, str):
        smiles_list = [smiles_list]
        single_input = True
        
    fp_list = []
    mass_list = []

    for sid, smi in enumerate(smiles_list):
        fp = smiles_to_ecfp_counts(smi)
        mass = smiles_to_mass(smi)
        fp_list.append(fp)
        mass_list.append(mass)
    mz_list, intensity_list = predict_from_fp(fp_list, mass_list, device, min_intensity, normalize_to, decimals)

    if single_input:
        return mz_list[0], intensity_list[0]
    return mz_list, intensity_list


def predict_from_fp(fp_list, mass_list, device="cuda", min_intensity=0.5, normalize_to=100.0, decimals=2):

    model = load_model(device=device)
    model.to(device)
    model.eval()

    # to numpy
    fp_arr = np.asarray(fp_list, dtype=np.float32)
    mass_arr = np.asarray(mass_list, dtype=np.int64)

    # normalize shapes
    if fp_arr.ndim == 1:
        fp_arr = fp_arr.reshape(1, -1)
    if mass_arr.ndim == 0:
        mass_arr = mass_arr.reshape(1)

    fp_tensor = torch.from_numpy(fp_arr).to(device)
    mass_tensor = torch.from_numpy(mass_arr).to(device)

    # predict
    with torch.no_grad():
        pred = model(fp_tensor, mass_tensor).cpu().numpy()

    N, L = pred.shape
    mz_base = np.arange(L, dtype=np.int32)

    mz_list = []
    intensity_list = []
    for i in range(N):
        p = pred[i]  # shape (L,)
        keep = p > min_intensity
        keep[0]=False

        if not np.any(keep):
            mz_i = np.array([], dtype=np.int32)
            intens_i = np.array([], dtype=np.float32)
        else:
            mz_i = mz_base[keep]
            intens_i = p[keep].astype(np.float32)
            max_val = intens_i.max()
            intens_i = intens_i / max_val * normalize_to
            
        intens_i = np.round(intens_i, decimals)
        
        mz_list.append(mz_i)
        intensity_list.append(intens_i)

    return mz_list, intensity_list


if __name__ == "__main__":
    smiles_list = ["CCO","CCN","CCCl"]
    pred_mz,pred_intensity = predict_from_smiles(smiles_list,"cuda:7")
    print(pred_mz)
    print(pred_intensity)
    