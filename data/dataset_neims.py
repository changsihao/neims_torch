import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matchms.importing import load_from_msp
from matchms.exporting import save_as_msp
from matchms import Spectrum
from dataclasses import dataclass
from matchms.filtering import normalize_intensities
import pathlib

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdMD
    from rdkit.Chem.Descriptors import ExactMolWt
    RDKit_OK = True
except Exception:
    RDKit_OK = False
    
    
@dataclass
class MSPEntry:
    id: str
    smiles: Optional[str]
    mass: Optional[float]
    peaks: List[Tuple[float, float]]
    meta: Dict[str, str]

def _get_md(md: Dict, *keys, default=None):
    for k in keys:
        if k in md and md[k] not in (None, ""):
            return md[k]
        lk, uk = k.lower(), k.upper()
        if lk in md and md[lk] not in (None, ""): return md[lk]
        if uk in md and md[uk] not in (None, ""): return md[uk]
    return default


def smiles_to_ecfp_counts(smiles: str, n_bits: int = 4096, radius: int = 2) -> np.ndarray:
    if not RDKit_OK:
        raise RuntimeError("RDKit required for SMILES→ECFP.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = rdMD.GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for idx, count in fp.GetNonzeroElements().items():
        arr[idx] = float(count)
    return arr

def mass_from_entry(e: MSPEntry) -> int:
    if e.mass is not None and not math.isnan(e.mass):
        return int(round(float(e.mass)))
    if e.smiles and RDKit_OK:
        mol = Chem.MolFromSmiles(e.smiles)
        if mol is not None:
            return int(round(ExactMolWt(mol)))
    if e.peaks:
        return int(round(max(m for m, _ in e.peaks)))
    return 0

def peaks_to_vector(peaks: List[Tuple[float, float]], max_mz: int, normalize_basepeak: bool = True) -> np.ndarray:
    vec = np.zeros((max_mz + 1,), dtype=np.float32)
    if not peaks:
        return vec
    if normalize_basepeak:
        base = max(i for _, i in peaks) or 1.0
        peaks = [(m, i * 100.0 / base) for m, i in peaks]
    for m, i in peaks:
        b = int(round(m))
        if 0 <= b <= max_mz:
            vec[b] += float(i)
    return vec

def load_msp_entries(path: str) -> List[MSPEntry]:
    entries: List[MSPEntry] = []
    for spec in load_from_msp(path):
        if spec is None:
            continue
        md = dict(spec.metadata or {})
        sid = _get_md(md, "name", "compound_name", "title", default=None) or f"SPECTRUM_{len(entries)}"
        smiles = _get_md(md, "smiles", default=None)
        mass_val = _get_md(md,"molecular_weight","exact_mass", "exactmass", "precursor_mz", "mw", "mass", default=None)
        try:
            mass = float(mass_val) if mass_val is not None else None
        except Exception:
            mass = None

        mz = np.asarray(spec.peaks.mz if spec.peaks is not None else [], dtype=np.float32)
        it = np.asarray(spec.peaks.intensities if spec.peaks is not None else [], dtype=np.float32)
        keep = (it > 0) & (mz >= 0)
        peaks = [(float(m), float(i)) for m, i in zip(mz[keep], it[keep])]

        meta = {k: v for k, v in md.items()}
        entries.append(MSPEntry(sid, smiles, mass, peaks, meta))
    return entries


class MSPDataset(Dataset):
    def __init__(self, data_file: str,cache_file:str, n_bits=4096, radius=2, max_mz=2000, drop_no_smiles=True):
        self.data_file = pathlib.Path(__file__).parent / data_file
        print(self.data_file)
        self.cache_file = pathlib.Path(__file__).parent / cache_file
        self.items = []
        self.max_mz = max_mz
        
        if os.path.exists(self.cache_file):
            # 直接加载缓存
            print("Load cache from",cache_file)
            data = np.load(self.cache_file, allow_pickle=True)
            self.items = list(data["items"])
        else:
            # 正常构造
            entries = load_msp_entries(self.data_file)
            for e in entries:
                if (not e.smiles) and drop_no_smiles:
                    continue
                try:
                    fp = smiles_to_ecfp_counts(e.smiles, n_bits, radius) if e.smiles else None
                except Exception:
                    continue
                mass = mass_from_entry(e)
                spec = peaks_to_vector(e.peaks, max_mz, normalize_basepeak=True)
                if fp is None:
                    continue
                self.items.append((e.id, fp.astype("float32"), spec.astype("float32"), int(mass)))
            # 存缓存
            np.savez_compressed(self.cache_file, items=np.array(self.items, dtype=object))
            
        
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, fp, spec, mass = self.items[idx]
        return (
            torch.from_numpy(fp),                  # [n_bits]
            torch.from_numpy(spec),                # [L]
            torch.tensor(mass, dtype=torch.long),  # scalar
            sid
        )