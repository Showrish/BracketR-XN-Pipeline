"""SMILES normalization for predictions and answers."""
from rdkit import Chem


def normalize(smiles, largest_fragment=True):
    """Return canonical SMILES or None. Remove dummy labels, not attachment atoms.

    Fragment cleanup is configurable because salts and mixtures can be meaningful.
    Equal-size fragments are resolved lexically for reproducibility.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    if largest_fragment:
        fragments = Chem.GetMolFrags(mol, asMols=True)
        mol = sorted(fragments, key=lambda m: (-m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))[0]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)
