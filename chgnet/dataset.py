import numpy as np
from pymatgen.core import Structure
from chgnet.utils import read_json
from chgnet.data.dataset import StructureData, get_train_val_test_loader


def load_dataset(json_path: str, batch_size: int = 2, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Load dataset from a JSON file and split it into train, validation, and test loaders.

    Args:
        json_path (str): Path to the JSON dataset file.
        batch_size (int): Batch size for data loaders. Default = 2.
        train_ratio (float): Ratio of training data. Default = 0.8.
        val_ratio (float): Ratio of validation data. Default = 0.1.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Read JSON dataset
    dataset_dict = read_json(json_path)

    # Extract structure information
    structures = [
        Structure(
            lattice=struct_dict["lattice"],
            species=struct_dict["species"],
            coords=struct_dict["coords"],
            coords_are_cartesian=False  # Assume coordinates are fractional
        )
        for struct_dict in dataset_dict["structures"]
    ]

    # Extract energy, force, and stress labels
    energies = np.array(dataset_dict["labels"]["energies"])  # Energies
    forces = np.array(dataset_dict["labels"]["forces"])      # Forces
    stresses = np.array(dataset_dict["labels"]["stresses"])  # Stresses (optional)

    # Create CHGNet-compatible StructureData
    dataset = StructureData(
        structures=structures,
        energies=energies,
        forces=forces,
        stresses=None,  # Set to None if stresses are not available
        magmoms=None  # Optional, magnetic moments
    )

    # Split dataset into train, validation, and test sets
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=batch_size, train_ratio=train_ratio, val_ratio=val_ratio
    )
    return train_loader, val_loader, test_loader
