import os
import warnings
import torch
import logging
from pymatgen.core import Structure
from chgnet.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics
from datetime import datetime

warnings.filterwarnings("ignore", module="ase")

# Setup logging
def setup_logger(log_dir="logs/finetuning_md", log_name="MD_Simulation"):
    """Set up logging for the simulation."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, log_name, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "simulation.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    
    return log_path, log_file


def run_md_simulation(
    structure_file,
    model_path,
    ensemble="npt",
    temperature=300,
    timestep=2,
    simulation_time=10 * 1000,  # 10 ps
    log_dir="logs/finetuning_md",
):
    """
    Run molecular dynamics (MD) simulation using a finetuned CHGNet model.

    Args:
        structure_file (str): Path to the CIF file of the structure.
        model_path (str): Path to the finetuned CHGNet model.
        ensemble (str): The type of ensemble ("npt", "nve", "nvt").
        temperature (float): Temperature in Kelvin.
        timestep (int): Timestep in femtoseconds.
        simulation_time (int): Total simulation time in femtoseconds.
        log_dir (str): Directory to save logs and outputs.

    Returns:
        str: Path to the directory containing the simulation logs and outputs.
    """
    # Set up logger
    log_path, log_file = setup_logger(log_dir=log_dir, log_name="MD_Simulation")
    logging.info("Starting molecular dynamics simulation.")

    try:
        # Load finetuned CHGNet model
        logging.info(f"Loading CHGNet model from: {model_path}")
        chgnet = CHGNet()
        chgnet.load_state_dict(torch.load(model_path, map_location="cpu"))
        chgnet.eval()

        # Load structure
        logging.info(f"Loading structure from: {structure_file}")
        struct = Structure.from_file(structure_file)

        # Prepare file paths for outputs
        trajectory_file = os.path.join(log_path, f"md_out_{ensemble}_T_{temperature}.traj")
        log_file_md = os.path.join(log_path, f"md_out_{ensemble}_T_{temperature}.log")

        # Setup MD simulation
        logging.info("Setting up MD simulation.")
        md = MolecularDynamics(
            atoms=struct,
            model=chgnet,
            ensemble=ensemble,
            temperature=temperature,
            timestep=timestep,
            trajectory=trajectory_file,
            logfile=log_file_md,
            loginterval=100,
        )

        # Run simulation
        logging.info(f"Starting MD simulation: {simulation_time} fs at {temperature} K.")
        md.run(simulation_time)
        logging.info(f"Simulation completed. Trajectory saved to {trajectory_file}, log saved to {log_file_md}.")
    
    except Exception as e:
        logging.error(f"Error occurred during simulation: {e}")
        raise

    return log_path


if __name__ == "__main__":
    STRUCTURE_FILE = "BaZrO3.cif"
    MODEL_PATH = "../pretraining_finetuning_chgnet/logs/CHGNet_finetuning/checkpoints/chgnet_finetuned_model.pth"
    ENSEMBLE = "npt"
    TEMPERATURE = 300  # in K
    TIMESTEP = 2  # in fs
    SIMULATION_TIME = 10 * 100  # 1 ps
    LOG_DIR = "logs/finetuning_md"

    # Run MD simulation
    output_dir = run_md_simulation(
        structure_file=STRUCTURE_FILE,
        model_path=MODEL_PATH,
        ensemble=ENSEMBLE,
        temperature=TEMPERATURE,
        timestep=TIMESTEP,
        simulation_time=SIMULATION_TIME,
        log_dir=LOG_DIR,
    )

    print(f"MD simulation logs and outputs saved in: {output_dir}")
