import warnings
import numpy as np
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from ase import Atom, Atoms
from ase.optimize import BFGS
import statsmodels.api as sm
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar
from chgnet.model.dynamics import MolecularDynamics

warnings.filterwarnings("ignore")


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging system
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/pretraining_md_simulation.log"),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='CHGNet pretraining MD Simulation')
    parser.add_argument('--structure-file', type=str, default="./structures/BaZrO3.cif",
                        help='Path to the structure CIF file')
    parser.add_argument('--ensemble', type=str, default="nvt",
                        choices=['npt', 'nve', 'nvt'],
                        help='Type of ensemble')
    parser.add_argument('--temperatures', type=float, nargs='+', default=[1000],
                        help='Temperatures for MD simulation (K), e.g. 800 900 1000')
    parser.add_argument('--timestep', type=float, default=1.0,
                        help='Timestep for MD simulation (fs)')
    parser.add_argument('--n-steps', type=int, default=40000,
                        help='Number of MD steps')
    parser.add_argument('--n-protons', type=int, default=1,
                        help='Number of protons to add')
    parser.add_argument('--output-dir', type=str, default='./pretraining_md_results',
                        help='Directory to save outputs')
    parser.add_argument('--window-size', type=int, default=None,
                        help='Window size for MSD calculation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')

    return parser.parse_args()


def add_protons(atoms: Atoms, n_protons: int) -> Atoms:
    """
    Add protons to the structure based on theoretical understanding.
    """
    logger = logging.getLogger(__name__)
    OH_BOND_LENGTH = 0.98  # Å
    MAX_NEIGHBOR_DIST = 3.0  # Å

    o_indices = [i for i, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == 'O']

    if len(o_indices) < n_protons:
        logger.warning(f"Number of protons ({n_protons}) exceeds number of O atoms ({len(o_indices)})")
        n_protons = len(o_indices)

    used_oxygens = []
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    for i in range(n_protons):
        available_oxygens = [idx for idx in o_indices if idx not in used_oxygens]
        if not available_oxygens:
            logger.warning("No more available oxygen atoms for proton incorporation")
            break

        o_idx = available_oxygens[0]
        used_oxygens.append(o_idx)
        o_pos = atoms.positions[o_idx]

        # Find neighboring oxygen atoms
        neighbors = []
        for other_idx in o_indices:
            if other_idx != o_idx:
                dist = atoms.get_distance(o_idx, other_idx, mic=True)
                if dist < MAX_NEIGHBOR_DIST:
                    vec = atoms.get_distance(o_idx, other_idx, vector=True, mic=True)
                    neighbors.append({'idx': other_idx, 'dist': dist, 'vec': vec})

        # Calculate proton position direction
        direction = np.zeros(3)
        if neighbors:
            for n in sorted(neighbors, key=lambda x: x['dist'])[:3]:
                weight = 1.0 / max(n['dist'], 0.1)
                direction -= n['vec'] * weight

            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([0, 0, 1])
        else:
            direction = np.array([0, 0, 1])

        h_pos = o_pos + direction * OH_BOND_LENGTH

        # Check position validity
        is_valid = True
        min_allowed_dist = 0.8  # Å
        for pos in atoms.positions:
            dist = np.linalg.norm(h_pos - pos)
            if dist < min_allowed_dist:
                is_valid = False
                break

        # Apply periodic boundary conditions if needed
        if any(pbc):
            scaled_pos = np.linalg.solve(cell.T, h_pos.T).T
            scaled_pos = scaled_pos % 1.0
            h_pos = cell.T @ scaled_pos

        if not is_valid:
            logger.warning(f"Invalid proton position near O atom {o_idx}, trying different direction")
            continue

        # Add proton
        atoms.append(Atom('H', position=h_pos))
        oh_dist = atoms.get_distance(-1, o_idx, mic=True)

        logger.info(f"Added proton {i+1}/{n_protons}:")
        logger.info(f"  Near O atom: {o_idx}")
        logger.info(f"  Position: {h_pos}")
        logger.info(f"  OH distance: {oh_dist:.3f} Å")

    logger.info(f"Successfully added {n_protons} protons")
    logger.info(f"Final composition: {atoms.get_chemical_formula()}")

    return atoms


def calculate_msd_sliding_window(trajectory: Trajectory, atom_indices: list,
                                 timestep: float = 1.0, window_size: int = None):
    """
    Calculate MSD using sliding window method for both directional and total MSD.
    """
    positions_all = np.array([atoms.get_positions() for atoms in trajectory])
    positions = positions_all[:, atom_indices]

    n_frames = len(positions)
    if window_size is None:
        window_size = n_frames // 4

    shift_t = window_size // 2  # Shift window by half its size

    # Initialize arrays for accumulating MSD values
    msd_x = np.zeros(window_size)
    msd_y = np.zeros(window_size)
    msd_z = np.zeros(window_size)
    msd_total = np.zeros(window_size)
    counts = np.zeros(window_size)

    # Calculate MSD using sliding windows
    n_windows = n_frames - window_size + 1
    for start in range(0, n_frames - window_size, shift_t):
        window = slice(start, start + window_size)
        ref_pos = positions[start]

        # Calculate displacements
        disp = positions[window] - ref_pos

        # Calculate MSD components
        msd_x += np.mean(disp[..., 0]**2, axis=1)
        msd_y += np.mean(disp[..., 1]**2, axis=1)
        msd_z += np.mean(disp[..., 2]**2, axis=1)
        msd_total += np.mean(np.sum(disp**2, axis=2), axis=1)
        counts += 1

    # Average MSDs
    msd_x /= counts
    msd_y /= counts
    msd_z /= counts
    msd_total /= counts

    # Calculate time array in picoseconds
    time = np.arange(window_size) * timestep / 1000

    # Calculate diffusion coefficients using np.polyfit
    D_x = np.polyfit(time, msd_x, 1)[0] / 2  # For 1D
    D_y = np.polyfit(time, msd_y, 1)[0] / 2
    D_z = np.polyfit(time, msd_z, 1)[0] / 2
    D_total = np.polyfit(time, msd_total, 1)[0] / 6  # For 3D

    # # Calculate diffusion coefficients using statsmodels OLS
    # model_x = sm.OLS(msd_x, sm.add_constant(time))
    # D_x = model_x.fit().params[1] / 2  # For 1D

    # model_y = sm.OLS(msd_y, sm.add_constant(time))
    # D_y = model_y.fit().params[1] / 2

    # model_z = sm.OLS(msd_z, sm.add_constant(time))
    # D_z = model_z.fit().params[1] / 2

    # model_total = sm.OLS(msd_total, sm.add_constant(time))
    # D_total = model_total.fit().params[1] / 6  # For 3D
    return time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total


def analyze_msd(trajectories: list, proton_index: int, temperatures: list,
                timestep: float, output_dir: Path, logger: logging.Logger,
                window_size: int = None) -> None:
    """
    Analyze MSD data and create separate plots for x, y, z directions and total MSD
    """
    # First create subplot figure with all components
    fontsize = 24
    components = ['x', 'y', 'z', 'total']
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    axes = axes.flatten()

    for traj_file, temp in zip(trajectories, temperatures):
        logger.info(f"Analyzing trajectory for {temp}K...")
        trajectory = Trajectory(str(traj_file), 'r')

        # Calculate MSD with directional components
        time, msd_x, msd_y, msd_z, msd_total, D_x, D_y, D_z, D_total = calculate_msd_sliding_window(
            trajectory, [proton_index], timestep=timestep, window_size=window_size
        )

        # Convert diffusion coefficients to cm²/s
        D_x_cm2s = D_x * 1e-16 / 1e-12
        D_y_cm2s = D_y * 1e-16 / 1e-12
        D_z_cm2s = D_z * 1e-16 / 1e-12
        D_total_cm2s = D_total * 1e-16 / 1e-12

        # Plot each component
        msds = [msd_x, msd_y, msd_z, msd_total]
        Ds = [D_x_cm2s, D_y_cm2s, D_z_cm2s, D_total_cm2s]

        for ax, msd, D, component in zip(axes, msds, Ds, components):
            # Plot MSD
            ax.plot(time, msd, label=f"{temp}K (D={D:.2e} cm²/s)")

            # Linear fit using np.polyfit
            slope = np.polyfit(time, msd, 1)[0]

            # Plot fit line
            ax.plot(time, time * slope, '--', alpha=0.5)

            # Customize plot
            ax.set_title(f"{component.upper()}-direction MSD" if component != 'total' else "Total MSD",
                         fontsize=fontsize)
            ax.set_xlabel("Time (ps)", fontsize=fontsize-4)
            ax.set_ylabel("MSD (Å²)", fontsize=fontsize-4)
            ax.tick_params(labelsize=fontsize-6)
            ax.legend(fontsize=fontsize-6)
            ax.grid(True, alpha=0.3)

            # Log results
            logger.info(f"Results for {temp}K ({component}-direction):")
            logger.info(f"  Diffusion coefficient: {D:6.4e} [cm²/s]")
            logger.info(f"  Maximum MSD: {np.max(msd):.2f} Å²")
            logger.info(f"  Average MSD: {np.mean(msd):.2f} Å²")

    plt.tight_layout()
    plt.savefig(output_dir / 'msd_analysis_all_components.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create additional single plot for total MSD with style matching reference
    plt.figure(figsize=(12, 8))

    for traj_file, temp in zip(trajectories, temperatures):
        trajectory = Trajectory(str(traj_file), 'r')
        time, _, _, _, msd_total, _, _, _, D_total = calculate_msd_sliding_window(
            trajectory, [proton_index], timestep=timestep, window_size=window_size
        )

        # Convert D to cm²/s
        D_cm2s = D_total * 1e-16 / 1e-12

        # Plot MSD with diffusion coefficient in label
        plt.plot(time, msd_total, label=f"{temp}K (D={D_cm2s:.2e} cm²/s)")

        # Linear fit using np.polyfit
        slope = np.polyfit(time, msd_total, 1)[0]

        # Plot fit line
        plt.plot(time, time * slope, '--', alpha=0.5)

        D = slope / 6
        logger.info(f"Total Results for {temp}K:")
        logger.info(f"  Diffusion coefficient: {D*1e-16*1e12:6.4e} [cm²/s]")

    plt.title("CHGnet pre-training by VASP", fontsize=fontsize)
    plt.xlabel("Time (ps)", fontsize=fontsize)
    plt.ylabel("MSD (Å²)", fontsize=fontsize)
    plt.tick_params(labelsize=fontsize-4)
    plt.legend(fontsize=fontsize-4)
    plt.tight_layout()

    plt.savefig(output_dir / 'msd_total.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_md_simulation(args) -> None:
    """
    Run molecular dynamics simulation at multiple temperatures
    """
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(str(output_dir / "logs"))
        logger = logging.getLogger(__name__)

        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # Load structure
        logger.info(f"Loading structure from: {args.structure_file}")
        structure = Structure.from_file(args.structure_file)
        logger.info(f"Structure loaded: {structure.composition.reduced_formula}")

        # Convert to ASE atoms and add protons
        atoms_adaptor = AseAtomsAdaptor()
        atoms = atoms_adaptor.get_atoms(structure)
        atoms = add_protons(atoms, args.n_protons)
        proton_index = len(atoms) - 1  # Last added atom is the proton

        # Save structure with protons
        pmg_structure = atoms_adaptor.get_structure(atoms)
        poscar = Poscar(pmg_structure)
        poscar.write_file(output_dir / "POSCAR_with_H")

        trajectory_files = []

        # Run simulation at each temperature
        for temp in args.temperatures:
            logger.info(f"\nStarting simulation at {temp}K...")

            temp_dir = output_dir / f"T_{temp}K"
            temp_dir.mkdir(exist_ok=True)

            traj_file = temp_dir / f"md_out_{args.ensemble}_T_{temp}.traj"
            md_log_file = temp_dir / f"md_out_{args.ensemble}_T_{temp}.log"
            trajectory_files.append(traj_file)

            # Setup MD simulation
            logger.info("Initializing MD simulation...")
            md = MolecularDynamics(
                atoms=pmg_structure,
                ensemble=args.ensemble,
                temperature=temp,
                timestep=args.timestep,
                trajectory=str(traj_file),
                logfile=str(md_log_file),
                loginterval=10,
                use_device='cpu'
            )

            # Run simulation
            logger.info(f"Running MD at {temp}K...")
            for step in range(args.n_steps):
                md.run(1)
                if step % 100 == 0:
                    logger.info(f"Temperature {temp}K - Step {step}/{args.n_steps}")

            logger.info(f"Simulation at {temp}K completed")

        # Analyze trajectories
        analyze_msd(trajectory_files, proton_index, args.temperatures,
                    args.timestep, output_dir, logger, args.window_size)

        logger.info("\nAll MD simulations and analysis completed successfully")

    except Exception as e:
        logger.error(f"MD simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    args = parse_args()
    try:
        run_md_simulation(args)
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
