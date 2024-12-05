from ase.io.trajectory import Trajectory
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def calculate_msd(traj):
    """
    Calculate the mean square displacement (MSD) from a trajectory file.
    
    Parameters:
    traj (Trajectory): ASE trajectory object
    
    Returns:
    np.ndarray: Array of MSD values
    """
    initial_positions = traj[0].get_positions()
    msd = []
    
    for atoms in traj:
        current_positions = atoms.get_positions()
        displacement = current_positions - initial_positions
        squared_displacement = np.sum(displacement**2, axis=1)
        msd.append(np.mean(squared_displacement))
    
    return np.array(msd)

def main():
    # Load the trajectory
    traj = Trajectory('md_out_npt_T_300.traj', 'r')
    
    # Calculate MSD
    msd = calculate_msd(traj)
    timestep_fs = 2  # femtoseconds
    time_steps = np.arange(len(msd)) * timestep_fs / 1000  # convert to picoseconds
    
    # Perform linear fit
    slope, intercept, _, _, _ = linregress(time_steps, msd)
    fitted_line = slope * time_steps + intercept
    
    # Plot MSD with fitted line
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, msd, label='MSD', linestyle='-', color='tab:blue')
    plt.plot(time_steps, fitted_line, label='fitted line', linestyle='-', color='tab:orange')
    plt.xlabel('Time (ps)')
    plt.ylabel('Mean Square Displacement (Å²)')
    plt.title('Mean Square Displacement with Linear Fit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('msd.png')
    plt.show()

if __name__ == "__main__":
    main()
