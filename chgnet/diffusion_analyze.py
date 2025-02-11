import warnings
import numpy as np
import logging
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt

# Physical constants
KB = 8.617333262145e-5  # eV/K, Boltzmann constant
KB_J = 1.380649e-23     # J/K
EV_TO_KJ_PER_MOL = 96.485  # Conversion factor from eV to kJ/mol

warnings.filterwarnings("ignore")


def setup_logging(log_dir: str = "logs") -> None:
    """
    Setup logging system

    Args:
        log_dir (str): log directory

    Returns:
        None
    """
    Path(log_dir).mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/diffusion_analysis.log"),
            logging.StreamHandler()
        ]
    )


def parse_md_logs(md_dir: Path) -> tuple:
    """
    Parse MD simulation logs to extract diffusion coefficients for different temperatures

    Args:
        md_dir (Path): Directory containing MD simulation results

    Returns:
        tuple: Lists of temperatures and corresponding diffusion coefficients
    """
    logger = logging.getLogger(__name__)
    temperatures = []
    diffusion_coeffs = []

    # First try to find the main log file
    log_files = list(md_dir.glob("logs/*.log"))
    if not log_files:
        # If no log in logs directory, try temperature directories
        logger.info("No log file found in logs directory, checking temperature directories...")
        temp_dirs = [d for d in md_dir.glob("T_*K") if d.is_dir()]
        if not temp_dirs:
            raise ValueError(f"No temperature directories or log files found in {md_dir}")

        # Process each temperature directory
        for temp_dir in temp_dirs:
            temp_match = re.search(r'T_(\d+)K', temp_dir.name)
            if not temp_match:
                continue

            temp = float(temp_match.group(1))
            log_files = list(temp_dir.glob("md_out_*_T_*.log"))

            if log_files:
                log_file = log_files[0]
                D_total = parse_single_log(log_file, temp)
                if D_total is not None:
                    temperatures.append(temp)
                    diffusion_coeffs.append(D_total)
    else:
        # Process main log file which contains all temperatures
        main_log = log_files[0]
        logger.info(f"Found main log file: {main_log}")

        with open(main_log, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "Total Results for" in line:
                temp_match = re.search(r'Total Results for (\d+)K', line)
                if temp_match:
                    temp = float(temp_match.group(1))
                    # Look for diffusion coefficient in next few lines
                    for next_line in lines[i:i+5]:
                        if "coefficient:" in next_line:
                            match = re.search(r'coefficient:\s+(\d+\.\d+e[+-]\d+)\s+\[cm²/s\]', next_line)
                            if match:
                                D_total = float(match.group(1))
                                temperatures.append(temp)
                                diffusion_coeffs.append(D_total)
                                logger.info(f"Found diffusion coefficient for {temp}K: {D_total:.2e} cm²/s")
                                break

    if not temperatures:
        raise ValueError("No diffusion coefficients found in the logs")

    # Sort by temperature
    temp_D_pairs = sorted(zip(temperatures, diffusion_coeffs))
    temperatures, diffusion_coeffs = zip(*temp_D_pairs)

    # Log summary
    logger.info("\nExtracted diffusion coefficients:")
    logger.info("Temperature (K) | Diffusion Coefficient (cm²/s)")
    logger.info("-" * 50)
    for T, D in zip(temperatures, diffusion_coeffs):
        logger.info(f"{T:13.1f} | {D:.6e}")

    return list(temperatures), list(diffusion_coeffs)


def parse_single_log(log_file: Path, temperature: float) -> float:
    """
    Parse a single MD log file to extract the diffusion coefficient

    Args:
        log_file (Path): Path to log file
        temperature (float): Temperature in K

    Returns:
        float: Diffusion coefficient
    """
    logger = logging.getLogger(__name__)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Look for the last total diffusion coefficient
    for line in reversed(lines):
        if "Total Results for" in line and str(int(temperature)) in line:
            # Find coefficient in next few lines
            idx = lines.index(line)
            for next_line in lines[idx:idx+5]:
                if "coefficient:" in next_line:
                    match = re.search(r'coefficient:\s+(\d+\.\d+e[+-]\d+)\s+\[cm²/s\]', next_line)
                    if match:
                        D_total = float(match.group(1))
                        logger.info(f"Found diffusion coefficient for {temperature}K: {D_total:.2e} cm²/s")
                        return D_total

    return None


def analyze_arrhenius(temperatures: list, diffusion_coeffs: list,
                      output_dir: Path, logger: logging.Logger) -> None:
    """
    Analyze diffusion coefficients using Arrhenius equation

    Args:
        temperatures (list): List of temperatures in K
        diffusion_coeffs (list): List of diffusion coefficients in cm²/s
        output_dir (Path): Output directory for plots and results
        logger (logging.Logger): Logger object
    """
    # --- 1. Arrhenius 拟合 ---
    temps = np.array(temperatures)
    D_values = np.array(diffusion_coeffs)
    inv_T = 1.0 / temps
    ln_D = np.log(D_values)

    popt, pcov = np.polyfit(inv_T, ln_D, 1, cov=True)
    slope, intercept = popt
    slope_err, intercept_err = np.sqrt(np.diag(pcov))

    # 计算激活能和前指数因子
    Ea = -slope * KB  # eV
    A = np.exp(intercept)  # cm²/s
    R_squared = np.corrcoef(inv_T, ln_D)[0, 1]**2

    # --- 2. 创建双轴图 ---
    fontsize = 16
    fig, ax_bottom = plt.subplots(figsize=(10, 8))
    ax_top = ax_bottom.twiny()  # 共享 y 轴，独立 x 轴

    # 下方 x 轴数据: x_bottom = 1000 / T
    x_bottom = 1000.0 * inv_T
    y_data = ln_D

    # --- 2.1 作散点 ---
    ax_bottom.scatter(x_bottom, y_data, color='blue', label='MD results')

    # 拟合直线: 为了和散点范围一致，这里只取你有的温度范围
    x_fit = np.linspace(x_bottom.min(), x_bottom.max(), 50)
    # 反推回去 -> invT_fit = x_fit / 1000
    invT_fit = x_fit / 1000.0
    ln_D_fit = slope * invT_fit + intercept
    ax_bottom.plot(x_fit, ln_D_fit, 'r--', label='Arrhenius fit')

    # --- 2.2 设置下方坐标轴 (1000/T) ---
    ax_bottom.set_xlabel('1000/T (K⁻¹)', fontsize=fontsize-2)
    ax_bottom.set_ylabel('ln(D) [D in cm²/s]', fontsize=fontsize-2)
    ax_bottom.set_title('Arrhenius Plot for Proton Diffusion (CHGNet Fine-tuning)', fontsize=fontsize)
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(fontsize=fontsize-4)

    # 根据你的温度，手动指定刻度位置
    x_ticks_bottom = [1000.0 / T for T in temps]
    # 如果你想让 700K 在右、1100K 在左，就 reverse=True
    x_ticks_bottom = sorted(x_ticks_bottom, reverse=True)
    ax_bottom.set_xticks(x_ticks_bottom)
    ax_bottom.set_xticklabels([f"{v:.2f}" for v in x_ticks_bottom])

    # 调整下方显示范围
    margin = 0.1*(max(x_bottom) - min(x_bottom))
    ax_bottom.set_xlim(min(x_bottom)-margin, max(x_bottom)+margin)

    # --- 2.3 设置上方坐标轴 (T in K) ---
    # 与下方共享相同的 x 范围
    ax_top.set_xlim(ax_bottom.get_xlim())

    # 定义一个函数：将 (1000/T) -> T
    def bottom_to_top(xvals):
        return [1000.0 / x if x != 0 else 0 for x in xvals]

    # 顶轴刻度位置与下轴相同，但标签要变成温度
    ax_top.set_xticks(x_ticks_bottom)
    ax_top.set_xticklabels([f"{int(bottom_to_top([xt])[0])} K" for xt in x_ticks_bottom])

    # 如果你不想显示顶轴的 "Temperature (K)"，可以注释掉下面这行
    # ax_top.set_xlabel('Temperature (K)', fontsize=fontsize-2)

    # Add text box with results
    # textstr = '\n'.join((
    #     f'Ea = {Ea:.3f} ± {Ea_err:.3f} eV',
    #     f'Ea = {Ea_kJ_mol:.1f} ± {Ea_err_kJ_mol:.1f} kJ/mol',
    #     f'A = {A:.2e} ± {A_err:.2e} cm²/s',
    #     f'R² = {R_squared:.4f}'
    # ))
    # --- 2.4 在图上添加文本框 ---
    textstr = '\n'.join((
        f'Ea = {Ea:.3f} eV',
        f'Do = {A:.2e} cm²/s',
        f'R² = {R_squared:.4f}'
    ))
    ax_bottom.text(
        0.05, 0.05,
        textstr,
        transform=ax_bottom.transAxes,
        fontsize=fontsize-4,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # --- 3. 保存并关闭 ---
    plt.tight_layout()
    plt.savefig(output_dir / 'arrhenius_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- 4. 打印或保存日志 ---
    logger.info("\nArrhenius Analysis Results:")
    logger.info(f"Activation Energy: {Ea:.3f} eV")
    logger.info(f"Pre-exponential factor: {A:.2e} cm²/s")
    logger.info(f"R-squared: {R_squared:.4f}")

    with open(output_dir / 'diffusion_analysis_results.txt', 'w') as f:
        f.write("Diffusion Analysis Results\n")
        f.write("==========================\n\n")
        f.write(f"Activation Energy: {Ea:.3f} eV\n")
        f.write(f"Pre-exponential factor: {A:.2e} cm²/s\n")
        f.write(f"R-squared: {R_squared:.4f}\n\n")
        f.write("Raw Data:\n")
        f.write("Temperature (K) | Diffusion Coefficient (cm²/s)\n")
        f.write("-" * 50 + "\n")
        for T, D in zip(temperatures, diffusion_coeffs):
            f.write(f"{T:13.1f} | {D:.6e}\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze MD simulation results')
    parser.add_argument('--md-dir', type=str, default='./pretraining_md_results',
                        help='Directory containing MD simulation results')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                        help='Output directory')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    md_dir = Path(args.md_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(str(output_dir / "logs"))
    logger = logging.getLogger(__name__)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        # Parse MD results
        logger.info(f"Parsing MD results from: {md_dir}")
        temperatures, diffusion_coeffs = parse_md_logs(md_dir)

        # Analyze results
        analyze_arrhenius(temperatures, diffusion_coeffs, output_dir, logger)
        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
