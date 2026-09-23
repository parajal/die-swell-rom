"""
Generate TRAINING data for the 2-parameter (λ, βv) extrudate swell experiment.

Initial design: 3×3 = 9 factorial grid (steady-state only).
Mobility (α) is held fixed at 0.2.
Output directory: lambda-beta-adaptive/
"""
import os
import shutil
import subprocess
import numpy as np
import time
import math
from io import StringIO


# ============================================================
#                MAIN PIPELINE (ENTRY POINT)
# ============================================================
def main():

    # -------------------------
    # Configuration
    # -------------------------
    data_dir = "lambda-beta-adaptive"

    deltat = 0.1
    numtimesteps = 200
    steady_state = True
    num_time_samples = 10

    # Fixed mobility parameter
    alphapar_fixed = 0.2

    lambda_values = np.linspace(1, 10, 3)        # {1, 8, 15}
    betav_values = np.linspace(0.05, 0.9, 3)      # {0.1, 0.5, 0.9}

    print(f"Training grid: {len(lambda_values)}×{len(betav_values)} "
          f"= {len(lambda_values)*len(betav_values)} samples")
    print(f"  λ  = {lambda_values}")
    print(f"  βv = {betav_values}")
    print(f"  α  = {alphapar_fixed} (fixed)")

    # -------------------------
    # Prepare filesystem
    # -------------------------
    prepare_data_directory(data_dir)

    # -------------------------
    # Time-step selection
    # -------------------------
    total_time = deltat * numtimesteps

    if steady_state:
        time_indices = [numtimesteps]
    else:
        time_indices = generate_log_time_steps(total_time, num_time_samples, deltat)

    # -------------------------
    # Save parameter table
    # -------------------------
    save_parameter_times(
        "parameters.txt",
        lambda_values, betav_values, alphapar_fixed,
        deltat, numtimesteps, time_indices
    )

    # -------------------------
    # Run all simulations
    # -------------------------
    run_simulations(
        executable="./extrudate_swell2d_c",
        lambda_vals=lambda_values,
        betav_vals=betav_values,
        alphapar_fixed=alphapar_fixed,
        deltat=deltat,
        numtimesteps=numtimesteps,
        time_indices=time_indices,
        data_dir=data_dir,
    )

    # -------------------------
    # Final clean-up
    # -------------------------
    if os.path.exists("parameters.txt"):
        shutil.move("parameters.txt", os.path.join(data_dir, "parameters.txt"))
        print(f"Moved parameters.txt → {data_dir}")

    print("\nAll training simulations done!")


# ============================================================
#              FILESYSTEM & CLEANUP HELPERS
# ============================================================
def prepare_data_directory(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    files_to_delete = [
        "parameters.txt", "snapshots.txt", "pressure.txt",
        "mesh_coor2.txt", "c-trace.txt", "b-trace.txt",
    ]
    for file in files_to_delete:
        for location in [".", data_dir]:
            path = os.path.join(location, file)
            if os.path.exists(path):
                os.remove(path)
                print(f"Deleted: {path}")


FORTRAN_TEMP_FILES = ["cval.out", "outputmesh.out", "outputmesh_inlet.out"]


def clear_fortran_temp_files():
    for f in FORTRAN_TEMP_FILES:
        if os.path.exists(f):
            os.remove(f)


def clear_vtk_files():
    for file in os.listdir("."):
        if file.endswith(".vtk"):
            try:
                os.remove(file)
                print(f"Deleted VTK file: {file}")
            except Exception as e:
                print(f"Could not delete {file}: {e}")


# ============================================================
#                SIMULATION EXECUTION
# ============================================================
def execute_simulation(executable, input_str):
    result = subprocess.run(
        [executable],
        input=input_str.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stderr:
        print(f"[stderr] {result.stderr.decode()}")
    return result.stdout.decode()


def run_simulations(executable, lambda_vals, betav_vals, alphapar_fixed,
                    deltat, numtimesteps, time_indices, data_dir):

    start = time.time()

    output_files = [
        "snapshots.txt", "pressure.txt",
        "mesh_coor2.txt", "c-trace.txt", "b-trace.txt",
    ]
    for file in output_files:
        open(os.path.join(data_dir, file), "w").close()

    for betav in betav_vals:
        for lam in lambda_vals:
            print(f"\nRunning λ={lam:.3f}, β={betav:.3f}, α={alphapar_fixed:.3f} (fixed)")

            input_str = generate_input_string(
                lam, alphapar_fixed, betav, deltat, numtimesteps
            )
            clear_fortran_temp_files()
            _ = execute_simulation(executable, input_str)

            clear_vtk_files()
            filter_and_append_output_files(
                data_dir, time_indices, numtimesteps, lam, betav
            )

    print(f"\nAll simulations finished in {time.time() - start:.2f} sec.")


# ============================================================
#              INPUT GENERATION & TIME SAMPLING
# ============================================================
# M3 mesh resolution (from convergence study)
DX_BOX   = 0.2
DX_WALL  = 0.04
DX_INLET = 0.1


def generate_input_string(lambda1, alphapar, betav, deltat, numtimesteps):
    s = StringIO()
    s.write("&comppar\n")
    s.write(f"lambda   = {lambda1:.16f}\n")
    s.write(f"mobility = {alphapar:.16f}\n")
    s.write(f"betav    = {betav:.16f}\n")
    s.write(f"deltat   = {deltat:.16f}\n")
    s.write(f"numtimesteps = {numtimesteps}\n")
    s.write(f"dx_box   = {DX_BOX:.16f}\n")
    s.write(f"dx_wall  = {DX_WALL:.16f}\n")
    s.write(f"dx_inlet = {DX_INLET:.16f}\n")
    s.write("/\n")
    return s.getvalue()


def generate_log_time_steps(total_time, num_points, deltat):
    t_min = deltat
    t_max = total_time
    times = np.geomspace(t_min, t_max, num_points)
    indices = [round(t / deltat) for t in times]
    indices = [max(1, min(i, math.ceil(total_time / deltat))) for i in indices]
    return np.unique(indices)


def filter_and_append_output_files(data_dir, time_indices, numtimesteps, lam, betav):
    output_files = [
        "snapshots.txt", "pressure.txt",
        "mesh_coor2.txt", "c-trace.txt", "b-trace.txt",
    ]
    for file in output_files:
        if not os.path.exists(file):
            continue
        with open(file, "r") as f:
            lines = f.readlines()
        total_lines = len(lines)
        if total_lines == 0 or total_lines % numtimesteps != 0:
            print(f"Warning: {file}: {total_lines} lines not divisible by {numtimesteps} steps, skipping")
            os.remove(file)
            continue
        lines_per_step = total_lines // numtimesteps
        filtered = []
        for idx in time_indices:
            start = (idx - 1) * lines_per_step
            end = idx * lines_per_step
            filtered.extend(lines[start:end])
        dst = os.path.join(data_dir, file)
        with open(dst, "a") as f:
            f.writelines(filtered)
        os.remove(file)
        print(f"Filtered & appended {file} ({lines_per_step} lines/step) for λ={lam:.3f}, β={betav:.3f}")


def save_parameter_times(filepath, lambda_vals, betav_vals, alphapar_fixed,
                         deltat, numtimesteps, time_indices):
    with open(filepath, "w") as f:
        for betav in betav_vals:
            for lam in lambda_vals:
                for idx in time_indices:
                    t = deltat * idx
                    f.write(
                        f"{lam:.16f} {betav:.16f} {alphapar_fixed:.16f} "
                        f"{t:.16f} {idx}\n"
                    )


# ============================================================
if __name__ == "__main__":
    main()
