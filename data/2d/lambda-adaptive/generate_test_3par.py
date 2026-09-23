"""
Generate TEST data for the (λ, βv, t) extrudate swell experiment.

Test design: 2D Latin Hypercube Sampling in interpolation regime with fixed alpha.
    λ  ∈ [1.5, 9.5]
    βv ∈ [0.15, 0.85]
    α  = 0.2 (fixed)

Output directory: lambda-beta-alpha-adaptive/
"""
import os
import shutil
import subprocess
import numpy as np
import time
import math
from io import StringIO
from scipy.stats import qmc


# ============================================================
#                MAIN PIPELINE (ENTRY POINT)
# ============================================================
def main():

    # -------------------------
    # Configuration
    # -------------------------
    data_dir = "lambda-beta-alpha_test"
    flow_vtk_dir = os.path.join(data_dir, "flow_vtk")

    deltat = 0.1
    numtimesteps = 200
    steady_state = False
    num_time_samples = 10
    alphapar_fixed = 0.2

    # -------------------------
    # 2D LHS parameter sampling in interpolation regime
    # -------------------------
    seed = 42
    n_samples = 50

    lambda_min, lambda_max = 1.5, 9.5
    betav_min,  betav_max  = 0.15, 0.85

    sampler = qmc.LatinHypercube(d=2, seed=seed)
    lhs_raw = sampler.random(n_samples)                         # (n, 2)
    scaled = qmc.scale(
        lhs_raw,
        [lambda_min, betav_min],
        [lambda_max, betav_max],
    )                                                           # (n, 2)

    lambda_values  = scaled[:, 0]
    betav_values   = scaled[:, 1]

    # Sort by lambda for nicer logging
    order = np.argsort(lambda_values)
    lambda_values   = lambda_values[order]
    betav_values    = betav_values[order]

    print(f"Generated {n_samples} 2D-LHS test samples (seed={seed}):")
    for i in range(n_samples):
        print(f"  [{i:3d}] λ={lambda_values[i]:.4f}  β={betav_values[i]:.4f}  α={alphapar_fixed:.4f} (fixed)")

    # -------------------------
    # Prepare filesystem
    # -------------------------
    prepare_data_directory(data_dir, flow_vtk_dir)

    # -------------------------
    # Time-step selection
    # -------------------------
    total_time = deltat * numtimesteps

    if steady_state:
        time_indices = [numtimesteps]
    else:
        time_indices = generate_log_time_steps(total_time, num_time_samples, deltat)

    # -------------------------
    # Save test-parameter table
    # -------------------------
    save_parameter_times(
        "parameters_test.txt",
        lambda_values, betav_values, alphapar_fixed,
        deltat, numtimesteps, time_indices
    )

    # -------------------------
    # Run all simulations
    # -------------------------
    run_simulations(
        executable="./extrudate_swell2d_c_test",
        lambda_vals=lambda_values,
        betav_vals=betav_values,
        alphapar_fixed=alphapar_fixed,
        deltat=deltat,
        numtimesteps=numtimesteps,
        time_indices=time_indices,
        vtk_dir=flow_vtk_dir,
        data_dir=data_dir,
    )

    # -------------------------
    # Final clean-up
    # -------------------------
    if os.path.exists("parameters_test.txt"):
        shutil.move("parameters_test.txt", os.path.join(data_dir, "parameters_test.txt"))
        print(f"Moved parameters_test.txt → {data_dir}")

    print("\nAll test simulations done!")


# ============================================================
#              FILESYSTEM & CLEANUP HELPERS
# ============================================================
def prepare_data_directory(data_dir, flow_vtk_dir):
    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(flow_vtk_dir):
        shutil.rmtree(flow_vtk_dir)
        print(f"Removed old VTK directory: {flow_vtk_dir}")
    os.makedirs(flow_vtk_dir)
    print(f"Created new VTK folder: {flow_vtk_dir}")

    files_to_delete = [
        "parameters_test.txt", "snapshots_test.txt", "pressure_test.txt",
        "mesh_coor2_test.txt", "c-trace_test.txt", "b-trace_test.txt",
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


def move_vtk_files(lam, betav, alphapar, vtk_dir):
    folder = os.path.join(vtk_dir, f"lambda-{lam:.3f}_beta-{betav:.3f}_alpha-{alphapar:.3f}")
    os.makedirs(folder, exist_ok=True)

    for file in os.listdir("."):
        if file.endswith(".vtk"):
            src = os.path.join(".", file)
            dst = os.path.join(folder, file)
            shutil.move(src, dst)
            print(f"Moved {file} -> {folder}")


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
                    deltat, numtimesteps, time_indices, vtk_dir, data_dir):

    start = time.time()
    n_samples = len(lambda_vals)

    output_files = [
        "snapshots_test.txt", "pressure_test.txt",
        "mesh_coor2_test.txt", "c-trace_test.txt", "b-trace_test.txt",
    ]
    for file in output_files:
        open(os.path.join(data_dir, file), "w").close()

    for i in range(n_samples):
        lam      = lambda_vals[i]
        betav    = betav_vals[i]
        alphapar = alphapar_fixed

        print(f"\n[{i+1}/{n_samples}] Running λ={lam:.3f}, β={betav:.3f}, α={alphapar:.3f}")

        input_str = generate_input_string(
            lam, alphapar, betav, deltat, numtimesteps
        )
        clear_fortran_temp_files()
        _ = execute_simulation(executable, input_str)

        move_vtk_files(lam, betav, alphapar, vtk_dir)
        filter_and_append_output_files(
            data_dir, time_indices, numtimesteps, lam, betav, alphapar
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


def filter_and_append_output_files(data_dir, time_indices, numtimesteps, lam, betav, alphapar):
    files = [
        "snapshots_test.txt", "pressure_test.txt",
        "mesh_coor2_test.txt", "c-trace_test.txt", "b-trace_test.txt",
    ]
    for file in files:
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
        print(f"Filtered & appended {file} ({lines_per_step} lines/step) for \u03bb={lam:.3f}, \u03b2={betav:.3f}, \u03b1={alphapar:.3f}")


def save_parameter_times(filepath, lambda_vals, betav_vals, alphapar_fixed,
                         deltat, numtimesteps, time_indices):
    n_samples = len(lambda_vals)
    with open(filepath, "w") as f:
        for i in range(n_samples):
            for idx in time_indices:
                t = deltat * idx
                f.write(
                    f"{lambda_vals[i]:.16f} {betav_vals[i]:.16f} {alphapar_fixed:.16f} "
                    f"{t:.16f} {idx}\n"
                )


# ============================================================
if __name__ == "__main__":
    main()
