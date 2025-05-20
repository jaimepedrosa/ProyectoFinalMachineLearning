import os
import subprocess
import sys

if __name__ == "__main__":
    # Directorio raíz del proyecto (proyecto_ml/)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Directorio raíz al sys.path
    sys.path.append(root_dir)

    # Rutas de los scripts
    exp_i = os.path.join(root_dir, "src/experiments/run_experiment_i.py")
    exp_ii = os.path.join(root_dir, "src/experiments/run_experiment_ii.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir + (os.pathsep + env.get("PYTHONPATH", ""))

    print("\n=========== EJECUTANDO EXPERIMENTO I (con T1 y T2) ===========")
    subprocess.run(["python3", exp_i], env=env, check=True)

    print("\n=========== EJECUTANDO EXPERIMENTO II (sin T1 ni T2) ===========")
    subprocess.run(["python3", exp_ii], env=env, check=True)
