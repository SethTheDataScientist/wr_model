import mlflow
import os
import subprocess
import numpy as np

def start_mlflow_ui(path_to_project: str = "../wr_model", port: int = 5000):
    """
    Starts the MLflow UI in the background using Poetry.

    Args:
        path_to_project (str): Relative or absolute path to the poetry project.
        port (int): Port for the MLflow UI (default is 5000).

    Returns:
        subprocess.Popen: The process handle in case you want to terminate it later.
    """
    abs_path = os.path.abspath(path_to_project)
    
    process = subprocess.Popen(
        ['poetry', 'run', 'mlflow', 'ui', '--port', str(port)],
        cwd=abs_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    print(f"MLflow UI started at http://localhost:{port} (PID: {process.pid})")
    return process

# Set the MLflow tracking URI to the local server
mlflow_proc = start_mlflow_ui()
mlflow_proc.terminate() 