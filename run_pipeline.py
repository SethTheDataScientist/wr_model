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

def run_command(command, cwd):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        raise Exception(f"Command failed with return code {result.returncode}")
    
def startup():
    run_command("poetry install", cwd=os.path.abspath("../wr_model"))
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

def run_data_ingestion():
    run_command("poetry run python ingest.py", cwd=os.path.abspath("../wr_model/data_ingestion"))

def run_model_training():
    run_command("poetry run python train.py", cwd=os.path.abspath("../wr_model/model_training"))

def run_model_evaluation():
    run_command("poetry run python evaluate.py", cwd=os.path.abspath("../wr_model/model_evaluation"))

if __name__ == "__main__":
    
    # set seed for reproducibility
    np.random.seed(123)


    # Set the MLflow tracking URI to the local server
    startup()
    # mlflow_proc = start_mlflow_ui()
    run_data_ingestion()
    run_model_training()
    run_model_evaluation()
    # mlflow_proc.terminate() 