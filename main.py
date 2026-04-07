from utils.run_manager import get_next_run_path
from pytorch_pipeline import run_pytorch
from tensorflow_pipeline import run_tensorflow

framework = "tensorflow"  # or "tensorflow"

base_dir = f"results/{framework}"
save_path = get_next_run_path(base_dir)
max_samples = 992 #(31) 3264 or 

if framework == "pytorch":
    run_pytorch(save_path, max_samples)

elif framework == "tensorflow": 
    run_tensorflow(save_path)