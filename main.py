import os
import numpy as np
from utils.run_manager import get_next_run_path
from pytorch_pipeline import run_inference_pytorch, create_pytorch_model, get_dataloader, run_cka_pytorch, save_outputs_pytorch
from tensorflow_pipeline import run_inference_tensorflow, create_tensorflow_model, get_data_tf, run_cka_tensorflow, save_outputs_tensorflow
# from conversion_pipeline import convert_pt_to_tf, convert_tf_to_py
from utils.preprocessing import preprocess_pt, preprocess_tf

print("START")

framework = "tensorflow"  # or "tensorflow"
dataset_path = "ImageNet/"

base_dir = f"results/{framework}"
run_path = get_next_run_path(base_dir)
max_samples = 992 #(31) 3264 or 
batch_size = 32

def create_subdirectories(run_path, baseline_name, converted_name):
    save_path_base = os.path.join(run_path, baseline_name)
    save_path_converted = os.path.join(run_path, converted_name)
    os.makedirs(save_path_base, exist_ok=True)
    os.makedirs(save_path_converted, exist_ok=True)

    return save_path_base, save_path_converted

if framework == "pytorch":
    print("RUNNING PYTORCH PIPELINE.")
    save_path_base_py, save_path_converted_py = create_subdirectories(run_path, "pytorch_baseline", "pt_to_tf_converted")

    # BASELINE
    model_pt = create_pytorch_model()
    data_loader = get_dataloader(dataset_path, preprocess_pt)
    all_outputs, all_labels, activations_py, inference_time = run_inference_pytorch(model_pt, data_loader, max_samples)
    cka_matrix, layers, cka_duration = run_cka_pytorch(activations_py)

    total_params = sum(p.numel() for p in model_pt.parameters())
    trainable_params = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
    metadata = {
        "model": "ResNet18",
        "framework": "PyTorch",
        "type": "baseline",
        "samples": max_samples,
        "inference_time": inference_time,
        "cka_time": cka_duration,
        "total_params": total_params,
        "trainable_params": trainable_params
    }
    #                   model, save_path, all_outputs, all_labels, cka_matrix, layer_names, version_name, metadata
    save_outputs_pytorch(model_pt, save_path_base_py, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)

    # CONVERTED
    # model_tf = convert_pt_to_tf(model_pt)
    # data_loader = get_dataloader(dataset_path, preprocess_pt)
    # all_outputs, all_labels, activations, metadata = run_pytorch(model_pt, data_loader, max_samples)
    # cka_matrix, layers, duration = run_cka_pytorch(activations)
    # save_outputs_pytorch(model_pt, save_path_base, all_outputs, all_labels, cka_matrix, layers, "PyTorch to TF Converted Model", metadata)

elif framework == "tensorflow": 
    print("RUNNING TENSORFLOW PIPELINE.")
    save_path_base_tf, save_path_converted_tf = create_subdirectories(run_path, "tensorflow_baseline", "tf_to_pt_converted")

    # BASELINE
    model_tf = create_tensorflow_model()
    data = get_data_tf(dataset_path, preprocess_tf, max_samples, batch_size)
    activations_tf, all_outputs, all_labels, inference_time = run_inference_tensorflow(model_tf, data, max_samples)
    if activations_tf is not None:
        cka_matrix, layers, cka_duration = run_cka_tensorflow(activations_tf)
    else:
        cka_matrix, layers, cka_duration = None, None, None

    total_params = model_tf.count_params()
    trainable_params = int(np.sum([ np.prod(v.shape) for v in model_tf.trainable_weights]))
    
    metadata = {
        "model": "ResNet18",
        "framework": "PyTorch",
        "type": "baseline",
        "samples": max_samples,
        "inference_time": inference_time,
        "cka_time": cka_duration,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_tensorflow(model_tf, save_path_base_tf, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)

#     run_tensorflow(model_tf, save_path_base, max_samples)
#     model_pt = convert_tf_to_py(model_tf)
#     run_pytorch(model_tf, save_path_converted, max_samples)

# PREPROCESS_CONFIG = {
#     "resize": 256,
#     "crop": 224,
#     "mean": [0.485, 0.456, 0.406],
#     "std": [0.229, 0.224, 0.225]
# }