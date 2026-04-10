import os
import numpy as np
from utils.run_manager import get_next_run_path
from pytorch_pipeline import run_inference_pytorch, create_pytorch_model, get_dataloader, run_cka_pytorch, save_outputs_pytorch
from tensorflow_pipeline import run_inference_tensorflow, create_tensorflow_model, get_data_tf, run_cka_tensorflow, save_outputs_tensorflow
# from conversion_pipeline import convert_pt_to_tf, convert_tf_to_py
from utils.preprocessing import preprocess_pt, preprocess_tf
from utils.cka_formating import compute_cka_matrix, compute_cross_cka
from utils.save_results import save_cka

print("START")

framework = "tensorflow"  # or "tensorflow"
dataset_path = "ImageNetSmall/"

base_dir = f"results/"
run_path = get_next_run_path(base_dir)
max_samples = 992 #(31) 3264 or 
batch_size = 32

print("was able to run main.")

def create_subdirectories(run_path, baseline_name, converted_name):
    save_path_base = os.path.join(run_path, baseline_name)
    save_path_converted = os.path.join(run_path, converted_name)
    os.makedirs(save_path_base, exist_ok=True)
    os.makedirs(save_path_converted, exist_ok=True)

    return save_path_base, save_path_converted

def run_pytorch(save_path_base_py):
    print("RUNNING PYTORCH PIPELINE.")
    
    model_pt = create_pytorch_model()
    data_loader = get_dataloader(dataset_path, preprocess_pt)
    all_outputs, all_labels, activations_py, inference_time = run_inference_pytorch(model_pt, data_loader, max_samples)
    # cka_matrix, layers, cka_duration = run_cka_pytorch(activations_py)

    # total_params = sum(p.numel() for p in model_pt.parameters())
    # trainable_params = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
    # metadata = {
    #     "model": "ResNet18",
    #     "framework": "PyTorch",
    #     "type": "baseline",
    #     "samples": max_samples,
    #     "inference_time": inference_time,
    #     "cka_time": cka_duration,
    #     "total_params": total_params,
    #     "trainable_params": trainable_params
    # }
    # #                   model, save_path, all_outputs, all_labels, cka_matrix, layer_names, version_name, metadata
    # save_outputs_pytorch(model_pt, save_path_base_py, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)
    return activations_py

def run_converted_pytorch(save_path_converted_py):
    return None

def run_tensorflow(save_path_base_tf):
    print("RUNNING TENSORFLOW PIPELINE.")
    # BASELINE
    model_tf = create_tensorflow_model()
    data = get_data_tf(dataset_path, preprocess_tf, max_samples, batch_size)
    activations_tf, all_outputs, all_labels, inference_time = run_inference_tensorflow(model_tf, data, max_samples)
    # if activations_tf is not None:
    #     cka_matrix, layers, cka_duration = run_cka_tensorflow(activations_tf)
    # else:
    #     cka_matrix, layers, cka_duration = None, None, None

    # total_params = model_tf.count_params()
    # trainable_params = int(np.sum([ np.prod(v.shape) for v in model_tf.trainable_weights]))
    
    # metadata = {
    #     "model": "ResNet18",
    #     "framework": "PyTorch",
    #     "type": "baseline",
    #     "samples": max_samples,
    #     "inference_time": inference_time,
    #     "cka_time": cka_duration,
    #     "total_params": total_params,
    #     "trainable_params": trainable_params
    # }

    # save_outputs_tensorflow(model_tf, save_path_base_tf, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)

    return activations_tf

def run_converted_tensorflow(save_path_converted_tf):
    return None

def run():
    # run pytorch
    save_path_base_py, save_path_converted_py = create_subdirectories(run_path, "pytorch_baseline", "pt_to_tf_converted")
    py_activations = run_pytorch(save_path_base_py)
    # run_converted_pytorch(save_path_converted_py)

    # run tensorflow
    save_path_base_tf, save_path_converted_tf = create_subdirectories(run_path, "tensorflow_baseline", "tf_to_pt_converted")
    tf_activations = run_tensorflow(save_path_base_tf)
    # run_converted_tensorflow(save_path_converted_tf)

    # run cka on both tf and py
    print("COMPUTING CKA FOR BOTH.")
    cka_matrix, stages = compute_cross_cka(py_activations, tf_activations)
    save_cka(cka_matrix, stages, run_path, "PT_vs_TF")
    # save_cka(cka_matrix, layer_names, save_path, framework):

# run()

# PREPROCESS_CONFIG = {
#     "resize": 256,
#     "crop": 224,
#     "mean": [0.485, 0.456, 0.406],
#     "std": [0.229, 0.224, 0.225]
# }

    # combined = {
    #     "pt_stage1": py_activations["stage1"],
    #     "tf_stage1": tf_activations["stage1"],
    #     "pt_stage2": py_activations["stage2"],
    #     "tf_stage2": tf_activations["stage2"],
    #     "pt_stage3": py_activations["stage3"],
    #     "tf_stage3": tf_activations["stage3"],
    #     "pt_stage4": py_activations["stage4"],
    #     "tf_stage4": tf_activations["stage4"],
    # }