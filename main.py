import os
import time
from utils.save_results import save_cka
from utils.run_manager import get_next_run_path
from utils.cka_formating import compute_cross_cka
from utils.data_loader import compute_visuals
from pytorch_pipeline import py_run_vgg16, py_run_resnet50, py_run_mobilenetv2
from tensorflow_pipeline import tf_run_vgg16, tf_run_resnet50, tf_run_mobilenetv2

print("START")

dataset_path = "ImageNetSmall/"
base_dir = f"results/"
max_samples = 992#34976
batch_size = 32
debugging = False
own_preprocessing = False

def create_subdirectories(run_path, model_type):
    base = os.path.join(run_path, model_type)

    save_path_py = os.path.join(base, "pytorch")
    save_path_tf = os.path.join(base, "tensorflow")

    os.makedirs(save_path_py, exist_ok=True)
    os.makedirs(save_path_tf, exist_ok=True)

    return base, save_path_py, save_path_tf

# def run_pytorch(save_path_base_py):
#     print("RUNNING PYTORCH PIPELINE.")
    
#     model_pt = create_pytorch_model()
#     data_loader = get_dataloader(dataset_path, preprocess_pt)
#     all_outputs, all_labels, activations_py, inference_time = run_inference_pytorch(model_pt, data_loader, max_samples)
#     # cka_matrix, layers, cka_duration = run_cka_pytorch(activations_py)

#     # total_params = sum(p.numel() for p in model_pt.parameters())
#     # trainable_params = sum(p.numel() for p in model_pt.parameters() if p.requires_grad)
#     # metadata = {
#     #     "model": "ResNet18",
#     #     "framework": "PyTorch",
#     #     "type": "baseline",
#     #     "samples": max_samples,
#     #     "inference_time": inference_time,
#     #     "cka_time": cka_duration,
#     #     "total_params": total_params,
#     #     "trainable_params": trainable_params
#     # }
#     # #                   model, save_path, all_outputs, all_labels, cka_matrix, layer_names, version_name, metadata
#     # save_outputs_pytorch(model_pt, save_path_base_py, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)
#     return activations_py

# def run_tensorflow(save_path_base_tf):
#     print("RUNNING TENSORFLOW PIPELINE.")
#     # BASELINE
#     model_tf = create_tensorflow_model()
#     data = get_data_tf(dataset_path, preprocess_tf, max_samples, batch_size, model_type)
#     activations_tf, all_outputs, all_labels, inference_time = run_inference_tensorflow(model_tf, data, max_samples)
#     cka_matrix, layers, cka_duration = run_cka_tensorflow(activations_tf)

#     total_params = model_tf.count_params()
#     trainable_params = int(np.sum([ np.prod(v.shape) for v in model_tf.trainable_weights]))
    
#     metadata = {
#         "model": "ResNet18",
#         "framework": "PyTorch",
#         "type": "baseline",
#         "samples": max_samples,
#         "inference_time": inference_time,
#         "cka_time": cka_duration,
#         "total_params": total_params,
#         "trainable_params": trainable_params
#     }

#     save_outputs_tensorflow(model_tf, save_path_base_tf, all_outputs, all_labels, cka_matrix, layers, "PyTorch Baseline", metadata)

#     return activations_tf

def run_mobilenetv2(run_path):
    model_name = "MobileNetV2"
    base, save_path_py, save_path_tf = create_subdirectories(run_path, model_name)

    # pytorch
    py_activations = py_run_mobilenetv2(save_path_py, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)
    # tensorflow
    tf_activations = tf_run_mobilenetv2(save_path_tf, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)

    if debugging == True:
        for k in py_activations:
            print("PT:", k, py_activations[k].shape)
            print("TF:", k, tf_activations[k].shape)

    # # compute CKA
    print("COMPUTING CKA FOR BOTH.")
    cka_start_time = time.time()
    cka_matrix, stages = compute_cross_cka(py_activations, tf_activations)
    cka_end_time = time.time()
    cka_duration = cka_end_time - cka_start_time
    print("CKA Processing time: " + str(cka_duration))
    save_cka(cka_matrix, stages, base, "PT_vs_TF_" + model_name)

    print("DONE.")

def run_resnet50(run_path):
    model_name = "ResNet50"
    base, save_path_py, save_path_tf = create_subdirectories(run_path, model_name)

    # pytorch
    py_activations = py_run_resnet50(save_path_py, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)
    # tensorflow
    tf_activations = tf_run_resnet50(save_path_tf, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)

    if debugging == True:
        for k in py_activations:
            print("PT:", k, py_activations[k].shape)
            print("TF:", k, tf_activations[k].shape)

    # compute CKA
    print("COMPUTING CKA FOR BOTH.")
    cka_start_time = time.time()
    cka_matrix, stages = compute_cross_cka(py_activations, tf_activations)
    cka_end_time = time.time()
    cka_duration = cka_end_time - cka_start_time
    print("CKA Processing time: " + str(cka_duration))
    save_cka(cka_matrix, stages, base, "PT_vs_TF_" + model_name)

    print("DONE.")

def run_vgg16(run_path):
    model_name = "VGG16"
    base, save_path_py, save_path_tf = create_subdirectories(run_path, model_name)

    # pytorch
    py_activations = py_run_vgg16(save_path_py, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)
    # tensorflow
    tf_activations = tf_run_vgg16(save_path_tf, model_name, max_samples, dataset_path, debugging, own_preprocessing, batch_size)

    if debugging == True:
        for k in py_activations:
            print("PT:", k, py_activations[k].shape)
            print("TF:", k, tf_activations[k].shape)

    # compute CKA
    print("COMPUTING CKA FOR BOTH.")
    cka_start_time = time.time()
    cka_matrix, stages = compute_cross_cka(py_activations, tf_activations)
    cka_end_time = time.time()
    cka_duration = cka_end_time - cka_start_time
    print("CKA Processing time: " + str(cka_duration))
    save_cka(cka_matrix, stages, base, "PT_vs_TF_" + model_name)

    print("DONE.")

def run():
    # run_num = "Run_1/"
    run_path, run_num = get_next_run_path(base_dir)
    print("---------MOBILENET---------")
    run_mobilenetv2(run_path)
    compute_visuals(base_dir +""+ run_num, "MobileNetV2/")
    print("----------RESNET----------")
    run_resnet50(run_path)
    compute_visuals(base_dir +""+ run_num, "Resnet50/")
    print("-----------VGG-----------")
    run_vgg16(run_path)
    compute_visuals(base_dir +""+ run_num, "VGG16/")

    print("DONE WITH PIPELINE.")

# def run():
#     frameworks = ['pytorch', 'tensorflow']
    
#     # run pytorch
#     save_path_base_py, save_path_converted_py = create_subdirectories(run_path, "pytorch_baseline", "pt_to_tf_converted")
#     py_activations = run_pytorch(save_path_base_py)
#     # converted_tf_model = run_converted_base_pytorch(save_path_converted_py)

#     data = get_data_tf(dataset_path, preprocess_tf, max_samples, batch_size)

#     activations_tf, all_outputs, all_labels, inference_time = run_inference_tensorflow(converted_tf_model, data, max_samples)
#     cka_matrix, layers, cka_duration = run_cka_tensorflow(activations_tf)


#     # total_params = converted_tf_model.count_params()
#     # trainable_params = int(np.sum([ np.prod(v.shape) for v in converted_tf_model.trainable_weights]))
    
#     metadata = {
#         "model": "ResNet18",
#         "framework": "PyTorch",
#         "type": "baseline",
#         "samples": max_samples,
#         "inference_time": inference_time,
#         "cka_time": cka_duration,
#         # "total_params": total_params,
#         # "trainable_params": trainable_params
#     }

#     save_outputs_tensorflow(model, save_path_converted_py, all_outputs, all_labels, cka_matrix, layers, "PyTorch2TF Conversion", metadata)




#     # run tensorflow
#     save_path_base_tf, save_path_converted_tf = create_subdirectories(run_path, "tensorflow_baseline", "tf_to_pt_converted")
#     tf_activations = run_tensorflow(save_path_base_tf)
#     # run_converted_tensorflow(save_path_converted_tf)

#     # run cka on both tf and py
#     print("COMPUTING CKA FOR BOTH.")
#     cka_matrix, stages = compute_cross_cka(py_activations, tf_activations)
#     save_cka(cka_matrix, stages, run_path, "PT_vs_TF")

#     print("DONE.")

run()