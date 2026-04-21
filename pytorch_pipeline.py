import time
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from utils.cka_formating import get_activation_pytorch
from utils.preprocessing import preprocess_pytorch, preprocess_py_mobilenet
from utils.save_results import save_metadata, save_metrics, save_predictions, save_model_state_pytorch, save_model_structure_pytorch

def clean_hooks(handles):
    print("CLEANING HOOKS.")
    for h in handles:
        h.remove()

def create_resnet50():
    print("PY: CREATING RESNET50 MODEL.")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    return model

def create_mobilenetv2():
    print("PY: CREATING MOBILENETV2 MODEL.")
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    return model

def create_vgg16():
    print("PY: CREATING VGG16 MODEL.")
    weights = models.VGG16_Weights.DEFAULT
    model = models.vgg16(weights=weights)
    model.eval()
    return model

def get_dataloader(path, preprocess_fn, model_name, debugging, own_preprocessing, batch_size=32):
    print("PY: GETTING DATA LOADER.")
    # dataset = datasets.ImageFolder(path, transform=(preprocess_fn(own_preprocessing, model_name)))
    
    dataset = datasets.ImageFolder(
        path,
        transform=lambda img: preprocess_fn(img, own_preprocessing, model_name)
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if debugging == True:
        for x, y in loader:
            print("PT batch shape:", x.shape)
            print("PT batch range:", x.min().item(), x.max().item())
            break
    return loader

def save_outputs_pytorch(model, save_path, all_outputs, all_labels, metadata, debugging):

    print("PY: SAVING METRICS.")
    probs = all_outputs
    if debugging == True:
        print("is softmax?")
        print(probs[0].sum())
    
    # SAVE MODEL STATE
    save_model_state_pytorch(model, save_path)
    # SAVE MODEL STRUCTURE
    save_model_structure_pytorch(model, save_path)

    # TOP-1
    top1 = all_outputs.argmax(dim=1)
    top1_acc = (top1 == all_labels).float().mean()
    print(f"Top-1 Accuracy: {top1_acc:.8f}")

    # TOP-5
    top5 = all_outputs.topk(5, dim=1).indices
    correct_top5 = (top5 == all_labels.unsqueeze(1)).any(dim=1)
    top5_acc = correct_top5.float().mean()
    print(f"Top-5 Accuracy: {top5_acc:.8f}")

    # TOP-10
    top10 = all_outputs.topk(10, dim=1).indices
    correct_top10 = (top10 == all_labels.unsqueeze(1)).any(dim=1)
    top10_acc = correct_top10.float().mean()
    print(f"Top-10 Accuracy: {top10_acc:.8f}")

    metrics = {
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "top10": float(top10_acc)
    }
    save_metrics(metrics, save_path)

    # CONFIDENCE
    top1_conf = probs.gather(1, top1.unsqueeze(1)).squeeze(1)
    top5_conf = probs.topk(5, dim=1).values
    top10_conf = probs.topk(10, dim=1).values

    top1 = top1.cpu().numpy()
    top5 = top5.cpu().numpy()
    top10 = top10.cpu().numpy()

    top1_conf = top1_conf.cpu().numpy()
    top5_conf = top5_conf.cpu().numpy()
    top10_conf = top10_conf.cpu().numpy()

    all_labels = all_labels.cpu().numpy()

    save_predictions(
        top1,
        top5,
        top10,
        top1_conf,
        top5_conf,
        top10_conf,
        all_labels,  
        save_path
    )

    # save_cka(cka_matrix, layer_names, save_path, version_name)
    save_metadata(metadata, save_path)

def run_inference(model_type, model, loader, activations, debugging, max_samples = 1):
    print("PY: RUNNING INFERENCE.")
    all_outputs = []
    all_labels = []
    count = 0
    activations.clear()
    processing_start_time = time.time()
    with torch.no_grad():
        for images, labels in loader:
            if count >= max_samples:
                break

            remaining = max_samples - count
            if images.size(0) > remaining:
                images = images[:remaining]
                labels = labels[:remaining]

            outputs = model(images)
            count += images.size(0)

            all_outputs.append(outputs)
            all_labels.append(labels)

    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time
    
    print(f"Processing time: {processing_time:.2f} seconds.")
    print("Total processed: " + str(count))

    # collect inference activations and outputs
    for k in activations:
        activations[k] = np.concatenate(activations[k], axis=0)

    if debugging == True:
        for name, act in activations.items():
            print(f"PT {name} shape:", act.shape)

    all_outputs = torch.cat(all_outputs)
    all_outputs = F.softmax(all_outputs, dim=1) 
    all_labels = torch.cat(all_labels)

    return all_outputs, all_labels, activations, processing_time

def register_hooks_resnet50(model):
    activations = {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "stage4": [],
        "stage5": []
    }
    
    handles = []
    handles.append(model.layer1.register_forward_hook(get_activation_pytorch("stage1", activations)))
    handles.append(model.layer2.register_forward_hook(get_activation_pytorch("stage2", activations)))
    handles.append(model.layer3.register_forward_hook(get_activation_pytorch("stage3", activations)))
    handles.append(model.layer4.register_forward_hook(get_activation_pytorch("stage4", activations)))
    handles.append(model.avgpool.register_forward_hook(get_activation_pytorch("stage5", activations)))

    return activations, handles

def register_hooks_mobilenet(model):
    activations = {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "stage4": [],
        "stage5": []
    }

    handles = []
    handles.append(model.features[2].register_forward_hook(get_activation_pytorch("stage1", activations)))
    handles.append(model.features[4].register_forward_hook(get_activation_pytorch("stage2", activations)))
    handles.append(model.features[7].register_forward_hook(get_activation_pytorch("stage3", activations)))
    handles.append(model.features[14].register_forward_hook(get_activation_pytorch("stage4", activations)))
    handles.append(model.features[18].register_forward_hook(get_activation_pytorch("stage5", activations)))

    return activations, handles

def register_hooks_vgg16(model):
    activations = {
        "stage1": [],
        "stage2": [],
        "stage3": [],
        "stage4": [],
        "stage5": []
    }

    handles = []

    # MaxPool layers mark stage ends in VGG16
    handles.append(model.features[4].register_forward_hook(get_activation_pytorch("stage1", activations)))
    handles.append(model.features[9].register_forward_hook(get_activation_pytorch("stage2", activations)))
    handles.append(model.features[16].register_forward_hook(get_activation_pytorch("stage3", activations)))
    handles.append(model.features[23].register_forward_hook(get_activation_pytorch("stage4", activations)))
    handles.append(model.avgpool.register_forward_hook(get_activation_pytorch("stage5", activations)))

    return activations, handles

def py_run_mobilenetv2(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size):
    print("RUNNING PY MOBILENETV2 PIPELINE.")
    if runConverted == False:
        model = create_mobilenetv2()
    else:
        model = None
    data_loader = get_dataloader(dataset_path, preprocess_py_mobilenet, model_name, debugging, own_preprocessing, batch_size)
    activations, handles = register_hooks_mobilenet(model)
    all_outputs, all_labels, activations, inference_time = run_inference(model_name, model, data_loader, activations, debugging, max_samples)
    clean_hooks(handles)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_pytorch(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations

def py_run_resnet50(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size):
    print("RUNNING PY RESNET50 PIPELINE.")
    if runConverted == False:
        model = create_resnet50()
    else:
        model = None
    data_loader = get_dataloader(dataset_path, preprocess_pytorch, model_name, debugging, own_preprocessing, batch_size)
    activations, handles = register_hooks_resnet50(model)
    all_outputs, all_labels, activations, inference_time = run_inference(model_name, model, data_loader, activations, debugging, max_samples)
    clean_hooks(handles)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_pytorch(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations

def py_run_vgg16(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size):
    print("RUNNING PY VGG16 PIPELINE.")
    if runConverted == False:
        model = create_vgg16()
    else:
        model = None
    data_loader = get_dataloader(dataset_path, preprocess_pytorch, model_name, debugging, own_preprocessing, batch_size)
    activations, handles = register_hooks_vgg16(model)
    all_outputs, all_labels, activations, inference_time = run_inference(model_name, model, data_loader, activations, debugging, max_samples)
    clean_hooks(handles)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_pytorch(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations
