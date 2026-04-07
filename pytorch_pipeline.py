import torch
from torchvision import datasets, models
from torch.utils.data import DataLoader
import time
from utils.save_results import save_metadata, save_cka, save_metrics, save_predictions, save_model_state_pytorch, save_model_structure_pytorch
from utils.cka_formating import get_activation_pytorch, compute_cka_matrix_pytorch
from torchvision import transforms

def clean_hooks(handles):
    print("CLEANING HOOKS.")
    for h in handles:
        h.remove()

def create_pytorch_model():
    print("CREATING PYTORCH MODEL.")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    return model

def get_dataloader(path, preprocess_fn):
    print("GETTING DATA LOADER.")
    dataset = datasets.ImageFolder(path, transform=preprocess_fn)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

def run_inference_pytorch(model, loader, max_samples = 992):
    print("RUNNING PYTORCH INFERENCE WITH " + str(max_samples) + " SAMPLES.")
    # add hooks
    activations = {}
    handles = []
    handles.append(model.layer1.register_forward_hook(get_activation_pytorch("layer1", activations)))
    handles.append(model.layer2.register_forward_hook(get_activation_pytorch("layer2", activations)))
    handles.append(model.layer3.register_forward_hook(get_activation_pytorch("layer3", activations)))
    handles.append(model.layer4.register_forward_hook(get_activation_pytorch("layer4", activations)))

    # run inference
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
    clean_hooks(handles)
    
    print(f"Processing time: {processing_time:.2f} seconds.")
    print("Total processed: " + str(count))

    # collect inference activations and outputs
    for k in activations:
        activations[k] = torch.cat(activations[k], dim=0)

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)

    return all_outputs, all_labels, activations, processing_time

def run_cka_pytorch(activations):
    print("RUNNING CKA.")
    cka_start_time = time.time()
    cka_matrix, layers = compute_cka_matrix_pytorch(activations)
    cka_end_time = time.time()
    duration = cka_end_time - cka_start_time
    print(f"CKA time: {cka_end_time - cka_start_time:.2f} seconds.")
    # save_cka(cka_matrix, layers, save_path, "PyTorch")
    return cka_matrix, layers, duration

def save_outputs_pytorch(model, save_path, all_outputs, all_labels, cka_matrix, layer_names, version_name, metadata):
    print("SAVING PYTORCH METRICS.")
    
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
    probs = torch.softmax(all_outputs, dim=1)
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

    save_cka(cka_matrix, layer_names, save_path, version_name)
    save_metadata(metadata, save_path)


# resnet18
def run_pytorch(model, save_path, max_samples = 992):
    print("RUNNING PYTORCH PIPELINE WITH " + str(max_samples) + " SAMPLES.")

    # download model
    # weights = models.ResNet18_Weights.DEFAULT
    # model = models.resnet18(weights=weights)
    # model.eval() # set to evaluation mode

    # get and prepare dataset
    # preprocess = weights.transforms()
    # preprocess = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),  # converts to [0,1]
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    # dataset = datasets.ImageFolder("ImageNet/", transform=preprocess_pt)
    # loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # # add hooks
    # activations = {}
    # handles = []
    # handles.append(model.layer1.register_forward_hook(get_activation_pytorch("layer1", activations)))
    # handles.append(model.layer2.register_forward_hook(get_activation_pytorch("layer2", activations)))
    # handles.append(model.layer3.register_forward_hook(get_activation_pytorch("layer3", activations)))
    # handles.append(model.layer4.register_forward_hook(get_activation_pytorch("layer4", activations)))

    # # run inference
    # all_outputs = []
    # all_labels = []
    # count = 0
    # activations.clear()
    # processing_start_time = time.time()
    # with torch.no_grad():
    #     for images, labels in loader:
    #         if count >= max_samples:
    #             break

    #         remaining = max_samples - count
    #         if images.size(0) > remaining:
    #             images = images[:remaining]
    #             labels = labels[:remaining]

    #         outputs = model(images)
    #         count += images.size(0)

    #         all_outputs.append(outputs)
    #         all_labels.append(labels)

    # processing_end_time = time.time()

    # print(f"Processing time: {processing_end_time - processing_start_time:.2f} seconds.")
    # print("Total processed: " + str(count))

    # # collect inference activations and outputs
    # for k in activations:
    #     activations[k] = torch.cat(activations[k], dim=0)

    # all_outputs = torch.cat(all_outputs)
    # all_labels = torch.cat(all_labels)



    # # SAVE OUTPUTS
    # preds = all_outputs.argmax(dim=1)
    # # categories = weights.meta["categories"]

    # # SAVE MODEL STATE
    # save_model_state_pytorch(model, save_path)

    # # SAVE MODEL STRUCTURE
    # save_model_structure_pytorch(model, save_path)

    # # TOP-1
    # top1 = all_outputs.argmax(dim=1)
    # top1_acc = (preds == all_labels).float().mean()
    # print(f"Top-1 Accuracy: {top1_acc:.8f}")

    # # TOP-5
    # top5 = all_outputs.topk(5, dim=1).indices
    # correct_top5 = (top5 == all_labels.unsqueeze(1)).any(dim=1)
    # top5_acc = correct_top5.float().mean()
    # print(f"Top-5 Accuracy: {top5_acc:.8f}")

    # # TOP-10
    # top10 = all_outputs.topk(10, dim=1).indices
    # correct_top10 = (top10 == all_labels.unsqueeze(1)).any(dim=1)
    # top10_acc = correct_top10.float().mean()
    # print(f"Top-10 Accuracy: {top10_acc:.8f}")

    # metrics = {
    #     "top1": float(top1_acc),
    #     "top5": float(top5_acc),
    #     "top10": float(top10_acc)
    # }
    # save_metrics(metrics, save_path)

    # # CONFIDENCE
    # probs = torch.softmax(all_outputs, dim=1)
    # top1_conf = probs.gather(1, top1.unsqueeze(1)).squeeze(1)
    # top5_conf = probs.topk(5, dim=1).values
    # top10_conf = probs.topk(10, dim=1).values

    # top1 = top1.cpu().numpy()
    # top5 = top5.cpu().numpy()
    # top10 = top10.cpu().numpy()

    # top1_conf = top1_conf.cpu().numpy()
    # top5_conf = top5_conf.cpu().numpy()
    # top10_conf = top10_conf.cpu().numpy()

    # all_labels = all_labels.cpu().numpy()

    # save_predictions(
    #     top1,
    #     top5,
    #     top10,
    #     top1_conf,
    #     top5_conf,
    #     top10_conf,
    #     all_labels,  
    #     save_path
    # )

    # # TOTAL PARAMS
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")

    # # TRAINABLE PARAMS
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Trainable parameters: {trainable_params:,}")

    # # run CKA
    # cka_start_time = time.time()
    # cka_matrix, layers = compute_cka_matrix_pytorch(activations)
    # cka_end_time = time.time()
    # print(f"CKA time: {cka_end_time - cka_start_time:.2f} seconds.")
    # save_cka(cka_matrix, layers, save_path, "PyTorch")

    # metadata = {
    #     "model": "ResNet18",
    #     "framework": "PyTorch",
    #     "samples": max_samples,
    #     "inference_time": float(processing_end_time - processing_start_time),
    #     "cka_time": float(cka_end_time - cka_start_time),
    #     "total_params": total_params,
    #     "trainable_params": trainable_params
    # }
    # save_metadata(metadata, save_path)

    # clean_hooks(handles)