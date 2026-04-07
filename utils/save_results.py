import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

# metrics example usage
# metrics = {
#     "top1": float(top1_acc),
#     "top5": float(top5_acc),
#     "top10": float(top10_acc)
# }

# save_metrics(metrics, "results/resnet_tf/")

def save_metrics(metrics_dict, save_path):
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)


# save_predictions(
#     top1,
#     top5,
#     top10,
#     top1_conf,
#     top5_conf,
#     top10_conf,
#     all_labels,  
#     "results/resnet50_tf_3264/"
# )
def save_predictions(top1, top5, top10,
                     top1_conf, top5_conf, top10_conf, 
                     all_labels, save_path):

    os.makedirs(save_path, exist_ok=True)

    # predictions
    np.save(os.path.join(save_path, "top1.npy"), top1)
    np.save(os.path.join(save_path, "top5.npy"), top5)
    np.save(os.path.join(save_path, "top10.npy"), top10)

    # confidences
    np.save(os.path.join(save_path, "top1_conf.npy"), top1_conf)
    np.save(os.path.join(save_path, "top5_conf.npy"), top5_conf)
    np.save(os.path.join(save_path, "top10_conf.npy"), top10_conf)

    np.save(os.path.join(save_path, "labels.npy"), all_labels)

# top1, top5, top10
# confidence - mean top1 summary
# cka matrix
# layer names

def save_cka(cka_matrix, layer_names, save_path, framework):
    os.makedirs(save_path, exist_ok=True)

    np.save(os.path.join(save_path, "cka.npy"), cka_matrix)

    # also save layer names
    with open(os.path.join(save_path, "layers.json"), "w") as f:
        json.dump(layer_names, f)

    plt.figure(figsize=(6, 5))
    plt.imshow(cka_matrix, interpolation='nearest')
    plt.colorbar(label="CKA Similarity")

    plt.xticks(range(len(layer_names)), layer_names, rotation=45)
    plt.yticks(range(len(layer_names)), layer_names)

    plt.title("CKA Similarity Between Layers - " + framework)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "cka_heatmap_"+framework+".png"), dpi=300)
    # plt.show()


# metadata = {
#     "model": "ResNet50",
#     "framework": "TensorFlow",
#     "samples": 3264,
#     "inference_time": float(infer_time),
#     "cka_time": float(cka_time)
# }
def save_metadata(metadata_dict, save_path):
    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata_dict, f, indent=4)

def save_model_state_pytorch(model, save_path):
    os.makedirs(save_path, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_path, "model_state.pt")
    )

def save_model_structure_pytorch(model, save_path):
    with open(os.path.join(save_path, "model_structure.txt"), "w") as f:
        for name, module in model.named_modules():
            f.write(f"{name}: {module.__class__.__name__}\n")

def save_model_structure_tf(model, save_path):
    with open(os.path.join(save_path, "model_structure.txt"), "w") as f:
        for layer in model.layers:
            f.write(f"{layer.name}: {layer.__class__.__name__}\n")