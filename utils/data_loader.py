import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_visuals(run_path, model_name):
    base = os.path.join(run_path, model_name)
    data = load_run(base)

    top1_conf_py = data['predictions']['top1_conf']['py']
    plot_confidence_distribution(base, top1_conf_py, model_name, "Pytorch")
    top1_conf_tf = data['predictions']['top1_conf']['tf']
    plot_confidence_distribution(base, top1_conf_tf, model_name, "Tensorflow")

    top1_py = data['predictions']['top1']['py']
    top1_tf = data['predictions']['top1']['tf']
    plot_prediction_agreement(top1_py, top1_tf, model_name)

    plot_accuracy(base, data["metrics"]["py"], model_name, "Pytorch")
    plot_accuracy(base, data["metrics"]["tf"], model_name, "Tensorflow")

    labels = data['labels']
    plot_confidence_correct_vs_wrong(base, top1_py, top1_conf_py, labels['py'], model_name, "Pytorch")
    plot_confidence_correct_vs_wrong(base, top1_tf, top1_conf_tf, labels['tf'], model_name, "Tensorflow")

    top5_conf_py = data['predictions']['top5_conf']['py']
    plot_topk_confidence(base, top5_conf_py, model_name, "Pytorch")

    top5_conf_tf = data['predictions']['top5_conf']['tf']
    plot_topk_confidence(base, top5_conf_tf, model_name, "Tensorflow")

def load_run(base):
    base_py = os.path.join(base, "pytorch/")
    base_tf = os.path.join(base, "tensorflow/")

    data = {
        "cka": load_cka(base),
        "metrics": load_metrics(base_py, base_tf),
        "metadata": load_metadata(base_py, base_tf),
        "predictions": load_predictions(base_py, base_tf),
        "labels": load_labels(base_py, base_tf)
    }
    return data

def load_cka(base):
    cka_path = os.path.join(base, "cka.npy")
    cka = np.load(cka_path)
    return cka

def load_metrics(base_py, base_tf):
    file_name = "metrics.json"
    metrics = {}
    with open(base_py + file_name, "r") as f:
        metrics["py"] = json.load(f)
    with open(base_tf + file_name, "r") as f:
        metrics["tf"] = json.load(f)
    return metrics

def load_metadata(base_py, base_tf):
    file_name = "metadata.json"
    metadata = {}
    with open(base_py + file_name, "r") as f:
        metadata["py"] = json.load(f)

    with open(base_tf + file_name, "r") as f:
        metadata["tf"] = json.load(f)
    return metadata

def load_predictions(path_py, path_tf):
    names = {
        "top1" : "top1.npy",
        "top5" : "top5.npy",
        "top10" : "top10.npy",
        "top1_conf" : "top1_conf.npy",
        "top5_conf" : "top5_conf.npy",
        "top10_conf" : "top10_conf.npy"
    }

    for x in names:
        # print(names[x])
        file_py = os.path.join(path_py, names[x])
        file_tf = os.path.join(path_tf, names[x])
        y = {
            "py": np.load(file_py),
            "tf": np.load(file_tf)
        }
        names[x] = y
    return names

def load_labels(path_py, path_tf):
    labels = {}
    file_py = os.path.join(path_py, 'labels.npy')
    file_tf = os.path.join(path_tf, 'labels.npy')
    labels['py'] = np.load(file_py)
    labels['tf'] = np.load(file_tf)
    return labels

def plot_accuracy(save_path, metrics, model_name, framework):
    labels = ["Top-1", "Top-5", "Top-10"]
    values = [
        metrics["top1"],
        metrics["top5"],
        metrics["top10"]
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.title(f"{model_name} Accuracy {framework}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(save_path, "Accuracy_"+framework+".png"), dpi=300)
    plt.close()

def plot_confidence_distribution(save_path, top1_conf, model_name, framework):
    plt.figure()
    plt.hist(top1_conf, bins=50)
    plt.title(f"{model_name} Top-1 Confidence Distribution {framework}")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, "Confidence_Distribution_"+framework+".png"), dpi=300)
    plt.close()
    
def plot_confidence_correct_vs_wrong(save_path, top1, top1_conf, labels, model_name, framework):
    correct = top1 == labels

    plt.figure()
    plt.hist(top1_conf[correct], bins=50, alpha=0.6, label="Correct")
    plt.hist(top1_conf[~correct], bins=50, alpha=0.6, label="Incorrect")
    plt.legend()
    plt.title(f"{model_name} Confidence: Correct vs Incorrect {framework}")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(save_path, "Confidence_Correct_vs_Incorrect_"+framework+".png"), dpi=300)
    plt.close()

def plot_topk_confidence(save_path, top5_conf, model_name, framework):
    mean_conf = np.mean(top5_conf, axis=0)

    plt.figure()
    plt.plot(range(1, 6), mean_conf, marker='o')
    plt.title(f"{model_name} Top-5 Confidence Trend {framework}")
    plt.xlabel("Rank")
    plt.ylabel("Confidence")
    plt.savefig(os.path.join(save_path, "Top-5_Confidence_Trend_"+framework+".png"), dpi=300)
    plt.close()

def plot_prediction_agreement(pt_top1, tf_top1, model_name):
    agreement = np.mean(pt_top1 == tf_top1)
    print(f"{model_name} Prediction agreement: {agreement:.4f}")