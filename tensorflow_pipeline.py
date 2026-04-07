import time
import numpy as np
import tensorflow as tf
from utils.cka_formating import compute_cka_matrix_tensorflow
from utils.save_results import save_cka, save_metadata, save_metrics, save_predictions, save_model_structure_tf

def run_tensorflow(save_path, max_samples=992):
    print("RUNNING TENSORFLOW PIPELINE WITH " + str(max_samples) + " SAMPLES.")
    batch_size = 32
    batches = max_samples // batch_size

    print("Batches:", batches)
    print("Expected samples:", batches * batch_size)

    # download model
    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocess = tf.keras.applications.resnet.preprocess_input

    # get and prepare dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        "ImageNet/",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    )
    dataset = dataset.map(lambda x, y: (preprocess(x), y))
    dataset = dataset.take(batches)

    # handle activation grabbing
    layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = {name: [] for name in layer_names}

    # run inference
    all_outputs = []
    all_labels = []
    processing_start_time = time.time()
    for images, labels in dataset:
        layer_outputs = activation_model(images)  # list of tensors

        for name, act in zip(layer_names, layer_outputs):
            activations[name].append(act.numpy())

        preds = model(images)
        all_outputs.append(preds.numpy())
        all_labels.append(labels.numpy())
    processing_end_time = time.time()
    print(f"Processing time: {processing_end_time - processing_start_time:.2f} seconds.")

    # collect inference activations and outputs
    for k in activations:
        activations[k] = np.concatenate(activations[k], axis=0)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    probs = all_outputs

    # SAVE MODEL STRUCTURE
    save_model_structure_tf(model, save_path)

    # ACCURACIES
    top1 = np.argmax(probs, axis=1)
    top5 = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    top10 = np.argsort(probs, axis=1)[:, -10:][:, ::-1]

    top1_acc = np.mean(top1 == all_labels)
    print(f"Top-1 Accuracy: {top1_acc:.4f}")

    top5_acc = np.mean([
        all_labels[i] in top5[i]
        for i in range(len(all_labels))
    ])
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    top10_acc = np.mean([
        all_labels[i] in top10[i]
        for i in range(len(all_labels))
    ])
    print(f"Top-10 Accuracy: {top10_acc:.4f}")

    metrics = {
        "top1": float(top1_acc),
        "top5": float(top5_acc),
        "top10": float(top10_acc)
    }
    save_metrics(metrics, save_path)

    # CONFIDENCE
    top1_conf = probs[np.arange(len(probs)), top1]
    top5_conf = np.take_along_axis(probs, top5, axis=1)
    top10_conf = np.take_along_axis(probs, top10, axis=1)

    # PREDICTIONS
    # categories = tf.keras.applications.imagenet_utils.decode_predictions(
    #     np.eye(1000), top=1
    # )
    # just to extract labels list
    # categories = [c[0][1] for c in categories]

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

    # TOTAL PARAMS
    total_params = model.count_params()
    print(f"Total parameters: {total_params:,}")

    # TOTAL TRAINABLE PARAMS
    trainable_params = np.sum([
        np.prod(v.shape) for v in model.trainable_weights
    ])
    print(f"Trainable parameters: {trainable_params:,}")

    # run CKA
    cka_start_time = time.time()
    cka_matrix, layer_names = compute_cka_matrix_tensorflow(activations)
    cka_end_time = time.time()
    print(f"CKA time: {cka_end_time - cka_start_time:.2f} seconds.")
    save_cka(cka_matrix, layer_names, save_path, "TensorFlow")

    metadata = {
        "model": "ResNet18",
        "framework": "TensorFlow",
        "samples": int(max_samples),
        "inference_time": float(processing_end_time - processing_start_time),
        "cka_time": float(cka_end_time - cka_start_time),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params)
    }
    save_metadata(metadata, save_path)