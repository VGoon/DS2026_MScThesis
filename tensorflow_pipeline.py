import time
import numpy as np
import tensorflow as tf
from utils.cka_formating import compute_cka_matrix_tensorflow
from utils.save_results import save_cka, save_metadata, save_metrics, save_predictions, save_model_structure_tf

def create_tensorflow_model():
    print("TF: CREATING MODEL.")
    return tf.keras.applications.ResNet50(weights="imagenet")

def get_data_tf(path, preprocess_fn, max_samples, batch_size):
    print("TF: GETTING DATA.")
    batches = max_samples // batch_size

    print("Batches:", batches)
    print("Expected samples:", batches * batch_size)

    # get and prepare dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False
    )
    dataset = dataset.map(lambda x, y: (preprocess_fn(x), y))
    dataset = dataset.take(batches)
    return dataset

def run_inference_tensorflow(model, data, max_samples = 992):
    print("TF: RUNNING TENSORFLOW INFERENCE WITH " + str(max_samples) + " SAMPLES.")
    is_keras = isinstance(model, tf.keras.Model)

    # handle activation grabbing
    if is_keras:
        tf_layer_map = {
            "stage1": "conv2_block3_out",
            "stage2": "conv3_block4_out",
            "stage3": "conv4_block6_out",
            "stage4": "conv5_block3_out",
        }
        # layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
        outputs = [model.get_layer(name).output for name in tf_layer_map.values()]
        activation_model = activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
        activations = {k: [] for k in tf_layer_map.keys()}
    else:
        activation_model = None
        activations = None

    # run inference
    all_outputs, all_labels = [], []
    processing_start_time = time.time()

    for images, labels in data:
        if activation_model is not None:
            layer_outputs = activation_model(images)  # list of tensors

            for (stage, _), act in zip(tf_layer_map.items(), layer_outputs):
                activations[stage].append(act.numpy())

        if is_keras:
            preds = model(images)
        else:
            infer = model.signatures["serving_default"]
            preds = list(infer(**{
                list(infer.structured_input_signature[1].keys())[0]: images
            }).values())[0]

        all_outputs.append(preds.numpy())
        all_labels.append(labels.numpy())

    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time
    print(f"Processing time: {processing_time:.2f} seconds.")

    # collect inference activations and outputs
    if activations is not None:
        for k in activations:
            activations[k] = np.concatenate(activations[k], axis=0)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return activations, all_outputs, all_labels, processing_time

def run_cka_tensorflow(activations):
    print("TF: RUNNING CKA.")
    cka_start_time = time.time()
    cka_matrix, layer_names = compute_cka_matrix_tensorflow(activations)
    cka_end_time = time.time()
    cka_duration = cka_end_time - cka_start_time
    print(f"CKA time: {cka_duration:.2f} seconds.")

    return cka_matrix, layer_names, cka_duration

def save_outputs_tensorflow(model, save_path, all_outputs, all_labels, cka_matrix, 
                            layer_names, version_name, metadata):
    print("TF: SAVING OUTPUTS.")
    probs = all_outputs

    # SAVE MODEL STATE
    # XXX
    # SAVE MODEL STRUCTURE
    save_model_structure_tf(model, save_path)

    # ACCURACIES
    top1 = np.argmax(probs, axis=1)
    top5 = np.argpartition(probs, -5, axis=1)[:, -5:]
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