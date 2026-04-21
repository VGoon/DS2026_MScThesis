import time
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_tf_mobilenet, preprocess_tf
from utils.save_results import save_cka, save_metadata, save_metrics, save_predictions, save_model_structure_tf

def create_resnet50():
    print("TF: CREATING ResNet50 MODEL.")
    return tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        weights="imagenet"
    )

def create_mobilenetv2():
    print("TF: CREATING MobileNetV2 MODEL.")
    return tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        weights='imagenet'
    )

def create_vgg16():
    print("TF: CREATING VGG16 MODEL.")
    return tf.keras.applications.VGG16(
        input_shape=(224, 224, 3),
        weights="imagenet"
    )

def get_data(path, preprocess_fn, max_samples, batch_size, model_type, debugging, own_preprocessing):
    batches = max_samples // batch_size
    print("TF: GETTING DATA.")
    print("Batches:", batches)
    print("Expected samples:", batches * batch_size)

    # get and prepare dataset
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(256, 256),
        batch_size=batch_size,
        shuffle=False
    )

    # shape B, H, W, C)
    dataset = dataset.map(lambda x, y: (preprocess_fn(x, own_preprocessing, model_type), y))

    if debugging == True:
        for img, label in dataset.take(1):
            print("TF input shape:", img.shape)
            print("TF input range:", tf.reduce_min(img), tf.reduce_max(img))

    dataset = dataset.take(batches)
    return dataset

def run_inference(model, data, activations, activation_model, layer_map, debugging, max_samples = 992):
    print("TF: RUNNING TENSORFLOW INFERENCE WITH " + str(max_samples) + " SAMPLES.")

    all_outputs, all_labels = [], []
    processing_start_time = time.time()

    for images, labels in data:
        layer_outputs = activation_model(images)  # list of tensors

        for (stage, _), act in zip(layer_map.items(), layer_outputs):
            if len(act.shape) == 4:
                act = tf.reduce_mean(act, axis=[1, 2])  # (B, H, W, C) → (B, C)
            elif len(act.shape) == 2:
                pass  # already (B, C)
            else:
                raise ValueError("Unexpected activation shape")
            activations[stage].append(act.numpy())

        preds = model(images)

        all_outputs.append(preds.numpy())
        all_labels.append(labels.numpy())

    processing_end_time = time.time()
    processing_time = processing_end_time - processing_start_time
    print(f"Processing time: {processing_time:.2f} seconds.")

    # collect inference activations and outputs
    for k in activations:
        activations[k] = np.concatenate(activations[k], axis=0)

    if debugging == True:
        for name, act in activations.items():
            print(f"TF {name} shape:", act.shape)

    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return activations, all_outputs, all_labels, processing_time

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_outputs_tensorflow(model, save_path, all_outputs, all_labels, metadata, debugging):
    print("TF: SAVING OUTPUTS.")

    if debugging == True:
        print("is softmax?")
        print(np.sum(all_outputs[0]))
    probs = all_outputs

    # SAVE MODEL STATE
    model.save(save_path + "/model.keras")
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

    metadata = {k: convert_numpy(v) for k, v in metadata.items()}
    save_metadata(metadata, save_path)

def register_activations_resnet(model):
    layer_map = {
        "stage1": "conv2_block3_out",
        "stage2": "conv3_block4_out",
        "stage3": "conv4_block6_out",
        "stage4": "conv5_block3_out",
        "stage5" : "avg_pool"
    }
    # layer_names = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    outputs = [model.get_layer(name).output for name in layer_map.values()]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = {k: [] for k in layer_map.keys()}
    return activations, activation_model, layer_map

def register_activations_mobilenet(model):
    layer_map = {
        "stage1": "block_2_add",
        "stage2": "block_5_add",
        "stage3": "block_12_add",
        "stage4": "block_15_add",
        "stage5": "out_relu"
    }

    outputs = [model.get_layer(name).output for name in layer_map.values()]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = {k: [] for k in layer_map.keys()}
    return activations, activation_model, layer_map

def register_activations_vgg(model):
    layer_map = {
        "stage1": "block1_pool",
        "stage2": "block2_pool",
        "stage3": "block3_pool",
        "stage4": "block4_pool",
        "stage5": "block5_pool"
    }

    outputs = [model.get_layer(name).output for name in layer_map.values()]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    activations = {k: [] for k in layer_map.keys()}
    return activations, activation_model, layer_map

def tf_run_mobilenetv2(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size = 32):
    print("RUNNING TF MOBILENETV2 PIPELINE.")
    if runConverted == False:
        model = create_mobilenetv2()
    else:
        model = None
    data = get_data(dataset_path, preprocess_tf_mobilenet, max_samples, batch_size, model_name, debugging, own_preprocessing)
    activations, activation_model, layer_map = register_activations_mobilenet(model)
    activations, all_outputs, all_labels, inference_time = run_inference(model, data, activations, activation_model, layer_map, debugging)
    total_params = model.count_params()
    trainable_params = np.sum([v.numpy().size for v in model.trainable_weights])

    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_tensorflow(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations

def tf_run_resnet50(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size = 32):
    print("RUNNING TF RESNET50 PIPELINE.")
    if runConverted == False:
        model = create_resnet50()
    else:
        model = None
    data = get_data(dataset_path, preprocess_tf, max_samples, batch_size, model_name, debugging, own_preprocessing)
    activations, activation_model, layer_map = register_activations_resnet(model)
    activations, all_outputs, all_labels, inference_time = run_inference(model, data, activations, activation_model, layer_map, debugging)
    total_params = model.count_params()
    trainable_params = np.sum([v.numpy().size for v in model.trainable_weights])

    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_tensorflow(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations

def tf_run_vgg16(save_path, model_name, max_samples, dataset_path, debugging, own_preprocessing, runConverted, batch_size = 32):
    print("RUNNING TF VGG16 PIPELINE.")
    if runConverted == False:
        model = create_vgg16()
    else:
        model = None
    data = get_data(dataset_path, preprocess_tf, max_samples, batch_size, model_name, debugging, own_preprocessing)
    activations, activation_model, layer_map = register_activations_vgg(model)
    activations, all_outputs, all_labels, inference_time = run_inference(model, data, activations, activation_model, layer_map, debugging)
    total_params = model.count_params()
    trainable_params = np.sum([v.numpy().size for v in model.trainable_weights])

    metadata = {
        "model": model_name,
        "framework": "PyTorch",
        "samples": max_samples,
        "inference_time": inference_time,
        "total_params": total_params,
        "trainable_params": trainable_params
    }

    save_outputs_tensorflow(model, save_path, all_outputs, all_labels, metadata, debugging)
    return activations