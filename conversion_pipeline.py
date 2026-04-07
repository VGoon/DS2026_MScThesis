# import torch
# # import onnx
# import tensorflow as tf
# import os
# # from onnx2tf import convert

# # def preprocess_pt(image):
# #     x = preprocess_base(image)
# #     return torch.tensor(x).permute(2, 0, 1)  # HWC → CHW


# # def preprocess_tf(image):
# #     x = preprocess_base(image)
# #     return tf.constant(x)  # stays HWC



# def convert_pt_to_tf(model, save_dir):
#     os.makedirs(save_dir, exist_ok=True)

#     model.eval()

#     # 1. Create dummy input (IMPORTANT: match your pipeline)
#     dummy_input = torch.randn(1, 3, 224, 224)

#     onnx_path = os.path.join(save_dir, "model.onnx")
#     tf_path = os.path.join(save_dir, "tf_model")

#     # 2. Export to ONNX
#     torch.onnx.export(
#         model,
#         dummy_input,
#         onnx_path,
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={
#             "input": {0: "batch_size"},
#             "output": {0: "batch_size"}
#         },

#         opset_version=11
#     )

#     # 3. Convert ONNX → TensorFlow (NEW WAY)
#     convert(
#         input_onnx_file_path=onnx_path,
#         output_folder_path=tf_path
#     )

#     # 4. Load TensorFlow model
#     tf_model = tf.saved_model.load(tf_path)

#     return tf_model

# def convert_tf_to_py(model):

#     return None