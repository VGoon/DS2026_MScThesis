# import torch
# # import onnx
# import tensorflow as tf
# import os
# from pytorch_pipeline import create_pytorch_model
# from onnx2tf import convert

# # def preprocess_pt(image):
# #     x = preprocess_base(image)
# #     return torch.tensor(x).permute(2, 0, 1)  # HWC → CHW


# # def preprocess_tf(image):
# #     x = preprocess_base(image)
# #     return tf.constant(x)  # stays HWC

# def export_pytorch_to_onnx(save_dir, onnx_name="model.onnx"):
#   print("EXPORTING PYTORCH TO ONNX.")
#   os.makedirs(save_dir, exist_ok=True)
#   pt_model = create_pytorch_model()
#   pt_model.eval()

#   dummy_input = torch.randn(1, 3, 224, 224)

#   torch.onnx.export(
#       pt_model,
#       dummy_input,
#       onnx_name,
#       input_names=["input"],
#       output_names=["output"],
#       dynamic_axes={
#           "input": {0: "batch_size"},
#           "output": {0: "batch_size"}
#       },
#       opset_version=12
#   )

#   print("Exported to ONNX")

# def convert_onnx_to_tf(onnx_name = "model.onnx", converted_name="converted_tf"):
#   print("CONVERTING FROM ONNX TO TF.")
#   convert(
#     input_onnx_file_path=onnx_name,
#     output_folder_path=converted_name
#   )

# def load_converted_tf(converted_name="converted_tf"):
#   print("LOADING CONVERTED TF MODEL.")
#   return tf.saved_model.load(converted_name)

# # def convert_pt_to_tf(save_dir):
# #     os.makedirs(save_dir, exist_ok=True)


# #     # 3. Convert ONNX → TensorFlow (NEW WAY)
# #     convert(
# #         input_onnx_file_path=onnx_path,
# #         output_folder_path=tf_path
# #     )

# #     # 4. Load TensorFlow model
# #     tf_model = tf.saved_model.load(tf_path)

# #     return tf_model

# def load_converted_tf(path):
#     model = tf.saved_model.load(path)
#     return model

# def convert_tf_to_py(model):

#     return None