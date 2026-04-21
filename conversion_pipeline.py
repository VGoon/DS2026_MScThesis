import os
import torch
import onnx
import tf2onnx
import tensorflow as tf
# from pytorch_pipeline import create_mobilenetv2, create_resnet50, create_vgg16
from onnx2tf import convert
from tensorflow_pipeline import create_mobilenetv2, create_resnet50, create_vgg16
from onnx2pytorch import ConvertModel


# def preprocess_pt(image):
#     x = preprocess_base(image)
#     return torch.tensor(x).permute(2, 0, 1)  # HWC → CHW


# def preprocess_tf(image):
#     x = preprocess_base(image)
#     return tf.constant(x)  # stays HWC

def export_pytorch_to_onnx(save_dir="/Converted_Models", onnx_name="model.onnx"):
  print("EXPORTING PYTORCH TO ONNX.")
  os.makedirs(save_dir, exist_ok=True)
  py_mobilenet = create_mobilenetv2();
  py_resnet = create_resnet50();
  py_vgg = create_vgg16();

  dummy_input = torch.randn(1, 3, 224, 224)

  torch.onnx.export(
      py_mobilenet,
      dummy_input,
      ("PY_Mobilenet_" + onnx_name),
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={
          "input": {0: "batch_size"},
          "output": {0: "batch_size"}
      },
      opset_version=12
  )

  torch.onnx.export(
      py_resnet,
      dummy_input,
      ("PY_Resnet_" + onnx_name),
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={
          "input": {0: "batch_size"},
          "output": {0: "batch_size"}
      },
      opset_version=12
  )

  torch.onnx.export(
      py_vgg,
      dummy_input,
      ("PY_VGG_" + onnx_name),
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={
          "input": {0: "batch_size"},
          "output": {0: "batch_size"}
      },
      opset_version=12
  )

  print("Exported to ONNX")

def convert_onnx_to_tf(onnx_name = "model.onnx", converted_name="converted_tf"):
  print("CONVERTING FROM ONNX TO TF.")
  convert(
    input_onnx_file_path=onnx_name,
    output_folder_path="/content/DS2026_MScThesis/Converted_Models/" + converted_name
  )

def export_tensorflow_to_onnx(save_dir="/content/DS2026_MScThesis/Converted_Models_TF/", onnx_name="model.onnx"):
  tf_mobile = create_mobilenetv2()
  tf_resnet = create_resnet50()
  # tf_vgg = create_vgg16()

  spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

  mobile_model_proto, _ = tf2onnx.convert.from_keras(tf_mobile, input_signature=spec, opset=12)
  resnet_model_proto, _ = tf2onnx.convert.from_keras(tf_resnet, input_signature=spec, opset=12)
  # vgg_model_proto, _ = tf2onnx.convert.from_keras(tf_vgg, input_signature=spec, opset=12)

  with open((save_dir + "TF_Mobilenet_" + onnx_name), "wb") as f:
      f.write(mobile_model_proto.SerializeToString())

  with open((save_dir + "TF_Resnet_" + onnx_name), "wb") as f:
      f.write(resnet_model_proto.SerializeToString())

  # with open(( save_dir + "TF_VGG_" + onnx_name), "wb") as f:
      # f.write(vgg_model_proto.SerializeToString())

def convert_onnx_to_py(save_dir="/Converted_Models", name="model.pth"):
  onnx_model = onnx.load(save_dir)
  pytorch_model = ConvertModel(onnx_model)
  torch.save(pytorch_model, name)