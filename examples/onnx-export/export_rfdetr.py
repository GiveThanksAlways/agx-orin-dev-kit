"""Export RF-DETR Base to ONNX."""
import os
from rfdetr import RFDETRBase

model = RFDETRBase()  # auto-downloads COCO checkpoint
output_dir = os.path.dirname(__file__)
model.export(output_dir=output_dir, simplify=False)
fpath = os.path.join(output_dir, "inference_model.onnx")
size_mb = os.path.getsize(fpath) / 1024 / 1024
print(f"RF-DETR-Base ONNX exported: {fpath} ({size_mb:.1f} MB)")
