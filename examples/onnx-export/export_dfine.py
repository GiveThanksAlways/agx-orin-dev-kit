"""Export D-FINE models to ONNX."""
import sys, os, torch, torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "D-FINE"))
from src.core import YAMLConfig

def export_dfine(variant, config_path, weights_path, output_path):
    cfg = YAMLConfig(config_path)
    cfg.yaml_cfg["eval_spatial_size"] = [640, 640]
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(weights_path, map_location="cpu")
    state = checkpoint.get("ema", {}).get("module", checkpoint.get("model"))
    cfg.model.load_state_dict(state)
    print(f"D-FINE-{variant} loaded!")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    data = torch.rand(1, 3, 640, 640)
    size = torch.tensor([[640, 640]])
    _ = model(data, size)

    torch.onnx.export(
        model, (data, size), output_path,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes={"images": {0: "N"}, "orig_target_sizes": {0: "N"}},
        opset_version=17, verbose=False, do_constant_folding=True,
        dynamo=False,
    )
    import onnx
    onnx.checker.check_model(onnx.load(output_path))
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"D-FINE-{variant} ONNX exported: {output_path} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    variant = sys.argv[1]  # n or s
    base = os.path.dirname(__file__)
    config = os.path.join(base, f"D-FINE/configs/dfine/dfine_hgnetv2_{variant}_coco.yml")
    weights = os.path.join(base, f"dfine_{variant}_coco.pth")
    output = os.path.join(base, f"dfine_{variant}_coco.onnx")
    export_dfine(variant.upper(), config, weights, output)
