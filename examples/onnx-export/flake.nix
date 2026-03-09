{
  description = "ONNX model export shell — ultralytics (YOLOv8) on Jetson AGX Orin";

  inputs = {
    # Follow control-loop's nixpkgs so torch (python312, CUDA, built from source)
    # is already in the local nix store — zero rebuild time.
    control-loop.url = "path:../control-loop";
    nixpkgs.follows = "control-loop/nixpkgs";
    jetpack-nixos.follows = "control-loop/jetpack-nixos";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, jetpack-nixos, flake-utils, control-loop }:
    flake-utils.lib.eachSystem [ "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            # Must match control-loop's config exactly so torch's derivation hash
            # is identical — Nix reuses the cached build instead of rebuilding.
            cudaSupport = true;
            cudaCapabilities = [ "8.7" ];
            cudaForwardCompat = false;
          };
          overlays = [
            jetpack-nixos.overlays.default
            (final: _: { inherit (final.nvidia-jetpack6) cudaPackages; })
          ];
        };

        # python312 to match control-loop (same torch derivation hash = cached).
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.torch         # reuses the already-built CUDA torch from control-loop
          ps.ultralytics
          ps.onnx
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv ];
          shellHook = ''
            echo ""
            echo "=== ONNX Export Shell ==="
            echo "  Export YOLOv8 variants:"
            echo "    python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640)\""
            echo "    python3 -c \"from ultralytics import YOLO; YOLO('yolov8s.pt').export(format='onnx', imgsz=640)\""
            echo "    python3 -c \"from ultralytics import YOLO; YOLO('yolov8m.pt').export(format='onnx', imgsz=640)\""
            echo ""
          '';
        };
      }
    );
}
