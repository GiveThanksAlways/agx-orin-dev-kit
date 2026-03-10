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
        #
        # ultralytics is NOT included in the nix pythonEnv: its nixpkgs derivation
        # transitively pulls opencv, torchvision, magma, and onnxruntime (all source
        # builds). Instead we pip install --no-deps in the shellHook — ultralytics
        # is a pure-Python wheel (~3 MB, no compilation).
        pythonEnv = pkgs.python312.withPackages (ps: [
          ps.torch   # reuses the already-built CUDA torch from control-loop
          ps.onnx    # pure Python, no build
        ]);
      in {
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv pkgs.python312Packages.pip ];
          shellHook = ''
            # Install ultralytics + its lightweight pure-Python deps into a local venv.
            # We skip onnxruntime (not needed for export) and cv2/torchvision
            # (not needed for ONNX export path).
            VENV="$PWD/.venv-export"
            if [ ! -d "$VENV" ]; then
              python3 -m venv --system-site-packages "$VENV"
              "$VENV/bin/pip" install --quiet \
                "ultralytics==8.3.221" \
                --no-deps
              # ultralytics hard-deps that ARE pure Python and not already in nix env:
              "$VENV/bin/pip" install --quiet \
                matplotlib pillow pyyaml requests scipy tqdm psutil py-cpuinfo \
                pandas seaborn lap ultralytics-thop \
                --no-deps 2>/dev/null || true
              # cv2 is imported unconditionally by ultralytics; use headless wheel (no GUI deps)
              "$VENV/bin/pip" install --quiet opencv-python-headless 2>/dev/null || true
              # ultralytics checks torchvision version at import time; install pre-built wheel
              # (--no-deps so it doesn't re-download torch)
              "$VENV/bin/pip" install --quiet torchvision==0.24.1 --no-deps 2>/dev/null || true
              # onnxslim is used post-export to simplify/optimize the ONNX graph
              "$VENV/bin/pip" install --quiet onnxslim 2>/dev/null || true
            fi
            export PYTHONPATH="$VENV/lib/python3.12/site-packages:$PYTHONPATH"

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
