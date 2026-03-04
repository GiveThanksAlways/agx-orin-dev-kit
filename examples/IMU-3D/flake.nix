{
  description = "IMU-3D: MPU-9250 → Three.js real-time 3D visualization";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";

  outputs = { self, nixpkgs, ... }:
    let
      eachSystem = nixpkgs.lib.genAttrs [ "aarch64-linux" "x86_64-linux" ];
    in {
      devShells = eachSystem (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (ps: with ps; [ aiohttp smbus2 ]))
              pkgs.i2c-tools
            ];
            shellHook = ''
              echo "IMU-3D dev shell ready"
              echo "  python server.py          — live IMU on I2C bus 7"
              echo "  python server.py --mock    — simulated IMU for testing"
              echo ""
              echo "From your laptop:"
              echo "  ssh -L 9090:localhost:9090 Orin-AGX-NixOS"
              echo "  open http://localhost:9090"
            '';
          };
        }
      );
    };
}
