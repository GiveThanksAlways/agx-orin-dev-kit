# incus.nix -- Incus VM/container hypervisor for Jetson Orin AGX
#
# Composable NixOS module. Import and enable:
#
#   imports = [ ./modules/incus.nix ];
#   services.orin-incus = {
#     enable = true;
#     users = [ "agent" "spencer" ];
#   };
#
# After rebuilding, reboot (required for incus-admin group to take effect), then:
#
#   incus version
#   incus launch images:ubuntu/noble/cloud ubuntu-vm --vm -c security.secureboot=false
#   incus launch images:alpine/3.23/cloud  alpine-vm --vm -c security.secureboot=false
#   incus launch images:fedora/43/cloud    fedora-vm --vm -c security.secureboot=false
#   incus list
#   incus shell ubuntu-vm
#
# No 'incus admin init' needed -- preseed handles initialization declaratively.

{ config, lib, pkgs, ... }:

let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.orin-incus;
in
{
  options.services.orin-incus = {
    enable = mkEnableOption "Incus VM/container hypervisor";

    users = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Users to add to the incus-admin group (non-root Incus access).";
      example = [ "agent" "spencer" ];
    };

    bridgeName = mkOption {
      type = types.str;
      default = "incusbr0";
      description = "Name of the Incus bridge interface.";
    };

    bridgeAddress = mkOption {
      type = types.str;
      default = "10.0.100.1/24";
      description = "IPv4 address/prefix for the Incus bridge.";
    };

    storageSize = mkOption {
      type = types.str;
      default = "35GiB";
      description = "Root disk size for instances in the default profile.";
    };

    storagePath = mkOption {
      type = types.str;
      default = "/var/lib/incus/storage-pools/default";
      description = "Host path for the directory-backed storage pool.";
    };
  };

  config = mkIf cfg.enable {
    # Incus requires nftables; iptables will fail eval.
    networking.nftables.enable = true;

    # Trust the Incus bridge so DHCP/DNS reach instances.
    networking.firewall.trustedInterfaces = [ cfg.bridgeName ];

    # Enable the Incus daemon.
    virtualisation.incus.enable = true;

    # Declarative initialization -- replaces 'incus admin init'.
    # Note: preseed never removes resources; changes are additive.
    virtualisation.incus.preseed = {
      networks = [
        {
          name = cfg.bridgeName;
          type = "bridge";
          config = {
            "ipv4.address" = cfg.bridgeAddress;
            "ipv4.nat" = "true";
          };
        }
      ];
      storage_pools = [
        {
          name = "default";
          driver = "dir";
          config.source = cfg.storagePath;
        }
      ];
      profiles = [
        {
          name = "default";
          devices = {
            eth0 = {
              name = "eth0";
              network = cfg.bridgeName;
              type = "nic";
            };
            root = {
              path = "/";
              pool = "default";
              size = cfg.storageSize;
              type = "disk";
            };
          };
        }
      ];
    };

    # Add requested users to incus-admin for non-root access.
    # Requires a reboot to take effect after first switch.
    users.groups.incus-admin.members = cfg.users;
  };
}
