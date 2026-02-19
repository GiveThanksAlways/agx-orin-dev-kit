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
# After rebuilding + rebooting (needed for incus-admin group), launch VMs:
#
#   incus version
#   incus launch images:ubuntu/noble/cloud ubuntu-vm --vm -c security.secureboot=false
#   incus launch images:alpine/3.23/cloud  alpine-vm --vm -c security.secureboot=false
#   incus launch images:fedora/43/cloud    fedora-vm --vm -c security.secureboot=false
#   incus list
#   incus shell ubuntu-vm
#
# No 'incus admin init' needed -- preseed handles initialization declaratively.
#
# Jetpack kernel note:
#   The Jetpack r36 kernel (5.15) has nf_tables but lacks nft_fib and nft_ct.
#   NixOS's built-in nftables firewall generates rules requiring both modules
#   ("fib saddr . mark . iif" and "ct state vmap"), so we disable the NixOS
#   firewall and let Incus manage its own "incus" nftables table directly.
#   See: https://wiki.nixos.org/wiki/Incus#Networking/Firewall

{ config, lib, ... }:

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
    # Incus requires nftables (the module asserts this).
    networking.nftables.enable = true;

    # Prevent Incus nftables table from being flushed on rule changes.
    # See: https://wiki.nixos.org/wiki/Incus#Networking/Firewall
    networking.nftables.flushRuleset = false;

    # Jetpack r36 kernel lacks nft_fib + nft_ct — NixOS firewall rules
    # that use "fib"/"ct state vmap" will fail. Disable the NixOS-managed
    # firewall; Incus manages its own nftables table for NAT/DHCP.
    # Safe on a dev-kit LAN; SSH is key-only.
    networking.firewall.enable = false;

    virtualisation.incus.enable = true;

    # Declarative initialization -- replaces 'incus admin init'.
    # Preseed is additive; it never removes resources.
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

    # Non-root Incus access. Requires reboot after first switch.
    users.groups.incus-admin.members = cfg.users;
  };
}
