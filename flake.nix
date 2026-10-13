{
  description = "gdi2m - real-time guitar DI to MIDI (console inference tool)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      packages = forAllSystems (pkgs: rec {
        gdi2m = pkgs.callPackage ./nix/package.nix {
          src = self;
          version = "0.1.0-unstable-${self.lastModifiedDate or "dirty"}";
        };
        default = gdi2m;
      });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            ninja
            pkg-config
            onnxruntime
            alsa-lib
          ];
        };
      });
    };
}
