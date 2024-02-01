{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/22.11";
    # using version of nixpkgs where pytorch version is 2.0.1
    #nixpkgs.url = "github:nixos/nixpkgs/904f1e3235d78269b5365f2166179596cbdedd66";
  };

  nixConfig.bash-prompt = "\\e[35m\[nix-develop (\\h)\]\\e[34m\\w\\e[39m$ ";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;

      devShells.${system} = rec {
	sshell = pkgs.mkShell {
 	  buildInputs = [
	    pkgs.python3
	  ];
	};
        default = pkgs.mkShell {
          buildInputs = [
            pkgs.python3Packages.dill
            pkgs.python3Packages.matplotlib
            pkgs.python3Packages.numpy
            pkgs.python3Packages.pandas
            pkgs.python3Packages.scikit-learn
            pkgs.python3Packages.scikit-optimize
            pkgs.python3Packages.scipy
            pkgs.python3Packages.pytorch
          ];
        };
      };
    };
}
