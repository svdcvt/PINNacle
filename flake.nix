{
  description = "Flake: Python 3.9 environment with Pytorch 1.11.0 (sshell supports CUDA 11.8)";

  inputs = {
    # using version of nixpkgs where pytorch version is 2.0 (mix 23.05)
    #nixpkgs.url = "github:nixos/nixpkgs/904f1e3235d78269b5365f2166179596cbdedd66";
    # using older version because of scikit
    nixpkgs.url = "github:nixos/nixpkgs/22.05";
  };

  nixConfig.bash-prompt = "\\e[35m\[nix-develop (\\h)\]\\e[34m\\w\\e[39m$ ";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        allowUnfree = true;
        cudaSupport = true;
    };
    in
    {
      formatter.${system} = pkgs.nixpkgs-fmt;

      devShells.${system} = rec {
        sshell = pkgs.mkShell {
          xlibPath = with pkgs.xorg; pkgs.lib.makeLibraryPath [ libX11 libXdmcp libXau libXext libxcb ];
          libPath = with pkgs ; pkgs.lib.makeLibraryPath [ 
                          stdenv.cc.cc cudaPackages_11.cudatoolkit
                          libtiff openssl zlib libbsd expat glib libjpeg libpng
                          libffi libvorbis gnutls libdrm gmp numactl elfutils
	                  libelf libogg p11-kit libtasn1 nettle e2fsprogs libgcrypt
                          keyutils bzip2 libgpg-error xz libllvm
                        ];
          buildInputs = [
           ( pkgs.python3.withPackages(ps: with ps; [
                dill
                scikit-optimize
                matplotlib
                numpy
                pandas
                scikit-learn
                scipy
                pytorch-bin
          ]))
         ] ++ [pkgs.lsb-release];
         shellHook= ''
           export LD_LIBRARY_PATH=${sshell.libPath}:${sshell.xlibPath}:${pkgs.cudaPackages_11.cudatoolkit}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
         '';  
        };

        dahushell = pkgs.mkShell {
          buildInputs = [
           ( pkgs.python3.withPackages(ps: with ps; [
                dill
                scikit-optimize
                matplotlib
                numpy
                pandas
                scikit-learn
                scipy
                pytorch
          ]))
         ];
        };
      };
    };
}
