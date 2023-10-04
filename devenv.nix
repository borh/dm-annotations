{ pkgs, lib, ... }:
let
  cuda = false;
  rocm = false;
in
lib.attrsets.recursiveUpdate
{
  # https://devenv.sh/basics/
  # https://devenv.sh/packages/
  packages = [
    pkgs.zlib
    pkgs.gcc-unwrapped.lib
    pkgs.cmake
    pkgs.nodejs
    pkgs.yarn
    pkgs.graphviz
  ] ++ lib.optionals (!cuda) [
    pkgs.stdenv
    pkgs.stdenv.cc
  ] ++ lib.optionals cuda [
    pkgs.cudaPackages.backendStdenv
    pkgs.cudaPackages.backendStdenv.cc
  ];
  # https://devenv.sh/scripts/
  enterShell = ''
    poetry run python -m ipykernel install --user --name=devenv-$(basename $PWD)
    export LICENSE=$(cat /var/run/agenix/prodigy)
    export OPENAI_API_KEY=$(cat /var/run/agenix/openai-api)
    # Integrating into pyproject.toml:
    # pip wheel prodigy -f https://$LICENSE@download.prodi.gy --wheel-dir wheels
    # pip install wheels/prodigy-1.12.5-cp311-cp311-linux_x86_64.whl
    # Direct install:
    # pip install prodigy -f https://$LICENSE@download.prodi.gy
  '';

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    package = pkgs.python311.withPackages (ps: with ps; [
      pygraphviz
    ]);
    poetry = {
      enable = true;
      activate.enable = true;
    };
  };

  env.LD_LIBRARY_PATH = lib.mkIf pkgs.stdenv.isLinux (
    lib.makeLibraryPath (with pkgs; [
      zlib
      gcc-unwrapped.lib
      stdenv.cc.libc
    ] ++ lib.optionals (!cuda) [

    ] ++ lib.optionals rocm [
      libdrm
    ] ++ lib.optionals cuda [
      # This would mess with normal nixpkgs packages:
      # cudaPackages.backendStdenv.cc.cc.lib
      # cudaPackages.backendStdenv.cc
      linuxPackages_latest.nvidia_x11
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      cudaPackages.libcublas
      cudaPackages.libcurand
      cudaPackages.libcufft
      cudaPackages.libcusparse
      cudaPackages.cuda_nvtx
      cudaPackages.cuda_cupti
      cudaPackages.cuda_nvrtc
      cudaPackages.nccl
    ])
  );

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
  (if cuda then
    {
      env.CUDA_HOME = "${pkgs.cudaPackages.cudatoolkit}";
      env.CUDA_PATH = "${pkgs.cudaPackages.cudatoolkit}";
    } else { })
