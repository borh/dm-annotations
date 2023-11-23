{ pkgs, lib, ... }:
let
  cuda = false;
  rocm = false;

  cudaVersion = "12.2";
  cuda-common-redist = with pkgs.cudaPackages_12_2; [
    cuda_cccl # <thrust/*>
    libcublas # cublas_v2.h
    libcurand
    libcusolver # cusolverDn.h
    libcusparse # cusparse.h
    libcufft
    cuda_cudart # cuda_runtime.h cuda_runtime_api.h
    cuda_nvcc
    cuda_nvtx
    cuda_cupti
    cuda_nvrtc
    libcusolver
    nccl
    pkgs.linuxPackages_latest.nvidia_x11
    pkgs.cudaPackages.cudnn
  ];
  cuda-native-redist = pkgs.symlinkJoin {
    name = "cuda-native-redist-${cudaVersion}";
    paths = cuda-common-redist;
  };

  rocm-all = pkgs.symlinkJoin {
    name = "rocm-all-meta";
    paths = with pkgs.rocmPackages.meta; [
      rocm-developer-tools
      rocm-ml-sdk
      rocm-ml-libraries
      rocm-hip-sdk
      rocm-hip-libraries
      rocm-openmp-sdk
      rocm-opencl-sdk
      rocm-opencl-runtime
      rocm-hip-runtime-devel
      rocm-hip-runtime
      rocm-language-runtime
    ];
  };

  # For rocm: https://github.com/NixOS/nixpkgs/pull/260299 
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
  ] ++ lib.optionals rocm [
    rocm-all
    # pkgs.rocmPackages.meta.rocm-hip-runtime
    # pkgs.rocmPackages.meta.rocm-hip-sdk
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

  env.PRODIGY_HOME = "./prodigy";
  env.LD_LIBRARY_PATH = lib.mkIf pkgs.stdenv.isLinux (
    lib.makeLibraryPath (with pkgs; [
      zlib
      gcc-unwrapped.lib
      stdenv.cc.libc
    ] ++ lib.optionals (!cuda) [

    ] ++ lib.optionals rocm [
      libdrm
      rocm-all
      # rocmPackages.meta.rocm-ml-sdk
      # rocmPackages.meta.rocm-developer-tools
      # rocmPackages.meta.rocm-hip-libraries
    ] ++ lib.optionals cuda [
      cuda-native-redist
    ])
  );

  scripts.check-gpu.exec = "python -c \"import spacy;print(spacy.prefer_gpu())\"";

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # https://devenv.sh/processes/
  # processes.ping.exec = "ping example.com";

  # See full reference at https://devenv.sh/reference/options/
}
  (if cuda then
    {
      env.CUDA_HOME = "${cuda-native-redist}";
      env.CUDA_PATH = "${cuda-native-redist}";
    } else if rocm then {
    env.CUPY_INSTALL_USE_HIP = 1;
    env.ROCM_HOME = "${rocm-all}";
  }
  else { })
