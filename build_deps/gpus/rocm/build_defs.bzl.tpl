# Macros for building HIP code for AMD GPUs. They keep the names of the CUDA
# macros so that the same BUILD files build with either toolkit.

def rocm_amdgpu_targets():
    """Returns the list of AMD GPU targets to compile for."""
    return %{amdgpu_targets}

def rocm_default_copts():
    """Default options for all HIP compilations."""
    return [
        "-x",
        "hip",
        "--rocm-path=%{rocm_path}",
        "-isystem",
        "%{rocm_path}/include/hipccl",
        "-DUSE_HIP=1",
    ] + ["--offload-arch=" + target for target in rocm_amdgpu_targets()]

def cuda_header_library(name,
                        hdrs,
                        include_prefix = None,
                        strip_include_prefix = None,
                        deps = [],
                        **kwargs):
    """Header-only library with both virtual and system include paths."""
    native.cc_library(
        name = name + "_virtual",
        hdrs = hdrs,
        include_prefix = include_prefix,
        strip_include_prefix = strip_include_prefix,
        deps = deps,
        visibility = ["//visibility:private"],
    )

    native.cc_library(
        name = name,
        textual_hdrs = hdrs,
        deps = deps + [":%s_virtual" % name],
        **kwargs
    )

def cuda_cc_library(copts = [], **kwargs):
    """Wrapper over cc_library which adds default HIP options."""
    native.cc_library(copts = rocm_default_copts() + copts, **kwargs)

def cuda_cc_binary(copts = [], **kwargs):
    """Wrapper over cc_binary which adds default HIP options."""
    native.cc_binary(copts = rocm_default_copts() + copts, **kwargs)

def cuda_cc_test(copts = [], **kwargs):
    """Wrapper over cc_test which adds default HIP options."""
    native.cc_test(copts = rocm_default_copts() + copts, **kwargs)
