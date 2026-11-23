# This file is expanded from a template by rocm_configure.bzl

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# The HIP runtime. The ROCm headers are found through the toolchain's
# built-in include directories.
cc_library(
    name = "rocm",
    linkopts = [
        "-L%{rocm_path}/lib",
        "-Wl,-rpath,%{rocm_path}/lib",
        "-lamdhip64",
    ],
)
