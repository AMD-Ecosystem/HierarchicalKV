# This file is expanded from a template by rocm_configure.bzl

load("@bazel_tools//tools/cpp:unix_cc_toolchain_config.bzl", "cc_toolchain_config")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "k8": ":cc-compiler-local",
        "k8|clang": ":cc-compiler-local",
    },
)

filegroup(name = "empty")

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":cc-compiler-local-config",
    toolchain_identifier = "rocm_clang",
)

cc_toolchain_config(
    name = "cc-compiler-local-config",
    abi_libc_version = "local",
    abi_version = "local",
    compile_flags = [
        "-U_FORTIFY_SOURCE",
        "-fstack-protector",
        "-Wall",
        "-fno-omit-frame-pointer",
    ],
    compiler = "clang",
    coverage_compile_flags = ["--coverage"],
    coverage_link_flags = ["--coverage"],
    cpu = "k8",
    cxx_builtin_include_directories = %{cxx_builtin_include_directories},
    cxx_flags = ["-std=c++17"],
    dbg_compile_flags = ["-g"],
    host_system_name = "local",
    link_flags = [
        "-Wl,-no-as-needed",
        "-Wl,-z,relro,-z,now",
    ],
    link_libs = [
        "-lstdc++",
        "-lm",
    ],
    opt_compile_flags = [
        "-g0",
        "-O2",
        "-D_FORTIFY_SOURCE=1",
        "-DNDEBUG",
        "-ffunction-sections",
        "-fdata-sections",
    ],
    opt_link_flags = ["-Wl,--gc-sections"],
    supports_start_end_lib = False,
    target_libc = "local",
    target_system_name = "local",
    tool_paths = {
        "ar": "%{llvm_path}/bin/llvm-ar",
        "cpp": "%{llvm_path}/bin/clang-cpp",
        "dwp": "/usr/bin/dwp",
        "gcc": "%{llvm_path}/bin/clang",
        "gcov": "/usr/bin/gcov",
        "ld": "%{llvm_path}/bin/ld.lld",
        "llvm-cov": "%{llvm_path}/bin/llvm-cov",
        "nm": "%{llvm_path}/bin/llvm-nm",
        "objcopy": "%{llvm_path}/bin/llvm-objcopy",
        "objdump": "%{llvm_path}/bin/llvm-objdump",
        "strip": "%{llvm_path}/bin/llvm-strip",
    },
    toolchain_identifier = "rocm_clang",
    unfiltered_compile_flags = [
        "-no-canonical-prefixes",
        "-Wno-builtin-macro-redefined",
        "-D__DATE__=\"redacted\"",
        "-D__TIMESTAMP__=\"redacted\"",
        "-D__TIME__=\"redacted\"",
    ],
)
