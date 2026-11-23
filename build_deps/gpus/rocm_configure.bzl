"""Repository rule for ROCm autoconfiguration.

`rocm_configure` creates `@local_config_rocm`, which holds a clang toolchain
that compiles HIP and the `cuda_cc_library` family of macros for HIP, so the
same BUILD files build for AMD GPUs. It depends on the following environment
variables:

  * `ROCM_PATH`: The ROCm installation root. Default is `/opt/rocm`.
  * `ROCM_AMDGPU_TARGETS`: Comma-separated AMD GPU targets, for example
    `gfx942,gfx1100`. Default is the GPUs found by `offload-arch`.
"""

load(
    "//build_deps/remote_config:common.bzl",
    "err_out",
    "get_host_environ",
    "raw_exec",
)

_ROCM_PATH = "ROCM_PATH"
_ROCM_AMDGPU_TARGETS = "ROCM_AMDGPU_TARGETS"

def _fail(msg):
    fail("\n%sROCm Configuration Error:%s %s\n" % ("\033[0;31m", "\033[0m", msg))

def _find_rocm_path(repository_ctx):
    rocm_path = get_host_environ(repository_ctx, _ROCM_PATH, "/opt/rocm")
    if not repository_ctx.path(rocm_path + "/include/hip/hip_runtime.h").exists:
        _fail("Cannot find hip/hip_runtime.h under %s; set %s to the ROCm root." %
              (rocm_path, _ROCM_PATH))
    return str(repository_ctx.path(rocm_path).realpath)

def _find_clang(repository_ctx, rocm_path):
    for llvm in ["lib/llvm", "llvm"]:
        clang = repository_ctx.path("%s/%s/bin/clang" % (rocm_path, llvm))
        if clang.exists:
            return "%s/%s" % (rocm_path, llvm)
    _fail("Cannot find the ROCm clang under %s." % rocm_path)

def _amdgpu_targets(repository_ctx, rocm_path):
    targets = get_host_environ(repository_ctx, _ROCM_AMDGPU_TARGETS, "")
    if not targets:
        result = raw_exec(repository_ctx, [rocm_path + "/bin/offload-arch"])
        targets = ",".join(result.stdout.split())
    found = []
    for target in targets.split(","):
        target = target.strip()
        if target and target not in found:
            found.append(target)
    if not found:
        _fail("No AMD GPU target found; set %s, for example gfx942." %
              _ROCM_AMDGPU_TARGETS)
    return found

def _cxx_inc_directories(repository_ctx, clang, lang):
    result = raw_exec(repository_ctx, [clang, "-E", "-x" + lang, "-", "-v"])
    stderr = err_out(result)
    begin = stderr.find("#include <...>")
    end = stderr.find("End of search list.")
    if begin == -1 or end == -1:
        return []
    begin = stderr.find("\n", begin)
    return [p.strip() for p in stderr[begin:end].strip().split("\n")]

def _create_local_rocm_repository(repository_ctx):
    rocm_path = _find_rocm_path(repository_ctx)
    llvm_path = _find_clang(repository_ctx, rocm_path)
    clang = llvm_path + "/bin/clang"
    targets = _amdgpu_targets(repository_ctx, rocm_path)

    resource_dir = raw_exec(
        repository_ctx,
        [clang, "-print-resource-dir"],
    ).stdout.strip()
    include_dirs = [rocm_path + "/include", resource_dir + "/include"]
    for lang in ["c++", "c"]:
        for d in _cxx_inc_directories(repository_ctx, clang, lang):
            if d not in include_dirs:
                include_dirs.append(d)

    repository_ctx.template(
        "rocm/BUILD",
        Label("//build_deps/gpus/rocm:BUILD.tpl"),
        {"%{rocm_path}": rocm_path},
    )
    repository_ctx.template(
        "rocm/build_defs.bzl",
        Label("//build_deps/gpus/rocm:build_defs.bzl.tpl"),
        {
            "%{rocm_path}": rocm_path,
            "%{amdgpu_targets}": str(targets),
        },
    )
    repository_ctx.template(
        "crosstool/BUILD",
        Label("//build_deps/gpus/rocm:crosstool.BUILD.tpl"),
        {
            "%{llvm_path}": llvm_path,
            "%{cxx_builtin_include_directories}": str(include_dirs),
        },
    )

rocm_configure = repository_rule(
    implementation = _create_local_rocm_repository,
    environ = [_ROCM_PATH, _ROCM_AMDGPU_TARGETS],
)
