load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaArchsInfo")

def _cuda_archs_flag_impl(ctx):
    specs_str = ctx.build_setting_value
    return CudaArchsInfo(arch_specs = cuda_helper.get_arch_specs(specs_str))

cuda_archs_flag = rule(
    implementation = _cuda_archs_flag_impl,
    build_setting = config.string(flag = True),
    provides = [CudaArchsInfo],
)
