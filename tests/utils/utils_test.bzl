load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "ArchSpecInfo", "Stage2ArchInfo")

def _get_arch_specs_test_impl(ctx):
    env = unittest.begin(ctx)

    asserts.equals(env, [], cuda_helper.get_arch_specs(""))

    asserts.equals(env, [], cuda_helper.get_arch_specs(";"))

    ref = [ArchSpecInfo(
        stage1_arch = "80",
        stage2_archs = [
            Stage2ArchInfo(arch = "80", virtual = False, gpu = True, lto = False),
            Stage2ArchInfo(arch = "86", virtual = False, gpu = True, lto = False),
        ],
    )]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("compute_80:sm_80,sm_86"))

    ref = [ArchSpecInfo(
        stage1_arch = "60",
        stage2_archs = [
            Stage2ArchInfo(arch = "60", virtual = True, gpu = False, lto = False),
            Stage2ArchInfo(arch = "61", virtual = False, gpu = True, lto = False),
            Stage2ArchInfo(arch = "62", virtual = False, gpu = True, lto = False),
        ],
    )]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("compute_60:compute_60,sm_61,sm_62"))

    ref = [ArchSpecInfo(
        stage1_arch = "80",
        stage2_archs = [Stage2ArchInfo(arch = "80", virtual = True, gpu = False, lto = False)],
    )]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("compute_80:compute_80"))

    ref = [ArchSpecInfo(
        stage1_arch = "80",
        stage2_archs = [
            Stage2ArchInfo(arch = "80", virtual = False, gpu = True, lto = False),
            Stage2ArchInfo(arch = "86", virtual = False, gpu = True, lto = False),
        ],
    )]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("sm_80,sm_86"))

    ref = [
        ArchSpecInfo(
            stage1_arch = "80",
            stage2_archs = [Stage2ArchInfo(arch = "80", virtual = False, gpu = True, lto = False)],
        ),
        ArchSpecInfo(
            stage1_arch = "86",
            stage2_archs = [Stage2ArchInfo(arch = "86", virtual = False, gpu = True, lto = False)],
        ),
    ]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("sm_80;sm_86"))

    ref = [ArchSpecInfo(
        stage1_arch = "80",
        stage2_archs = [Stage2ArchInfo(arch = "80", virtual = True, gpu = False, lto = False)],
    )]
    asserts.equals(env, ref, cuda_helper.get_arch_specs("compute_80"))

    return unittest.end(env)

get_arch_specs_test = unittest.make(_get_arch_specs_test_impl)
