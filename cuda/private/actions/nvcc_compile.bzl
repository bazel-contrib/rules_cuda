def compile(ctx, cuda_toolchain, cc_toolchain, includes, system_includes, quote_includes, headers, translation_unit, output_basename, pic = False, rdc = False):
    ""
    actions = ctx.actions
    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    obj_ext = ".o"  # TODO: platform dependent
    rdc_ext = ".rdc"
    pic_ext = ".pic"
    ext = []

    if rdc:
        ext.append(rdc_ext)
    if pic:
        ext.append(pic_ext)
    ext.append(obj_ext)
    ext = "".join(ext)

    obj_file = actions.declare_file(output_basename + ext)

    args = actions.args()
    args.add("-ccbin", host_compiler)
    host_compiler_options = "-xc++"
    if pic: # FIXME: not MSVC
        host_compiler_options += " -fPIC"
    args.add("-Xcompiler", host_compiler_options)
    args.add("-x", "cu")
    args.add("-rdc", "true" if rdc else "false")
    args.add_all(includes, before_each="-I", uniquify=True)
    args.add_all(system_includes, before_each="-isystem", uniquify=True)
    args.add_all(quote_includes, before_each="-iquote", uniquify=True)
    args.add("-c", translation_unit.path)
    args.add("-o", obj_file.path)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [obj_file],
        inputs = depset([translation_unit], transitive = [headers, cc_toolchain.all_files]),
        env = {
            "PATH": "/usr/bin",
        },
    )

    return obj_file
