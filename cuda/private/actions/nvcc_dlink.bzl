""

def device_link(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        objects,
        output_basename,
        link_flags = [],
        pic = False,
        rdc = False,
        dlto = False):
    """perform device link, return a dlink-ed object file"""

    actions = ctx.actions
    cuda_compiler = cuda_toolchain.compiler_executable

    obj_ext = ".o"
    pic_ext = ".pic" if pic else ""
    dlink_suffix = "_dlink" if rdc else ""

    obj = actions.declare_file(output_basename + dlink_suffix + pic_ext + obj_ext)
    args = actions.args()
    args.add_all(link_flags)
    args.add("-dlink")
    args.add_all(objects)
    args.add("-o", obj.path)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [obj],
        inputs = objects,
        env = {
            "PATH": "/usr/bin",
        },
    )

    return obj
