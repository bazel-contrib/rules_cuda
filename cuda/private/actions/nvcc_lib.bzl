""

def create_library(ctx, cuda_toolchain, objects, output_basename, pic = False):
    """create static library"""

    actions = ctx.actions
    cuda_compiler = cuda_toolchain.compiler_executable

    lib_ext = ".a"
    pic_ext = ".pic" if pic else ""

    lib = actions.declare_file(output_basename + pic_ext + lib_ext)
    args = actions.args()
    args.add("-lib")
    args.add_all(objects)
    args.add("-o", lib.path)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [lib],
        inputs = objects,
        env = {
            "PATH": "/usr/bin",
        },
    )

    return lib
