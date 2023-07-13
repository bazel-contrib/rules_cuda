# `optf` Example

This example demonstrates how to use `.optf` files (aka. 'option file') in your build.

If you've never heard of this feature, here's the gist:
Instead of storing compiler flags and defines as string variables in your `BUILD.bazel` file,
you can pass a filename to `nvcc` (via the `-optf` argument), which points to a file that stores
your flags.

When using `.optf` files together with `bazel`, you gain a lot of flexibility in your build.
For example, you can have multiple sets of `.optf` files, e.g. one per architecture,
and pick the right set during analysis stage.
Or, you could even generate them during the execution stage using `genrule()`.

All `.optf` files are treated as dependencies to the objects that use it during build.
This way, `bazel` is aware that it needs to rebuild the library, if the `.optf` file changes,
just like you would expect with a source code change.

## per library example

The first example demonstrates how you can specify an `.optf` file for a whole library.
The `.optf` file used is the `example.optf` in this folder.
It also shows that you can use the `.optf` file for both defines (e.g. `-DFEATURE_A`),
as well as compiler options (e.g. `-fma=true`).

Build and run it like so:

```bash
bazel run //optf:executable_with_global_optf
```

## per file example

The second example demonstrates how you can specify an `.optf` file per object.

It uses a small macro that tries to find the correct `.optf` file for each `.cu` file,
and, if yes, adds it to the rule's argument list.
Finally, it creates a library target that ties all objects together.

Something like this can be useful for example, if you can't tolerate the precision loss of
`--use-fast-math` in some kernels, but still want the speedup in other places.

Build and run it like so:

```bash
bazel run //optf:executable_with_per_file_optf
```
