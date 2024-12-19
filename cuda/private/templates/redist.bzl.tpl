load("@rules_cuda//cuda:repositories.bzl", "cuda_component")

def rules_cuda_components():
    %{rules_cuda_components_content}

def rules_cuda_components_and_toolchains(register=False):
    rules_cuda_components()
    register_detected_cuda_toolchains()
