load("@rules_cuda//cuda:repositories.bzl", "cuda_component", "rules_cuda_toolchains")

def rules_cuda_components():
    # See template_helper.generate_redist_bzl(...) for body generation logic
    %{rules_cuda_components_body}

    return %{components_mapping}

def rules_cuda_components_and_toolchains(register_toolchains = False):
    components_mapping = rules_cuda_components()
    rules_cuda_toolchains(
        components_mapping= components_mapping,
        register_toolchains = register_toolchains,
    )
