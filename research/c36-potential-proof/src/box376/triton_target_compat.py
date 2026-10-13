"""Opt-in per-process compatibility for BSC's legacy Torch target tuples."""
import functools


def enable_legacy_target_compat():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.runtime.driver import driver

    original = triton.compile

    @functools.wraps(original)
    def compile_with_target(src, target=None, options=None):
        if isinstance(target, (tuple, list)):
            current = driver.active.get_current_target()
            if len(target) not in (2, 3) or target[0] != current.backend or target[1] != current.arch:
                raise ValueError("Legacy compiler target differs from the actual CUDA device")
            if len(target) == 3 and target[2] != current.warp_size:
                raise ValueError("Legacy compiler warp size differs from the actual CUDA device")
            assert isinstance(current, GPUTarget)
            target = current
        return original(src, target=target, options=options)

    triton.compile = compile_with_target
    return {"legacy_tuple_target_adapter": True, "triton": triton.__version__,
            "scope": "Target representation only; original source, compiler options and actual device unchanged"}
