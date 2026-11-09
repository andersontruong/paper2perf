"""Scoped compatibility for the private ShapeEnv API used by bundled Parabolic.

PyTorch 2.10 calls backed-symbol storage ``var_to_val`` and registration
``add_var_to_val``. Newer versions call them ``backed_var_to_val`` and
``add_backed_var_to_val``. They are NOT the unbacked-symbol storage.
"""
from contextlib import contextmanager
from importlib import metadata
from threading import RLock


_lock = RLock()
REQUIRED_TORCH_VERSION = "2.10.0"


def torch_version_status():
    try:
        installed = metadata.version("torch")
    except metadata.PackageNotFoundError:
        installed = None
    return {"required": REQUIRED_TORCH_VERSION, "installed": installed,
            "matches": installed is not None and installed.split("+", 1)[0] == REQUIRED_TORCH_VERSION}


@contextmanager
def shape_env_compatibility(shape_env_type=None):
    """Expose renamed APIs only during simulator execution; restore on all exits."""
    if shape_env_type is None:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        shape_env_type = ShapeEnv
    with _lock:
        probe = shape_env_type()
        try:
            has_backed = hasattr(probe, "backed_var_to_val")
            has_legacy = hasattr(probe, "var_to_val")
            modern_registration = callable(getattr(probe, "add_backed_var_to_val", None))
            legacy_registration = callable(getattr(probe, "add_var_to_val", None))
        finally:
            cleanup = getattr(probe, "cleanup", None)
            if callable(cleanup):
                cleanup()
        if not has_backed and not has_legacy:
            raise RuntimeError("Unsupported PyTorch ShapeEnv: backed-symbol storage is missing")
        if not modern_registration and not legacy_registration:
            raise RuntimeError("Unsupported PyTorch ShapeEnv: backed-symbol registration is missing")
        added = []
        try:
            if not has_backed:
                setattr(shape_env_type, "backed_var_to_val", property(lambda self: self.var_to_val))
                added.append("backed_var_to_val")
            if not modern_registration:
                setattr(shape_env_type, "add_backed_var_to_val", shape_env_type.add_var_to_val)
                added.append("add_backed_var_to_val")
            yield {"backed_storage": "backed_var_to_val" if has_backed else "var_to_val",
                   "registration": "add_backed_var_to_val" if modern_registration else "add_var_to_val",
                   "aliases_applied": list(added)}
        finally:
            for name in reversed(added):
                delattr(shape_env_type, name)
