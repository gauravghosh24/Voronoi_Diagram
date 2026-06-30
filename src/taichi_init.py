"""Taichi backend initialization helpers."""

import warnings


def _load_taichi():
    try:
        import taichi as ti
    except ImportError as exc:
        raise ImportError(
            "Taichi is required for GPU/CPU kernels. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return ti


def _reset_taichi(ti):
    try:
        ti.reset()
    except Exception:
        # Older Taichi builds may not need or expose a successful reset before init.
        pass


def _active_arch_name(ti, requested):
    try:
        active_arch = ti.lang.impl.current_cfg().arch
        if active_arch == ti.cpu:
            return "cpu"
    except Exception:
        pass
    return "gpu" if requested == "gpu" else "cpu"


def init_taichi(arch="gpu"):
    """Initialize Taichi and return ``(ti, active_arch_name)``.

    Parameters
    ----------
    arch:
        ``"gpu"`` or ``"cpu"``. If GPU initialization fails, CPU is used.
    """

    ti = _load_taichi()
    requested = str(arch).lower()
    if requested not in ("gpu", "cpu"):
        raise ValueError("arch must be 'gpu' or 'cpu'")

    _reset_taichi(ti)

    if requested == "cpu":
        ti.init(arch=ti.cpu)
        print("Taichi backend: cpu")
        return ti, "cpu"

    try:
        ti.init(arch=ti.gpu)
        active = _active_arch_name(ti, requested)
        print("Taichi backend: {}".format(active))
        return ti, active
    except Exception as exc:
        message = "GPU Taichi initialization failed; falling back to CPU. {}".format(exc)
        warnings.warn(message)
        print("WARNING: {}".format(message))
        _reset_taichi(ti)
        ti.init(arch=ti.cpu)
        print("Taichi backend: cpu")
        return ti, "cpu"
