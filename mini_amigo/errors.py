"""Exceptions for mini_amigo (CONSTITUTION Art. VI.5)."""


class MiniAmigoError(RuntimeError):
    """Base class for all mini_amigo errors."""


class InputError(MiniAmigoError):
    """Invalid or inconsistent problem definition."""


class BuildError(MiniAmigoError):
    """Library or model compilation failure (full toolchain output attached)."""


class SimError(MiniAmigoError):
    """Simulation failed at the C level."""
