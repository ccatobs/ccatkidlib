from contextlib import contextmanager
import ccatkidlib.utils as ccat_utils

"""
Configuration management utilities for analysis workflows.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
"""


@contextmanager
def return_prefix(analysis_cfg: dict) -> None:
    """Context manager to temporarily enable the |RFSoC| return prefix setting.

    Sets ``analysis_cfg["runtime"]["return_prefix"]`` to **True** for the duration of the
    ``with`` block, yielding the original value so callers know its prior state. Upon exit,
    restores the original value.

    Args:
        analysis_cfg (dict): Analysis configuration dictionary containing
            ``['runtime']['return_prefix']`` key.
    Yields:
        Original value of ``analysis_cfg["runtime"]["return_prefix"]`` before the context
        manager entry.
    """
    ret_prefix = analysis_cfg["runtime"]["return_prefix"]
    if ret_prefix:
        yield ret_prefix
    else:
        analysis_cfg["runtime"]["return_prefix"] = True
        try:
            yield ret_prefix
        finally:
            analysis_cfg["runtime"]["return_prefix"] = ret_prefix
