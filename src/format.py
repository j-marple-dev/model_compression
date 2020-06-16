# -*- coding: utf-8 -*-
"""String formats for logging.

- Author: Curt-Park
- Email: jwpark@jmarple.ai
"""


def default_format(x: float) -> str:
    """General format used for loss, hyper params, etc."""
    return str(round(x, 6))


def percent_format(x: float) -> str:
    """Return a formatted string for percent."""
    return f"{x:.2f}%"
