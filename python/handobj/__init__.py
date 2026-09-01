"""Vendored HandObject hand detector (100DOH-TinyExplorer-Tuned Faster R-CNN).

The heavy PyTorch model lives under ``handobj.inference``; import it lazily so
that merely importing this package (or the ``hand_handobj`` detector) does not
require torch or the native ``model._C`` extension to be present.
"""

__all__ = ["inference"]
