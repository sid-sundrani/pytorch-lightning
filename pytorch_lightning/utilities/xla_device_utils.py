from warnings import warn

warn("`xla_device_utils` package has been renamed to `xla_device` since v1.2 and will be removed in v1.4",
     DeprecationWarning)

from pytorch_lightning.utilities.xla_device import *  # noqa: F403 E402 F401
