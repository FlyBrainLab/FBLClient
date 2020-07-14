from __future__ import absolute_import

__version__ = "0.1.0"
from .Client import *
from .widget import WidgetManager

_initialized = False


def init():
    """Initialize FBLClient for current kernel"""
    global _initialized
    global widget_manager
    global client_manager
    if _initialized:
        return
    widget_manager = WidgetManager()
    client_manager = MetaClient()
    _initialized = True
