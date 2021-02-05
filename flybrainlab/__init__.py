from __future__ import absolute_import

__version__ = "0.1.1"
from .Client import *
from .widget import WidgetManager

_initialized = False


def init():
    """Initialize FBLClient for current kernel"""
    global _initialized
    global widget_manager
    global client_manager
    global get_client
    if _initialized:
        return
    widget_manager = WidgetManager()
    client_manager = MetaClient()
    def get_client(id = 0):
        keys = list(client_manager.clients.keys())
        key = keys[id]
        client = client_manager.clients[key]['client']
        return client
    _initialized = True
