from ipykernel.comm import Comm
from collections import OrderedDict
import dataclasses


@dataclasses.dataclass
class Widget:
    """Widget instance

    Keeps track of all the information regarding a widget
    """

    widget_type: str  # neu3d, neugfx, etc.
    comm: Comm
    widget_id: str
    model: "typing.Any"
    msg_data: "typing.Any"
    isDisposed: bool = False
    commOpen: bool = True

    def send_data(self, data):
        if self.comm:
            self.comm.send(data)


class WidgetManager(object):
    """Widget Manager
    Keeps track of all widgets associated with the current FBLClient session

    # Attributes
        widgets (dict): a dictionary of instances of Widget
        _comms (dict): all comm objects opend for this client
    """

    def __init__(self):
        self._comms = OrderedDict()
        self.widgets = OrderedDict()

    def add_widget(self, widget_id, widget_type, comm_target):
        """Add a widget to manager
        
        # Return:
            
        """
        # create comm if does not exist
        comm = self.find_comm(comm_target=comm_target)
        if not comm:
            comm = Comm(target_name=comm_target)
            self._comms[widget_id] = comm

            @comm.on_msg
            def handle_msg(msg):
                comm_id = msg["content"]["comm_id"]
                data = msg["content"]["data"]
                nonlocal self
                widget = self.find_widget_by_comm_id(comm_id)

                widget.msg_data = data
                if data == "dispose":
                    widget.isDisposed = True

            @comm.on_close
            def handle_close(msg):
                comm_id = msg["content"]["comm_id"]
                nonlocal self
                widget = self.find_widget_by_comm_id(comm_id)
                widget.commOpen = False

        if widget_id not in self.widgets:
            self.widgets[widget_id] = Widget(
                widget_type=widget_type,
                widget_id=widget_id,
                model=None,
                comm=comm,
                isDisposed=False,
                msg_data=None,
            )

    def find_comm(self, comm_id=None, comm_target=None):
        """Find a comm object either by id or by target name
        """
        if comm_id:
            comm = [c for c in self._comms.values() if c.comm_id == comm_id]
        elif comm_target:
            comm = [c for c in self._comms.values() if c.target_name == comm_id]
        else:
            comm = []
        if len(comm) == 1:
            return comm[0]
        else:
            return None

    def find_widget_by_comm_id(self, comm_id):
        w = [w for w in self.widgets.values() if w.comm.comm_id == comm_id]
        if len(w) == 1:
            return w[0]
        else:
            return None

    def send_data(self, widget_id, data):
        if widget_id in self.widgets:
            self.widgets[widget_id].send_data(data)
