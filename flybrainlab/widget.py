from ipykernel.comm import Comm
from collections import OrderedDict
import dataclasses


@dataclasses.dataclass
class Widget:
    """Widget class that keeps track of all the information regarding a widget.
    """

    widget_type: str  # neu3d, neugfx, etc.
    widget_id: str
    client_id: str
    model: "typing.Any"
    comm: Comm
    msg_data: "typing.Any"
    isDisposed: bool = False
    commOpen: bool = True
    

    def send_data(self, data):
        if self.comm:
            self.comm.send(data)

class CallbackManager:
    """Callback manager class that stores a number of callbacks that try to access messages sent from frontend to the Python kernel.

    # Example:
        Start by creating a Neu3D widget in the frontend, then execute this code in a Jupyter notebook running the same kernel.
        ```python
        my_data = []
        def process_data(data): # Example callback function
            if 'hello' in data:
                my_data.append(data)
        fbl.widget_manager.callback_manager.reset() # Remove old callbacks
        fbl.widget_manager.callback_manager.add(process_data)
        # fbl.widget_manager.callback_manager.add(process_data, "69409cc87c344c9db06654848821b58a") # If you want a specific comm id, give it as the second argument
        # The comm id corresponds to _id attribute of the CommHandler in the frontend, or to one of
        # print([i.comm.comm_id for i in fbl.widget_manager.widgets.values()])
        # in the backend.
        print(fbl.widget_manager.widgets)
        # in JS, execute:
        # window.neu3d_widget.comm.send('hello world')
        # window.neu3d_widget.comm.send('my world')
        print(my_data) # returns ['hello world']
        ```

    # Attributes:
        callbacks (list): A list of functions.

    """
    def __init__(self):
        """Initializer function for the callback manager.
        
        # Returns:
            
        """
        self.callbacks = {}
        self.global_callbacks = []

    def reset(self):
        """Removes existing callbacks.
        
        # Returns:
            
        """
        self.global_callbacks = []
        self.callbacks = {}

    def add(self, func, comm_id  = None):
        """Add a function to the callback manager.
        
        # Returns:
            
        """
        if comm_id is None:
            self.global_callbacks.append(func)
        else:
            if comm_id not in self.callbacks:
                self.callbacks[comm_id] = []
            self.callbacks[comm_id].append(func)
            

    def run(self, comm_id, data):
        """Executes the stored callbacks one by one. Automatically called by WidgetManager.
        
        # Returns:
            
        """
        if comm_id in self.callbacks:
            for func in self.callbacks[comm_id]:
                func(data)
        for func in self.global_callbacks:
            func(data)

    
class WidgetManager(object):
    """Widget Manager class that keeps track of all widgets associated with the current FBLClient session.

    # Attributes:
        widgets (dict): a dictionary of instances of Widget.
        _comms (dict): all comm objects opend for this client.
    """

    def __init__(self):
        self._comms = OrderedDict()
        self.widgets = OrderedDict()
        self.callback_manager = CallbackManager()

    def add_widget(self, widget_id, client_id, widget_type, comm_target):
        """Add a widget to manager.
        
        # Returns:
            
        """
        comm = self.find_comm(comm_target=comm_target) # Try to find a comm object
        if not comm: # Create comm if does not exist
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
                self.callback_manager.run(comm_id, data)

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
                client_id = client_id,
                model=None,
                comm=comm,
                isDisposed=False,
                msg_data=None,
            )

        # make sure that comm is open
        comm.open()

    def find_comm(self, comm_id=None, comm_target=None):
        """Find a comm object either by id or by target name
        """
        if comm_id:
            comm = [c for c in self._comms.values() if c.comm_id == comm_id]
        elif comm_target:
            comm = [c for c in self._comms.values() if c.target_name == comm_target]
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


