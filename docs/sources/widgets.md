### CallbackManager


```python
flybrainlab.widget.CallbackManager()
```


Callback manager class that stores a number of callbacks that try to access messages sent from frontend to the Python kernel.

__Example:__

# Start by creating a Neu3D widget in the frontend, then execute this code in a Jupyter notebook running the same kernel.
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

__Attributes:__

callbacks (list): A list of functions.


----

### WidgetManager


```python
flybrainlab.widget.WidgetManager()
```


Widget Manager class that keeps track of all widgets associated with the current FBLClient session.

__Attributes:__

widgets (dict): a dictionary of instances of Widget.
_comms (dict): all comm objects opend for this client.


----

### MetaClient


```python
flybrainlab.Client.MetaClient(initializer=None)
```


FlyBrainLab MetaClient class that tracks available FBL clients and connected frontend widgets.

__Attributes:__

clients (obj): A list of dictionaries with the following keys: (i) 'name': Contains the common name of the client. (ii) 'client': A reference to the client object. (iii) 'widgets': List of widget names associated with the client.


----

