from GFXComponent import *
import time

from configparser import ConfigParser

root = os.path.expanduser("/")
home = os.path.expanduser("~")
filepath = os.path.dirname(os.path.abspath(__file__))
config_files = []
config_files.append(os.path.join(home, "config", "ffbo.FBLClient.ini"))
config_files.append(os.path.join(root, "config", "ffbo.FBLClient.ini"))
config_files.append(os.path.join(home, "config", "config.ini"))
config_files.append(os.path.join(root, "config", "config.ini"))
config_files.append(os.path.join(filepath, "..", "FBLClient.ini"))
config = ConfigParser()
configured = False
file_type = 0
for config_file in config_files:
    if os.path.exists(config_file):
        config.read(config_file)
        configured = True
        break
    file_type += 1
if not configured:
    raise Exception("No config file exists for this component")

user = config["USER"]["user"]
secret = config["USER"]["secret"]
ssl = eval(config["AUTH"]["ssl"])
websockets = "wss" if ssl else "ws"
if "ip" in config["SERVER"]:
    ip = config["SERVER"]["ip"]
else:
    ip = "localhost"
port = int(config["NLP"]['expose-port'])
url =  "{}://{}:{}/ws".format(websockets, ip, port)
realm = config["SERVER"]["realm"]
authentication = eval(config["AUTH"]["authentication"])
debug = eval(config["DEBUG"]["debug"])
ca_cert_file = config["AUTH"]["ca_cert_file"]
intermediate_cert_file = config["AUTH"]["intermediate_cert_file"]


_FFBOLABComponent = ffbolabComponent(user = user, secret=secret, url = url, ssl = ssl, debug = debug, realm = realm,
                                     ca_cert_file = ca_cert_file, intermediate_cert_file = intermediate_cert_file,
                                     authentication = authentication)

print(printHeader('FFBOLab Client GFX') + 'Connection successful.')

while True:
    print(printHeader('FFBOLab Client GFX') + 'Operations nominal.')
    mainThreadExecute(_FFBOLABComponent)
    time.sleep(1)
