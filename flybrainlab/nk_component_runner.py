from flybrainlab.NeurokernelComponent import *
import time
from configparser import ConfigParser
import argparse

if __name__ == "__main__":
    import neurokernel.mpi_relaunch

    from twisted.internet._sslverify import OpenSSLCertificateAuthorities
    from twisted.internet.ssl import CertificateOptions
    import OpenSSL.crypto

    from configparser import ConfigParser


    parser = argparse.ArgumentParser()
    parser.add_argument('--name', dest='name', type=str, default = 'nk',
                    help='Name of the server')
    args = parser.parse_args()

    root = os.path.expanduser("/")
    home = os.path.expanduser("~")
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_files = []
    config_files.append(os.path.join(home, ".ffbo/config", "ffbo.FBLClient.ini"))
    config_files.append(os.path.join(root, ".ffbo/config", "ffbo.FBLClient.ini"))
    config_files.append(os.path.join(home, ".ffbo/config", "config.ini"))
    config_files.append(os.path.join(root, ".ffbo/config", "config.ini"))
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

    Component = ffbolabComponent(user = user, secret=secret, url = url, ssl = ssl, debug = debug, realm = realm,
                                 ca_cert_file = ca_cert_file, intermediate_cert_file = intermediate_cert_file,
                                 authentication = authentication, server_name = args.name)
    server = neurokernel_server()

    print(printHeader("FFBO Neurokernel Component") + "Connection successful.")

    while True:
        # print(printHeader('FFBO Neurokernel Component') + 'Operations nominal.')
        mainThreadExecute(Component, server)
        time.sleep(1)
