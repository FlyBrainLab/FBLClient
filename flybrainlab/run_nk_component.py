from NeurokernelComponent import *
import time


if __name__ == "__main__":
    import neurokernel.mpi_relaunch

    from twisted.internet._sslverify import OpenSSLCertificateAuthorities
    from twisted.internet.ssl import CertificateOptions
    import OpenSSL.crypto

    Component = ffbolabComponent(secret=u"")
    server = neurokernel_server()

    print(printHeader("FFBO Neurokernel Component") + "Connection successful.")

    while True:
        # print(printHeader('FFBO Neurokernel Component') + 'Operations nominal.')
        mainThreadExecute(Component, server)
        time.sleep(1)
