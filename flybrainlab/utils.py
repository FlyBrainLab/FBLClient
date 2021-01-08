
from itertools import islice

from autobahn.websocket.protocol import WebSocketProtocol


def setProtocolOptions(transport,
                       version=None,
                       utf8validateIncoming=None,
                       acceptMaskedServerFrames=None,
                       maskClientFrames=None,
                       applyMask=None,
                       maxFramePayloadSize=None,
                       maxMessagePayloadSize=None,
                       autoFragmentSize=None,
                       failByDrop=None,
                       echoCloseCodeReason=None,
                       serverConnectionDropTimeout=None,
                       openHandshakeTimeout=None,
                       closeHandshakeTimeout=None,
                       tcpNoDelay=None,
                       perMessageCompressionOffers=None,
                       perMessageCompressionAccept=None,
                       autoPingInterval=None,
                       autoPingTimeout=None,
                       autoPingSize=None):
    """ from autobahn.websocket.protocol.WebSocketClientFactory.setProtocolOptions """
    transport.factory.setProtocolOptions(
            version = version,
            utf8validateIncoming = utf8validateIncoming,
            acceptMaskedServerFrames = acceptMaskedServerFrames,
            maskClientFrames = maskClientFrames,
            applyMask = applyMask,
            maxFramePayloadSize = maxFramePayloadSize,
            maxMessagePayloadSize = maxMessagePayloadSize,
            autoFragmentSize = autoFragmentSize,
            failByDrop = failByDrop,
            echoCloseCodeReason = echoCloseCodeReason,
            serverConnectionDropTimeout = serverConnectionDropTimeout,
            openHandshakeTimeout = openHandshakeTimeout,
            closeHandshakeTimeout = closeHandshakeTimeout,
            tcpNoDelay = tcpNoDelay,
            perMessageCompressionOffers = perMessageCompressionOffers,
            perMessageCompressionAccept = perMessageCompressionAccept,
            autoPingInterval = autoPingInterval,
            autoPingTimeout = autoPingTimeout,
            autoPingSize = autoPingSize)

    if version is not None:
        if version not in WebSocketProtocol.SUPPORTED_SPEC_VERSIONS:
            raise Exception("invalid WebSocket draft version %s (allowed values: %s)" % (version, str(WebSocketProtocol.SUPPORTED_SPEC_VERSIONS)))
        if version != transport.version:
            transport.version = version

    if utf8validateIncoming is not None and utf8validateIncoming != transport.utf8validateIncoming:
        transport.utf8validateIncoming = utf8validateIncoming

    if acceptMaskedServerFrames is not None and acceptMaskedServerFrames != transport.acceptMaskedServerFrames:
        transport.acceptMaskedServerFrames = acceptMaskedServerFrames

    if maskClientFrames is not None and maskClientFrames != transport.maskClientFrames:
        transport.maskClientFrames = maskClientFrames

    if applyMask is not None and applyMask != transport.applyMask:
        transport.applyMask = applyMask

    if maxFramePayloadSize is not None and maxFramePayloadSize != transport.maxFramePayloadSize:
        transport.maxFramePayloadSize = maxFramePayloadSize

    if maxMessagePayloadSize is not None and maxMessagePayloadSize != transport.maxMessagePayloadSize:
        transport.maxMessagePayloadSize = maxMessagePayloadSize

    if autoFragmentSize is not None and autoFragmentSize != transport.autoFragmentSize:
        transport.autoFragmentSize = autoFragmentSize

    if failByDrop is not None and failByDrop != transport.failByDrop:
        transport.failByDrop = failByDrop

    if echoCloseCodeReason is not None and echoCloseCodeReason != transport.echoCloseCodeReason:
        transport.echoCloseCodeReason = echoCloseCodeReason

    if serverConnectionDropTimeout is not None and serverConnectionDropTimeout != transport.serverConnectionDropTimeout:
        transport.serverConnectionDropTimeout = serverConnectionDropTimeout

    if openHandshakeTimeout is not None and openHandshakeTimeout != transport.openHandshakeTimeout:
        transport.openHandshakeTimeout = openHandshakeTimeout

    if closeHandshakeTimeout is not None and closeHandshakeTimeout != transport.closeHandshakeTimeout:
        transport.closeHandshakeTimeout = closeHandshakeTimeout

    if tcpNoDelay is not None and tcpNoDelay != transport.tcpNoDelay:
        transport.tcpNoDelay = tcpNoDelay

    if perMessageCompressionOffers is not None and pickle.dumps(perMessageCompressionOffers) != pickle.dumps(transport.perMessageCompressionOffers):
        if type(perMessageCompressionOffers) == list:
            #
            # FIXME: more rigorous verification of passed argument
            #
            transport.perMessageCompressionOffers = copy.deepcopy(perMessageCompressionOffers)
        else:
            raise Exception("invalid type %s for perMessageCompressionOffers - expected list" % type(perMessageCompressionOffers))

    if perMessageCompressionAccept is not None and perMessageCompressionAccept != transport.perMessageCompressionAccept:
        transport.perMessageCompressionAccept = perMessageCompressionAccept

    if autoPingInterval is not None and autoPingInterval != transport.autoPingInterval:
        transport.autoPingInterval = autoPingInterval

    if autoPingTimeout is not None and autoPingTimeout != transport.autoPingTimeout:
        transport.autoPingTimeout = autoPingTimeout

    if autoPingSize is not None and autoPingSize != transport.autoPingSize:
        assert(type(autoPingSize) == float or type(autoPingSize) == int)
        assert(4 <= autoPingSize <= 125)
        transport.autoPingSize = autoPingSize


def chunks(data, SIZE=1000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in islice(it, SIZE)}
