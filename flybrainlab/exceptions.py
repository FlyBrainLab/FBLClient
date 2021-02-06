


class FlyBrainLabClientException(Exception):
    pass

class FlyBrainLabVersionMismatchException(Exception):
    pass

class FlyBrainLabBackendException(Exception):
    pass


class FlyBrainLabNAserverException(FlyBrainLabBackendException):
    pass

class FlyBrainLabNLPserverException(FlyBrainLabBackendException):
    pass

class FlyBrainLabNKserverException(FlyBrainLabBackendException):
    pass
