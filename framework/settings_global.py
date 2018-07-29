#from framework.generic_classes import BoltzmannFrameworkError


class Settings(object):
    """Holds global settings for the Boltzmann framework.

    """

    def __init__(self):
        #raise BoltzmannFrameworkError("This class is not meant to be instantiated")
        raise RuntimeError("This class is not meant to be instantiated")

    debug_mode = False