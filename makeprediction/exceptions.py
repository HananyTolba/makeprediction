



class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if instance is used before fitting a time serie model.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    """

class NotKernelError(Exception):
    """Exception class to raise if kernel instance is unknowned.
    """

class LoadingError(Exception):
    """Exception class  to be raised if the model is non-existent or corrupted .
    """
class NotValidModelError(Exception):
    """Exception class to raise if  the input is not a valid  model.
    """