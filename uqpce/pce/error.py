
class Error(Exception):
    """
    Parent class for UQPCE errors.
    """
    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field

class VariableInputError(Error):
    """
    Inputs: message- the message to be printed when the error is raised

    Error raised for errors in Variable inputs.
    """

    def __init__(self, message="The UQPCE Variable cannot be created.", field=None):
        self.message = message
        super().__init__(self.message, field=None)

class DimensionError(Error):
    """
    Inputs: message- the message to be printed when the error is raised

    Error raised for errors in Variable inputs.
    """

    def __init__(self,  message="The UQPCE model dimensions are not correct.", field=None):
        self.message = message
        super().__init__(self.message, field=None)