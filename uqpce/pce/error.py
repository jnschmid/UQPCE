class Error(Exception):
    """
    Parent class for UQPCE errors.
    """
    pass


class VariableInputError(Error):
    """
    Inputs: message- the message to be printed when the error is raised

    Error raised for errors in Variable inputs.
    """

    def __init__(self, message="The UQPCE Variable cannot be created."):
        self.message = message
        super().__init__(self.message)

class DimensionError(Error):
    """
    Inputs: message- the message to be printed when the error is raised

    Error raised for errors in Variable inputs.
    """

    def __init__(self,  message="The UQPCE model dimensions are not correct."):
        self.message = message
        super().__init__(self.message)
