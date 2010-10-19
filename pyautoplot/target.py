


class Target:
    """
Contains information about a target.
    """
    name=''
    direction=None
    def __init__(self, name, direction):
        """ name: a string, direction: an angle.EquatorialDirection object. """
        self.name = name
        self.direction = direction
        pass
