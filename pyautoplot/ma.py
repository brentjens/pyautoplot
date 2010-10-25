try:
    from numpy.ma import *
except (ImportError, ):    # Use this construction to be backwards
    e = sys.exc_info()[1]  # compatible with Python 2.5.  Proper
                           # Python 2.6/2.7/3.0 is
                           # except ValueError as e:
                           # etc...
    from numpy.core.ma import *
