try:
    from numpy.ma import *
except (ImportError, ) as e:
    from numpy.core.ma import *
