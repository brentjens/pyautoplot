import os,sys
from numpy import isnan
import ma

def is_list(obj):
    return type(obj) == type([])

def is_masked_array(obj):
    return type(obj) == type(ma.array([]))

def full_path_listdir(directory):
    return map(lambda d: os.path.join(directory,d), os.listdir(directory))

def set_nan_zero(data_array):
    """
    Set any NaN values in *data_array* to zero and return result. This function modifies *data_array*.
    """
    where_it_is_nan = isnan(data_array)
    if is_masked_array(data_array):
        data_array.mask[where_it_is_nan] = True
    data_array[where_it_is_nan] = 0
    return data_array


def printnow (s):
    print s
    sys.stdout.flush()
    pass
