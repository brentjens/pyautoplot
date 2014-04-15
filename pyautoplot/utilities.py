import os,sys
from numpy import isnan, concatenate, arange
import ma
import logging

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
    logging.debug('pyautoplot.utilities.set_nan_zero(%r)', data_array.shape)
    where_it_is_nan = isnan(data_array)
    if is_masked_array(data_array):
        data_array.mask[where_it_is_nan] = True
    data_array[where_it_is_nan] = 0
    return data_array


def printnow (s):
    print s
    sys.stdout.flush()
    pass


def map_casa_table(function, casa_table, column_name='DATA', flag_name='FLAG', chunksize=10000, rowincr=1, nrow=None, max_chunks=None):
    """function should take a complex array of (timeslots,channels,polarizations) dimension, and return an array of values
    per timeslot. """
    chunksize=chunksize-(chunksize % rowincr)
    selection = casa_table
    nrows = selection.nrows()
    if nrow is not None:
        nrows = min(nrow, nrows)
    selection = selection.selectrows(arange(0, nrows, rowincr))
    nrows = selection.nrows()
    lastset = nrows % chunksize
    complete_chunks = nrows / chunksize
    results = []
    for chunk in range(complete_chunks):
        if max_chunks:
            if chunk >= max_chunks:
                break
        logging.info('%d -- %d / %d', chunk*chunksize+1, (chunk+1)*chunksize, nrows)
        results += [function(set_nan_zero(ma.array(selection.getcol(column_name, startrow=chunk*chunksize, nrow=chunksize),
                                                   mask=selection.getcol(flag_name, startrow=chunk*chunksize, nrow=chunksize))))]
    if max_chunks is None or (chunk == max_chunks-1):
        logging.info('%d -- %d / %d', complete_chunks*chunksize+1, nrows, nrows)
        results += [function(set_nan_zero(ma.array(selection.getcol(column_name,startrow=complete_chunks*chunksize, nrow=lastset),
                                                   mask=selection.getcol(flag_name,startrow=complete_chunks*chunksize, nrow=lastset))))]
    return concatenate(results, axis=0)
