from pyrap import tables as tables
from pylab import *
import ma # masked arrays
import scipy.ndimage as ndimage
from exceptions import *
from angle import *
import uvplane
import os




def is_list(obj):
    return type(obj) == type([])

def full_path_listdir(directory):
    return map(lambda d: directory+d, os.listdir(directory))

class NotImplementedError(Exception):
    pass





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





class TableFormatter:
    """Implements text based formatting for tables in for HTML and
    TXT. A table consists of a list of lists, where each sub list
    contains one row. The first row is considered the header, and
    should only contain string elements."""

    formats = {'html': {'table_start' : '<table>',
                        'table_end'   : '</table>',
                        'head_start'  : '<th><td>',
                        'head_end'    : '</td></th>',
                        'line_start'  : '<tr><td>',
                        'line_end'    : '</td></tr>',
                        'cell_sep'    : '</td><td>'},
               
               'txt': {'table_start' : '',
                       'table_end'   : '--------------------------------------------------------------------------------',
                       'head_start'  : '================================================================================',
                       'head_end'    : '--------------------------------------------------------------------------------',
                       'line_start'  : '',
                       'line_end'    : '',
                       'cell_sep'    : ' '}}
    
    def __init__(self, table_as_list, format='txt', col_widths=15, cell_formatters=str):
        self.header = table_as_list[0]
        self.data   = table_as_list[1:]
        
        self.default_format          = format
        self.default_col_widths      = col_widths
        self.default_cell_formatters = cell_formatters
        pass
    
    def __repr__(self):
        return self.__str__()


    def __str__(self):
        return self.format(self.default_format, self.default_col_widths, self.default_cell_formatters)


    def format(self,format='txt', col_widths=15, cell_formatters=str):
        f = self.formats[format]
        if not is_list(col_widths):
            col_widths = [col_widths]*len(self.header)
        if not is_list(cell_formatters):
            cell_formatters = [cell_formatters]*len(self.header)
        return '\n'.join([f['table_start'],
                          f['head_start'],
                          f['cell_sep'].join(map(lambda x,w:str(x).ljust(w), self.header, col_widths)),
                          f['head_end']]+
                         [f['line_start']+f['cell_sep'].join(map(lambda x,w,fmt: fmt(x).ljust(w), row, col_widths, cell_formatters))+f['line_end'] for row in self.data]+
                         [f['table_end']])
    pass





class AutocorrelationStatistics:
    def __init__(self,antenna_number):
        self.antenna_number = antenna_number
        
        pass
    pass





class MeasurementSetSummary:
    msname = ''
    times  = []
    mjd_start = 0.0
    mjd_end   = 0.0
    duration_seconds  = 0.0
    integration_times = []

    tables           = {}

    channel_frequencies=[]
    subband_frequencies=[]
    
    def __init__(self, msname):
        self.msname = msname
        self.read_metadata()
        pass

    def subtable(self, subtable_name):
        return tables.table(tables.table(self.msname).getkeyword(subtable_name))

    def read_subtable(self, subtable_name, columns=None):
        subtab = self.subtable(subtable_name)
        colnames = subtab.colnames()
        if columns is not None:
            colnames = columns
        cols = [subtab.getcol(col) for col in colnames]
        return [['ID']+colnames]+[[i]+[col[i] for col in cols]  for i in range(subtab.nrows())]

    def read_metadata(self):
        ms = tables.table(self.msname)
        self.times = unique(ms.getcol('TIME'))
        self.mjd_start = self.times.min()
        self.mjd_end   = self.times.max()
        self.duration_seconds  = self.mjd_end - self.mjd_start
        self.integration_times = unique(ms.getcol('EXPOSURE'))
        
        self.tables['targets']  = TableFormatter(
            self.read_subtable('FIELD',
                               ['NAME','REFERENCE_DIR']),
            col_widths=[5,20,30],
            cell_formatters=[str,
                             str,
                             lambda x:
                                 str(EquatorialDirection(RightAscension(x[0,0]),
                                                         Declination(x[0,1])))])
        self.tables['antennae'] = TableFormatter(self.read_subtable('ANTENNA',['NAME', 'POSITION']),
                                       col_widths=[5,15,40])
        self.tables['spectral_windows'] = TableFormatter(self.read_subtable('SPECTRAL_WINDOW',
                                                                  ['REF_FREQUENCY',
                                                                   'TOTAL_BANDWIDTH',
                                                                   'NUM_CHAN']),
                                               col_widths=[5,15,18,8])
        self.channel_frequencies = self.subtable('SPECTRAL_WINDOW').getcol('CHAN_FREQ')
        self.subband_frequencies  = self.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')
        pass
    
    def baseline(self, ant1, ant2, column='DATA', subband=0, **kwargs):
        selection = tables.table(self.msname).query('ANTENNA1 == %d && ANTENNA2 == %d || ANTENNA1 == %d && ANTENNA2 == %d && DATA_DESC_ID == %d' % (ant1,ant2,ant2,ant1, subband))
        data=selection.getcol(column, **kwargs)
        data[:,0,:] = 0.0
        
        flags=selection.getcol('FLAG', **kwargs)
        return  ma.array(data,mask=flags)

    
    def statistics(self):
        pass









def cmp_ms_freq(ms_a, ms_b):
    return apply(cmp, [ms.subband_frequencies.min() for ms in [ms_a, ms_b]])



def split_data_col(data_col):
    """Returns xx, xy, yx, yy, num_pol"""
    flags=logical_or.reduce(data_col.mask, axis=2)
    
    return (ma.array(data_col[:,:,0], mask=flags),
            ma.array(data_col[:,:,1], mask=flags),
            ma.array(data_col[:,:,2], mask=flags),
            ma.array(data_col[:,:,3], mask=flags),
            data_col.shape[-1])

def single_correlation_flags(tf_plane, threshold=5.0, max_iter=5, previous_sums=[]):
    flags    = tf_plane.mask
    if max_iter <= 0:
        return ndimage.binary_dilation(flags,iterations=2)
    med       = ma.median(tf_plane.real) +1j*ma.median(tf_plane.imag)
    sigma     = max(ma.std(tf_plane), ma.std(tf_plane.imag))
    bad_data  = abs(tf_plane.data-med) > threshold*sigma
    new_flags = logical_or(flags,bad_data)
    new_data  = ma.array(tf_plane.data, mask=new_flags)
    sum_flags = new_flags.sum()
    print sum_flags
    print     '%5.3f%s flagged\nstd: %6.4f' % ((sum_flags*100.0/product(tf_plane.shape)),'%', ma.std(new_data))

    print     sum_flags
    print     previous_sums
    print     '------------------------------------------------------------'
    if sum_flags == reduce(max, previous_sums, 0):
        return single_correlation_flags(new_data, threshold=threshold, max_iter=0, previous_sums=previous_sums+[sum_flags])
    else:
        return single_correlation_flags(new_data, threshold=threshold, max_iter=max_iter-1, previous_sums=previous_sums+[sum_flags])
    
    
def bad_data(data_col, threshold=5.0,max_iter=5, fubar_fraction=0.5):
    xx,xy,yx,yy,num_pol = split_data_col(data_col)
    flags = reduce(logical_or,
                   map(lambda x: single_correlation_flags(x,threshold=threshold,max_iter=max_iter),
                       [xx, xy, yx, yy]))
    bad_channels  = ma.sum(flags,axis=0) > data_col.shape[0]*fubar_fraction
    bad_timeslots = ma.sum(flags,axis=1) > data_col.shape[1]*fubar_fraction

    flags |= logical_or(bad_channels[newaxis,:], bad_timeslots[:,newaxis])
    

    full_flags=zeros(data_col.shape, dtype=bool)
    for i in range(4):
        full_flags[:,:,i] = flags
        pass
    return full_flags
    

def flag_data(data_col, **kwargs):
    return ma.array(data_col, mask=bad_data(data_col,**kwargs))


def statistics(array):
    return {'mean'   : ma.mean(array),
            'median' : ma.median(array.real)+1j*ma.median(array.imag),
            'max'    : ma.max(abs(array)),
            'min'    : ma.min(abs(array)),
            'std'    : ma.std(array),
            'stdmean': ma.std(array)/sqrt(sum(logical_not(array.mask))-1)}


def all_statistics(data_col):
    xx,xy,yx,yy,num_pol = split_data_col(data_col)
    
    return {'xx': statistics(xx),
            'xy': statistics(xy),
            'yx': statistics(yx),
            'yy': statistics(yy),
            'flagged': xx.mask.sum()*1.0/product(xx.shape),
            'num_pol': num_pol}



class Flagger:
    def __init__(self, msname):
        self.ms=MeasurementSetSummary(msname)
        pass
    pass



def plot_complex_image(plot_title, image, good_data=None, amin=None, amax=None, scaling_function=None):
    title(plot_title)
    xlabel('Channel', fontsize=16)
    ylabel('Timeslot', fontsize=16)
    rgb = uvplane.rgb_from_complex_image(image,amin=amin, amax=amax,scaling_function=scaling_function)
    if good_data is not None:
        rgb *= good_data[:,:,newaxis]
        rgb[:,:,2] += 1.0-good_data
        pass
    imshow(rgb,interpolation='nearest')
    pass
    

def plot_all_correlations(data_col, plot_flags=True):
    flags = bad_data(data_col, threshold=4.0, max_iter=20)
    flagged_data = ma.array(data_col.data, mask=flags)
    xx,xy,yx,yy,num_pol = split_data_col(ma.array(flagged_data))
    
    scale=ma.max(abs(flagged_data))
    stddev = max(ma.std(flagged_data.real), ma.std(flagged_data.imag))
    amax=scale-stddev

    print 'scale: %f\nsigma: %f' % (scale, stddev)
    good=logical_not(xx.mask)
    if not plot_flags:
        good = None
    clf()
    if num_pol is 2:
        subplot(121)
        plot_complex_image('XX',xx, good, amin=0.0, amax=amax)
        subplot(122)
        plot_complex_image('YY',yy, good, amin=0.0, amax=amax)
    elif num_pol is 4:
        subplot(141)
        plot_complex_image('XX',xx, good, amin=0.0, amax=amax)
        subplot(142)
        plot_complex_image('XY',xy, good, amin=0.0, amax=amax)
        subplot(143)
        plot_complex_image('YX',yx, good, amin=0.0, amax=amax)
        subplot(144)
        plot_complex_image('YY',yy, good, amin=0.0, amax=amax)
        pass
    pass
    

def delay_fringe_rate(tf_plane,padding=1):
    nt,nf = tf_plane.shape
    padded_plane=zeros((nt,padding*nf),dtype=complex64)
    padded_plane[:,(padding/2):(padding/2+nf)] = tf_plane*logical_not(tf_plane.mask)
    return fftshift(ifft2(padded_plane))





def plot_baseline(ms_summary, baseline, plot_flags=True,padding=1,timeslots=None):
    flagged_data = flag_data(apply(ms_summary.baseline, baseline), threshold=4.0, max_iter=20)
    xx,xy,yx,yy,num_pol = split_data_col(ma.array(flagged_data))


    scale=ma.max(abs(flagged_data))
    stddev = max(ma.std(flagged_data.real), ma.std(flagged_data.imag))
    amax=scale-stddev

    print 'scale: %f\nsigma: %f' % (scale, stddev)
    good=logical_not(xx.mask)
    if not plot_flags:
        good = None
    names = ['XX', 'XY', 'YX', 'YY']

    clf()
    t = gcf().text(0.5,
                   0.95, '%s: %6.3f MHz' % (', '.join(ms_summary.msname.split('/')[-2:]), ms_summary.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')[0]/1e6),
                   horizontalalignment='center',
                   fontsize=24)

    for i,name,data in zip(range(4), names, [xx, xy, yx, yy]):
        subplot(241+i)
        plot_complex_image(name, data, good, amin=0.0, amax=amax)
        pass
    
    if timeslots is None:
        plots = map(lambda tf: delay_fringe_rate(tf,padding=padding), [xx,xy,yx,yy])
    else:
        plots = map(lambda tf: delay_fringe_rate(tf[timeslots,:],padding=padding), [xx,xy,yx,yy])

    amax = array([abs(d).max() for d in plots]).mean()*1.2
    width=20
    height=40

    for i,d in enumerate(plots):
        subplot(245+i)
        xlabel('Delay',fontsize=16)
        ylabel('Fringe rate',fontsize=16)
        ny,nx = d.shape
        print d.shape
        imshow(abs(d)[ny/2-height/2:ny/2+height/2,nx/2-width/2:nx/2+width/2],interpolation='nearest',extent=(-(width/2) -0.5, -(width/2) + width-0.5, -(height/2) + height-0.5, -(height/2) -0.5),
               vmin=0.0,vmax=amax)
        grid()
        colorbar()
        pass
    pass




def fringe_rate_spectra(ms_summary, baseline):
    flagged_data = split_data_col(flag_data(ms_summary.baseline(*baseline),threshold=4.0, max_iter=20))[:-1]
    return [ma.std(x,axis=0) for x in flagged_data]
    #spectra = [abs(ifft(x*logical_not(x.mask),axis=0)).std(axis=0) for x in flagged_data]
    #weights = [logical_not(x.mask).mean(axis=0) for x in flagged_data]
    #return [ma.array(x/w,mask=(w==0.0)) for x,w in zip(spectra, weights)]

