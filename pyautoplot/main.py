from pyautoplot import __version__

#from exceptions import *
import os,sys,gc,pickle
import scipy.ndimage as ndimage

try:
    import pyrap.tables as tables
except ImportError:
    print(sys.exc_info()[1])
    print('No problem during setup. During normal use, ensure pyrap is in PYTHONPATH')

from numpy import complex64, float32, float64, int64
from numpy import arange, concatenate, logical_not, logical_or, newaxis, product
from numpy import log10, conj, ceil, ones, sqrt, unique, zeros, where, median
from numpy.fft import fft, fftshift, ifft, ifft2

from pylab import clf, gcf, imshow, axis, grid, title, xlabel, ylabel, plot
from pylab import colorbar, figure, Figure, scatter, legend, savefig, yscale
from pylab import rcParams, subplot
import matplotlib.cm as cm

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import FigureCanvasBase

try:
    import pyautoplot.ma      as ma
    import pyautoplot.forkmap as forkmap
    import pyautoplot.uvplane as uvplane
    from pyautoplot.tableformatter import *
    from pyautoplot.angle import *
    from pyautoplot.utilities import *
    from pyautoplot.lofaroffline import *
except:
    import ma      as ma
    import forkmap as forkmap
    import uvplane as uvplane
    from tableformatter import *
    from angle import *
    from utilities import *
    from lofaroffline import *


def corr_type(corr_id):
    CORR_TYPE=array(['Undefined',
                     'I', 'Q', 'U', 'V',
                     'RR', 'RL', 'LR', 'LL',
                     'XX', 'XY', 'YX', 'YY',
                     'RX', 'RY', 'LX', 'LY',
                     'XR', 'XL', 'YR', 'YL',
                     'PP', 'PQ', 'QP', 'QQ',
                     'RCircular', 'LCircular',
                     'Linear',
                     'Ptotal', 'Plinear',
                     'PFtotal', 'PFlinear', 'Pangle'])
    if is_list (corr_id):
        return [corr_type(corr) for corr in corr_id]
    else:
        return CORR_TYPE[corr_id]

    


def split_data_col(data_col):
    """Returns xx, xy, yx, yy, num_pol"""
    flags = logical_or.reduce(data_col.mask, axis = 2)
    
    return (ma.array(data_col[:,:,0], mask=flags),
            ma.array(data_col[:,:,1], mask=flags),
            ma.array(data_col[:,:,-2], mask=flags),
            ma.array(data_col[:,:,-1], mask=flags),
            data_col.shape[-1])

def single_correlation_flags(tf_plane, threshold=5.0, max_iter=5, previous_sums=[], verbose=False):
    flags    = tf_plane.mask
    sum_flags=flags.sum()
    if verbose:
        print('sum(flags): %s' % (sum_flags,))
        print('%5.3f%s flagged\n' % ((sum_flags*100.0/product(tf_plane.shape)),'%'))
    if sum_flags == product(flags.shape):
        return flags
    if max_iter <= 0:
        return ndimage.binary_dilation(flags,iterations=2)
    med       = ma.median(tf_plane.real) +1j*ma.median(tf_plane.imag)
    sigma     = sqrt(ma.std(tf_plane.real)**2 + ma.std(tf_plane.imag)**2)
    bad_vis   = abs(tf_plane.data-med) > threshold*sigma
    new_flags = logical_or(flags, bad_vis)
    new_data  = ma.array(tf_plane.data, mask=new_flags)
    sum_flags = new_flags.sum()
    if verbose:
        print('sum_flags: %s' % (sum_flags,))
        print('%5.3f%s flagged\nstd: %6.4f' % ((sum_flags*100.0/product(tf_plane.shape)),'%', ma.std(new_data)))
        print(sum_flags)
        print(previous_sums)
        print('------------------------------------------------------------')
    if sum_flags == reduce(max, previous_sums, 0):
        return single_correlation_flags(new_data,
                                        threshold = threshold,
                                        max_iter  = 0,
                                        previous_sums = previous_sums+[sum_flags])
    else:
        return single_correlation_flags(new_data, threshold=threshold, max_iter=max_iter-1, previous_sums=previous_sums+[sum_flags])
    
    
def bad_data(data_col, threshold=5.0,max_iter=5, fubar_fraction=0.5, verbose=False):
    xx, xy, yx, yy, num_pol = split_data_col(data_col)
    flags = reduce(logical_or,
                   map(lambda x: single_correlation_flags(x,threshold=threshold,max_iter=max_iter, verbose=verbose),
                       [xx, xy, yx, yy]))
    bad_channels  = ma.sum(flags,axis=0) > data_col.shape[0]*fubar_fraction
    bad_timeslots = ma.sum(flags,axis=1) > data_col.shape[1]*fubar_fraction

    flags |= logical_or(bad_channels[newaxis, :], bad_timeslots[:, newaxis])
    

    full_flags = zeros(data_col.shape, dtype = bool)
    for i in range(4):
        full_flags[:,:,i] = flags
    return full_flags
    

def flag_data(data_col, **kwargs):
    return ma.array(data_col, mask=bad_data(data_col,**kwargs))


def statistics(numpy_array):
    return {'mean'   : ma.mean(numpy_array),
            'median' : ma.median(numpy_array.real)+1j*ma.median(numpy_array.imag),
            'max'    : ma.max(abs(array)),
            'min'    : ma.min(abs(array)),
            'std'    : ma.std(array),
            'stdmean': ma.std(numpy_array)/sqrt(sum(logical_not(numpy_array.mask))-1)}


def all_statistics(data_col):
    xx,xy,yx,yy,num_pol = split_data_col(data_col)
    
    return {'xx': statistics(xx),
            'xy': statistics(xy),
            'yx': statistics(yx),
            'yy': statistics(yy),
            'flagged': xx.mask.sum()*1.0/product(xx.data.shape),
            'num_pol': num_pol}


def regrid_time_frequency_correlation_cube(cube, time_slots):
    new_cube=ma.array(zeros((max(time_slots)-min(time_slots)+1,cube.shape[1],cube.shape[2]),dtype=complex64))
    new_cube.mask=ones(new_cube.data.shape,dtype=bool)
    new_cube.data[time_slots-min(time_slots),:,:] = cube.data
    new_cube.mask[time_slots-min(time_slots),:,:] = cube.mask
    return new_cube



def hanning(n):
    c = zeros(n,dtype=float32)
    c[0]=0.5
    c[1]=0.25
    c[-1]=0.25
    return fftshift(fft(c))

def apply_taper(complex_array, taper):
    # Use fft for vis -> lag and ifft for lag -> vis
    return ifft(fft(complex_array, axis=1)*(fftshift(taper)[newaxis,:,newaxis]),axis=1)

def ra_dec_formatter(ra_dec):
    """
    *ra_dec* is a 2D array containing RA, dec pairs
    """
    return str(EquatorialDirection(RightAscension(ra_dec[0,0]),
                                   Declination(ra_dec[0,1])))

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
    
    endian_swap=False
    
    def __init__(self, msname, endian_swap=False):
        self.msname = msname
        self.read_metadata()
        self.endian_swap=endian_swap

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
        self.times = array(sorted(unique(ms.getcol('TIME'))))
        self.mjd_start = self.times.min()
        self.mjd_end   = self.times.max()
        self.duration_seconds  = self.mjd_end - self.mjd_start
        self.integration_times = unique(ms.getcol('EXPOSURE'))
        
        self.tables['targets']  = TableFormatter(
            self.read_subtable('FIELD',
                               ['NAME','REFERENCE_DIR']),
            col_widths=[5,20,30],
            cell_formatters=[str, str, ra_dec_formatter])
        self.tables['antennae'] = TableFormatter(self.read_subtable('ANTENNA',['NAME', 'POSITION']),
                                       col_widths=[5,15,40])
        self.tables['spectral_windows'] = TableFormatter(self.read_subtable('SPECTRAL_WINDOW',
                                                                  ['REF_FREQUENCY',
                                                                   'TOTAL_BANDWIDTH',
                                                                   'NUM_CHAN']),
                                               col_widths=[5,15,18,8])
        self.tables['polarization'] = TableFormatter(self.read_subtable('POLARIZATION',
                                                                        ['CORR_TYPE']),
                                                     col_widths=[5,15])
        self.channel_frequencies = self.subtable('SPECTRAL_WINDOW').getcol('CHAN_FREQ')
        self.subband_frequencies  = self.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')
        pass


    def baseline_table(self, ant1, ant2, subband=0):
        print('MeasurementSetSummary.baseline_table subband: '+str(subband))
        print('(ANTENNA1 == %d && ANTENNA2 == %d || ANTENNA1 == %d && ANTENNA2 == %d) && DATA_DESC_ID == %d' % (ant1,ant2,ant2,ant1, subband))
        return tables.table(self.msname).query('(ANTENNA1 == %d && ANTENNA2 == %d || ANTENNA1 == %d && ANTENNA2 == %d) && DATA_DESC_ID == %d' % (ant1,ant2,ant2,ant1, subband))

    
    def baseline(self, ant1, ant2, column='DATA', subband=0, taper=None, **kwargs):
        """Returns a tuple with the time,frequency,correlation masked
        array cube of the visibility on baseline ant1-ant2 for
        subband, as well as an array containing the time slot number
        of each row/plane in the cube. The mask of the cube is the
        contents of the 'FLAG' column. The second element of the tuple
        is the time slot number of each row in the cube."""
        print('MeasurementSetSummary.baseline subband: '+str(subband))
        selection = self.baseline_table(ant1, ant2, subband=subband)
        data=selection.getcol(column, **kwargs)
        if self.endian_swap:
            data.byteswap(True)
            pass
        data=set_nan_zero(data)
        time_centroids = selection.getcol('TIME_CENTROID', **kwargs)
        time_slots     = array((time_centroids - min(self.times))/self.integration_times[0] +0.5, dtype=int64)

        flags=selection.getcol('FLAG', **kwargs)
        raw = regrid_time_frequency_correlation_cube(ma.array(data,mask=flags),time_slots)
        if taper is not None:
            return ma.array(apply_taper(raw.data, taper(raw.data.shape[1])),mask=raw.mask)
        else:
            return raw


    def map_flagged_baseline(self, ant1, ant2, function, chunksize=1000, rowincr=1,nrow=None):
        """function should take a complex array of (timeslots,channels,polarizations) dimension, and return an array of values
        per timeslot. """
        chunksize=chunksize-(chunksize % rowincr)
        selection = self.baseline_table(ant1, ant2)
        nrows = selection.nrows()
        selection = selection.selectrows(arange(0,nrows, rowincr))
        nrows = selection.nrows()
        if nrow is not None:
            nrows = min(nrow, nrows)
        lastset = nrows % chunksize
        complete_chunks = nrows / chunksize
        results = []
        for chunk in range(complete_chunks):
            print('%d -- %d / %d' % (chunk*chunksize+1, (chunk+1)*chunksize, nrows))
            results += [function(flag_data(self.baseline(ant1,ant2,startrow=chunk*chunksize, nrow=chunksize),threshold=4.0, max_iter=10))]
            pass
        print('%d -- %d / %d' % (complete_chunks*chunksize+1, nrows, nrows))
        results += [function(flag_data(self.baseline(ant1,ant2,startrow=complete_chunks*chunksize, nrow=lastset), threshold=4.0, max_iter=10))]
        return concatenate(results, axis=0)
        

    def map_baseline(self, ant1, ant2, function, chunksize=1000, rowincr=1, nrow=None):
        """function should take a complex array of (timeslots,channels,polarizations) dimension, and return an array of values
        per timeslot. """
        chunksize=chunksize-(chunksize % rowincr)
        selection = self.baseline_table(ant1, ant2)
        nrows = selection.nrows()
        if nrow is not None:
            nrows = min(nrow, nrows)
        selection = selection.selectrows(arange(0,nrows, rowincr))
        nrows = selection.nrows()
        lastset = nrows % chunksize
        complete_chunks = nrows / chunksize
        results = []
        for chunk in range(complete_chunks):
            print('%d -- %d / %d' % (chunk*chunksize+1, (chunk+1)*chunksize, nrows))
            results += [function(set_nan_zero(selection.getcol('DATA', startrow=chunk*chunksize, nrow=chunksize)))]
            pass
        print('%d -- %d / %d' % (complete_chunks*chunksize+1, nrows, nrows))
        results += [function(set_nan_zero(selection.getcol('DATA',startrow=complete_chunks*chunksize, nrow=lastset)))]
        return concatenate(results, axis=0)

    
    def statistics(self):
        pass





def cmp_ms_freq(ms_a, ms_b):
    """Compare the minimum suband frequencies of ms_a and ms_b with
    the function cmp()"""
    return apply(cmp, [ms.subband_frequencies.min() for ms in [ms_a, ms_b]])








def plot_complex_image(plot_title, image, good_data=None, amin=None, amax=None, scaling_function=None):
    title(plot_title)
    xlabel('Channel', fontsize=16)
    ylabel('Timeslot', fontsize=16)
    if is_masked_array(image):
        img = image.data
    else:
        img = image
    rgb = uvplane.rgb_from_complex_image(img,amin=amin, amax=amax,scaling_function=scaling_function)
    if good_data is not None:
        rgb *= good_data[:,:,newaxis]
        rgb[:,:,2] += logical_not(good_data)
        pass
    imshow(rgb,interpolation='nearest')
    axis('tight')
    pass
    

def plot_all_correlations(data_col, plot_flags=True,amax_factor=1.0):
    flags = bad_data(data_col, threshold=4.0, max_iter=20)
    flagged_data = ma.array(data_col.data, mask=flags)
    xx,xy,yx,yy,num_pol = split_data_col(ma.array(flagged_data))
    
    scale=ma.max(abs(flagged_data))
    stddev = max(ma.std(flagged_data.real), ma.std(flagged_data.imag))
    if flags.sum() == product(flags.shape):
        amax=1.0
    else:
        amax=(scale-stddev)*amax_factor
    

    print('scale: %f\nsigma: %f' % (scale, stddev))
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
    if isnan(tf_plane).sum() > 0:
        ValueError('*tf_plane* contains NaN values. Please sanitize it before plotting using, e.g. tf_plane[isnan(tf_plane)] == 0.0, or pyautoplot.utilities.set_nan_zero(tf_plane)')
    nt,nf = tf_plane.shape
    padded_plane=zeros((nt,padding*nf),dtype=complex64)
    padded_plane[:,(padding/2):(padding/2+nf)] = tf_plane.data*logical_not(tf_plane.mask)
    return fftshift(ifft2(padded_plane))





def plot_baseline(ms_summary, baseline, plot_flags=True, padding=1, amax_factor=1.0, num_delay=80, num_fringe_rate=160,cmap=cm.hot, subband=0, taper=None, column='DATA', apply_flagger = True, **kwargs):
    """
    Plot time/frequency planes and fringerate/delay plots for  baseline (i,j).
    
    ms_summary : a MeasurementSetSummary instance
    baseline   : a tuple (ant_i, ant_j)
    padding    : if larger than one, increase the fringe rate / delay resolution
    amax_factor: scale the maximum displayable amplitude with this factor
    startrow   : first timeslot to plot
    nrow       : number of time slots to plot
    rowincr    : take every rowincr th timeslot
    """
    print('plot_baseline subband: '+str(subband))
    data            = ms_summary.baseline(baseline[0], baseline[1], subband=subband, taper=taper, column=column, **kwargs)
    if apply_flagger:
        flagged_data    = flag_data(data, threshold=5.0, max_iter=20)
    else:
        flagged_data    = data
        
    pp,pq,qp,qq,num_pol = split_data_col(ma.array(flagged_data))
    antenna_names   = array(ms_summary.subtable('ANTENNA').getcol('NAME'))[list(baseline)]
    print(antenna_names)

    if flagged_data.mask.sum() == product(flagged_data.data.shape):
        scale=1.0
        stddev=0.1
        amax=1.0*amax_factor
    else:
        means = array([ma.mean(x) for x in [pp,pq,qp,qq]])
        stddevs= [max(ma.std(x.real), ma.std(x.imag)) for x in [pp,pq,qp,qq]]
        scale=max(abs(means))
        stddev = max(stddevs)
        amax=(scale+2.5*stddev)*amax_factor

    
#    print('%f%% of time slots available' % (int((max(ms_summary.times[kwargs['startrow']:kwargs['startrow']+kwargs['nrow']*kwargs['rowincr']:kwargs['rowincr']]) - min(ms_summary.times[kwargs['startrow']:kwargs['startrow']+kwargs['nrow']*kwargs['rowincr']:kwargs['rowincr']]))/ms_summary.integration_times[0]+0.5)*100.0/len(time_slots),))
    print('scale: %f\nsigma: %f' % (scale, stddev))
    good=logical_not(pp.mask)
    if not plot_flags:
        good = None
    names = corr_type(ms_summary.tables['polarization']['CORR_TYPE'][0])

    clf()
    t = gcf().text(0.5,
                   0.95, '%s-%s %s: %6.3f MHz' % (antenna_names[0], antenna_names[1], ', '.join(ms_summary.msname.split('/')[-2:]), ms_summary.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')[subband]/1e6),
                   horizontalalignment='center',
                   fontsize=24)

    for i,name,data in zip(range(4), names, [pp,pq,qp,qq]):
        subplot(241+i)
        plot_complex_image(name, data, good, amin=0.0, amax=amax)
        pass
    
    plots = map(lambda tf: delay_fringe_rate(tf, padding=padding), [pp,pq,qp,qq])
    
    amax = array([abs(d).max() for d in plots]).max()*amax_factor
    width=min(num_delay, pp.data.shape[1])
    height=min(num_fringe_rate, pp.data.shape[0])

    for i,d in enumerate(plots):
        subplot(245+i)
        ny,nx = d.shape
        xlabel('Delay [samples]',fontsize=16)
        if i == 0:
            ylabel('Fringe rate [mHz]',fontsize=16)
        else:
            ylabel('')
            pass
        duration=ny*ms_summary.integration_times[0]
        imshow(abs(d)[ny/2-height/2:ny/2+height/2,nx/2-width/2:nx/2+width/2],
               interpolation='nearest',
               extent=(-(width/2) -0.5, -(width/2) + width-0.5, (-(height/2) -0.5)*1000/duration, (-(height/2) + height-0.5)*1000/duration),
               vmin=0.0,vmax=amax,
               aspect='auto',
               origin='lower',
               cmap=cmap)
        grid()
        if i == len(plots)-1:
            colorbar()
        pass
    pass




def fringe_rate_spectra(ms_summary, baseline):
    flagged_data = split_data_col(flag_data(ms_summary.baseline(*baseline),threshold=4.0, max_iter=20))[:-1]
    return [ma.std(x,axis=0) for x in flagged_data]
    #spectra = [abs(ifft(x*logical_not(x.mask),axis=0)).std(axis=0) for x in flagged_data]
    #weights = [logical_not(x.mask).mean(axis=0) for x in flagged_data]
    #return [ma.array(x/w,mask=(w==0.0)) for x,w in zip(spectra, weights)]



def vis_movie(file_prefix, timeseries, titles, maxamps, chunksize=60):
    full_chunks=len(timeseries[0])/chunksize
    rest=len(timeseries[0]) % chunksize
    figure(figsize=(6*len(titles),6),dpi=80)
    
    for chunk in range(full_chunks):
        clf()
        
        subsets = [ts[chunk*chunksize:(chunk+1)*chunksize,:] for ts in timeseries]
        for col,title_text,subset,maxamp in zip(range(len(titles)), titles, subsets,maxamps):
            print(col)
            subplot(100+10*len(titles)+col+1)
            title(title_text+' t = %4d min' % chunk)
            scatter(subset[:,0].real, subset[:,0].imag,c='blue',label='XX')
            scatter(subset[:,3].real, subset[:,3].imag,c='red', label='YY')
            xlabel('Real part')
            ylabel('Imaginary  part')
            axis([-maxamp,maxamp,-maxamp,maxamp])
            legend()
            grid()
            pass
        savefig('%s-%s.png' % (file_prefix, str(chunk).rjust(4,'0')))
        pass
    pass
                
        

def bl_mean(array):
    return ma.mean(array, axis=1)

def bl_median(array):
    return ma.median(array.real, axis=1)+1j*ma.median(array.imag, axis=1)

def bl_std(array):
    return ma.std(array.real,axis=1) +1j*ma.std(array.imag, axis=1)

def bl_mean_no_edges(array):
    return ma.mean(array[:,3:-3,:], axis=1)

def bl_median_no_edges(array):
    return ma.median(array.real[:,3:-3,:], axis=1)+1j*ma.median(array.imag[:,3:-3,:], axis=1)

def bl_std_no_edges(array):
    return ma.std(array.real[:,3:-3,:],axis=1) +1j*ma.std(array.imag[:,3:-3,:], axis=1)



def compute_baseline_stat(msname, bl_stat_function=bl_mean_no_edges, flag_data=False, rowincr=1):
    ms            = MeasurementSetSummary(msname)
    num_stations  = ms.subtable('ANTENNA').nrows()
    station_names = ms.subtable('ANTENNA').getcol('NAME')
    baselines     = [(i,j) for i in  range(num_stations) for j in range(i+1, num_stations)]
    if flag_data:
        return [('%s -- %s' % (station_names[i], station_names[j]), 
                 ms.map_flagged_baseline(i, j, bl_stat_function))
                for (i,j) in baselines]
    else:
        return [('%s -- %s' % (station_names[i], station_names[j]), 
                 ms.map_baseline(i, j, bl_stat_function, rowincr=rowincr))
                for (i,j) in baselines]
        

def compute_single_baseline_stat(msname, baseline, bl_stat_function=bl_mean_no_edges, flag_data=False, rowincr=1):
    ms            = MeasurementSetSummary(msname)
    num_stations  = ms.subtable('ANTENNA').nrows()
    station_names = ms.subtable('ANTENNA').getcol('NAME')
    i,j=baseline
    if flag_data:
        return ('%s -- %s' % (station_names[i], station_names[j]), 
                 ms.map_flagged_baseline(i, j, bl_stat_function))
    else:
        return ('%s -- %s' % (station_names[i], station_names[j]), 
                 ms.map_baseline(i, j, bl_stat_function, rowincr=rowincr))




def collect_all_subbands_stats(msnames, baseline):
    def stats_from_array(arr, bad=None):
        if bad is None:
            allstd=array([arr[:,1:-1,i].std() for i in range(arr.shape[-1])])
            return stats_from_array(arr, abs(arr) > 4*allstd[newaxis,newaxis,:])
        else:
            good_arr=ma.array(arr, mask=bad)
            good_std= array([good_arr[:,1:-1,i].std() for i in range(arr.shape[-1])])
            new_bad= abs(arr) > 4*good_std[newaxis,newaxis,:]
            if all(bad == new_bad):
                return {'all-std': [arr[:,1:-1,i].std() for i in range(arr.shape[-1])],
                        'good-std': [good_arr[:,1:-1,i].std() for i in range(arr.shape[-1])],
                        'flagged%': (100.0*bad.sum())/product(bad.shape)}
            else:
                return stats_from_array(arr, new_bad)
            
    def compute_stats(msname):
        ms=MeasurementSetSummary(msname)
        data=ms.baseline(*baseline)

        return {'name': msname, 'baseline': baseline,
                'real':stats_from_array(data.real), 'imag':stats_from_array(data.imag)}
    
    return forkmap.map(compute_stats, msnames,n=10*forkmap.nprocessors())


def subband_from_stats(sbs):
    name = sbs['name']
    return int(name[name.rfind('B')+1:-3])


def plot_stats(sbstats):
    subbands = map(subband_from_stats, sbstats)
    flagged = [max(sb['real']['flagged%'],sb['imag']['flagged%']) for sb in sbstats]
    astd    = [sb['real']['all-std'] for sb in sbstats]
    gstd    = [sb['real']['good-std'] for sb in sbstats]

    clf()
    
    subplot(311)
    title('Flagged')
    plot(subbands, flagged)
    xlabel('Subband number')
    
    subplot(312)
    title(r'$\sigma$ all')
    plot(subbands, astd)
    yscale('log')
    xlabel('Subband number')
    
    subplot(313)
    title(r'$\sigma$ good')
    plot(subbands, gstd)
    yscale('log')
    xlabel('Subband number')
    pass
    

def plot_baseline_stat(msname, bl_stat_function=lambda x: abs(bl_median(x)), title_text='Abs(median)', flag_data=False, plot_max=None, plot_min=None, rowincr=1):
    stats = compute_baseline_stat(msname, bl_stat_function, flag_data=flag_data, rowincr=rowincr)
    n = len(stats)
    clf()
    ms = MeasurementSetSummary(msname)
    t = gcf().text(0.5,
                   0.95, '%s %6.3f MHz: %s' % (', '.join(ms.msname.split('/')[-2:]), ms.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')[0]/1e6, title_text),
                   horizontalalignment='center',
                   fontsize=30)
    times = ms.times - ms.times[0]
    for (i,(caption, data)) in enumerate(stats):
        subplot(n*100+10+1+i)
        #title(caption,fontsize=10)
        ylabel(caption,rotation='horizontal')
        plot(times[::rowincr], data[:,0], label='XX')
        plot(times[::rowincr], data[:,1], label='XY')
        plot(times[::rowincr], data[:,2], label='YX')
        plot(times[::rowincr], data[:,3], label='YY')
        axis([None, None, plot_min, plot_max])
        legend()
        pass
    xlabel('Time [s]')
    pass



def ant_ant_stat_plot(fig, ax, title_text, single_pol_array, station_names, **kwargs):
    img = ax.imshow(single_pol_array, interpolation='nearest', **kwargs)
    ax.axis('equal')
    for y,n in enumerate(station_names):
        ax.text(y, -1,n, rotation=90, verticalalignment='bottom')
        ax.text(len(station_names), y, n, horizontalalignment='left',verticalalignment='center')
        pass
    fig.colorbar(img, pad=0.2, ax=ax)
    ax.set_xlabel(title_text)
    pass



def ant_ant_stat_frame(title_text, full_pol_array, station_names, output_name=None, dpi=50, **kwargs):
    if output_name is None:
        fig = figure(figsize=(32,30), dpi=dpi)
    else:
        fig=Figure(figsize=(32,30), dpi=dpi)

    if output_name is None:
        fig.suptitle(title_text, fontsize=30)
    else:
        fig.suptitle(title_text+' '+output_name, fontsize=30)
    
    ax1=fig.add_subplot(2,2,1)
    ant_ant_stat_plot(fig, ax1, title_text+' XX', full_pol_array[:,:,0], station_names, **kwargs)
    
    ax2=fig.add_subplot(2,2,2)
    ant_ant_stat_plot(fig, ax2, title_text+' XY', full_pol_array[:,:,1], station_names, **kwargs)
    
    ax3=fig.add_subplot(2,2,3)
    ant_ant_stat_plot(fig, ax3, title_text+' YX', full_pol_array[:,:,2], station_names, **kwargs)
    
    ax4=fig.add_subplot(2,2,4)
    ant_ant_stat_plot(fig, ax4, title_text+' YY', full_pol_array[:,:,3], station_names, **kwargs)

    if output_name is not None:
        canvas = FigureCanvasAgg(fig)
        if output_name[-4:] in ['.jpg', '.JPG']:
            canvas.print_jpg(output_name, dpi=dpi, quality=55)
        else:
            canvas.print_figure(output_name, dpi=dpi)
        pass
    pass




def collect_stats_ms(msname, max_mem_bytes=4*(2**30), first_timeslot=0, max_timeslots=None):
    ms = MeasurementSetSummary(msname)
    num_ant=len(ms.tables['antennae'])
    def timeslot(mjd):
        return int((mjd-ms.times[0])/ms.integration_times[0] +0.5)
    num_timeslots = timeslot(ms.times[-1])+1
    integration_time = array(ms.integration_times).mean()
    num_bl        = num_ant*(num_ant+1)/2
    num_chan      = ms.tables['spectral_windows']['NUM_CHAN'][0]
    bandwidth     = ms.tables['spectral_windows']['TOTAL_BANDWIDTH'][0]
    num_pol       = 4
    bytes_per_vis = 8
    fudge_factor  = 8
    max_mem_timeslots = max_mem_bytes/(bytes_per_vis*num_pol*num_chan*num_bl*fudge_factor)
    if max_timeslots is None:
        max_ts = max_mem_timeslots
    else:
        max_ts = min(max_mem_timeslots, max_timeslots)
        pass
    

    num_ts = min(max_ts, max(0,num_timeslots-first_timeslot))
    data_shape  = (num_ant, num_ant, num_pol, num_ts, num_chan)
    data = ma.array(zeros(data_shape, dtype=complex64),
                    mask=zeros(data_shape, dtype=bool))

    ms_table = tables.table(ms.msname)
    nrows = min(ms_table.nrows(), num_bl*num_ts)
    rows      = ms_table[0:nrows]
    xx,xy,yx,yy = 0,1,2,3
    printnow('reading data')
    for row in rows:
        ts = timeslot(row['TIME'])
        d = row['DATA']
        f = row['FLAG']
        for pol in xx,xy,yx,yy:
            data[row['ANTENNA1'], row['ANTENNA2'], pol, ts, :] = ma.array(d[:,pol], mask= f[:,pol])
            pass
        pass

    data=set_nan_zero(data)

    def where_true(a):
        indices = where(a)
        return tuple([x[0] for x in indices])

    def calc_bl_stat(fn):
        """
        result must return a complex float
        """
        result = zeros(data_shape[0:3], dtype=complex64)
        for ant1 in range(num_ant):
            for ant2 in range(num_ant):
                for pol in range(num_pol):
                    result[ant1, ant2, pol] = fn(data[ant1,ant2,pol,:,3:-3])
                    pass
                pass
            pass
        gc.collect()
        return result


    def calc_delay_rate_stats(data_array):
        peak       = zeros(data_shape[0:3], dtype=float64)
        peak_delay = zeros(data_shape[0:3], dtype=float64)
        peak_rate  = zeros(data_shape[0:3], dtype=float64)

        for ant1 in range(num_ant):
            printnow(str(ant1+1)+'/'+str(num_ant))
            for ant2 in range(num_ant):
                for pol in range(num_pol):
                    delay_rate=delay_fringe_rate(data_array[ant1,ant2,pol,:,:])
                    abs_delay_rate= abs(delay_rate)
                    max_abs       = abs_delay_rate.max()
                    peak[ant1,ant2,pol] = max_abs
                    rate_idx,delay_idx=where_true(abs_delay_rate == max_abs)
                    peak_delay[ant1,ant2,pol] = (delay_idx-num_chan/2)/bandwidth
                    peak_rate[ant1,ant2,pol] = (rate_idx-num_ts/2)/(integration_time*num_ts)
                    pass
                pass
            gc.collect()
            pass
        
        return peak, peak_delay, peak_rate
    
    printnow('computing std')
    bls_std    = calc_bl_stat(ma.std)
    printnow('computing mean')
    bls_mean   = calc_bl_stat(ma.mean)
    printnow('computing zeroes')
    bls_zeroes = calc_bl_stat(lambda x: (x==0.0+0.0j).sum()/product(x.shape))


    printnow('flagging iteration 1/2')
    deviant = abs(data - bls_mean[:,:,:,newaxis,newaxis]) > 4*bls_std[:,:,:,newaxis,newaxis]
    deviant_all_pol = reduce(logical_or, [deviant[:,:,0,:,:], deviant[:,:,1,:,:], deviant[:,:,2,:,:], deviant[:,:,3,:,:]])
    data.mask = logical_or(data.mask, deviant_all_pol[:,:,newaxis,:,:])
    deviant=None
    deviant_all_pol=None
    gc.collect()
    
    printnow('computing flagged mean')
    bls_flagged_mean   = calc_bl_stat(ma.mean)

    printnow('computing flagged std')
    bls_flagged_std    = calc_bl_stat(ma.std)

    printnow('flagging iteration 2/2')
    deviant = abs(data - bls_flagged_mean[:,:,:,newaxis,newaxis]) > 4*bls_flagged_std[:,:,:,newaxis,newaxis]
    deviant_all_pol = reduce(logical_or, [deviant[:,:,0,:,:], deviant[:,:,1,:,:], deviant[:,:,2,:,:], deviant[:,:,3,:,:]])
    data.mask = logical_or(data.mask, deviant_all_pol[:,:,newaxis,:,:])
    deviant=None
    deviant_all_pol=None
    gc.collect()


    printnow('computing flags')
    bls_flags  = calc_bl_stat(lambda x: x.mask.sum()*1.0/product(x.shape))

    printnow('counting unflagged points')
    bls_good   = calc_bl_stat(lambda x: logical_not(x.mask).sum())
    
    printnow('computing flagged mean')
    bls_flagged_mean   = calc_bl_stat(ma.mean)

    printnow('computing flagged std')
    bls_flagged_std    = calc_bl_stat(ma.std)

    #bls_sn = abs(bls_flagged_mean)/abs(bls_flagged_std/sqrt(bls_good-1))
    #bls_sn = set_nan_zero(bls_sn)

    #printnow('computing delay/rate')
    #peak, peak_delay, peak_rate = calc_delay_rate_stats(data)

    return {'Antennae'          : ms.tables['antennae'],
            'Spectral windows'  : ms.tables['spectral_windows'],
            'Targets'           : ms.tables['targets'],
            'Standard deviation': bls_std,
            'Mean'              : bls_mean,
            'Zeroes'            : bls_zeroes,
            'Good points'       : bls_good,
            #'Fringe SNR 0'      : bls_sn,
            'Flags'             : bls_flags,
            'Flagged mean'      : bls_flagged_mean,
            #'Delay'             : peak_delay,
            #'Rate'              : peak_rate,
            #'Fringe SNR'        : peak/abs(bls_flagged_std/sqrt(bls_good-1)),
            'Flagged standard deviation': bls_flagged_std}


def inspect_ms(msname, ms_id, max_mem_bytes=4*(2**30), root=os.path.expanduser('~/inspect/'), first_timeslot=0, max_timeslots=None, cmap=cm.gray_r, output_dir=None):
    extension = '.png'
    if 'print_jpg' in dir(FigureCanvasBase):
        extension = '.jpg'
        
    results = collect_stats_ms(msname, max_mem_bytes)

    ant_names=results['Antennae']['NAME']

    if output_dir is None:
        output_dir = os.path.join(root, str(ms_id))
    else:
        output_dir = os.path.join(root, output_dir)
        
    try:
        os.mkdir(output_dir)
    except Exception:
        pass
    
    def write_plot(quantity_name, scaling_function=lambda x: x, **kwargs):
        return ant_ant_stat_frame(quantity_name, scaling_function(abs(results[quantity_name])), ant_names,
                                  os.path.join(output_dir,msname.split('/')[-1][:-3]+'-'+quantity_name.lower().replace(' ','-')+extension),
                                  **kwargs)

    write_plot('Flagged mean', log10, vmax=8.0, vmin=4.0, cmap=cmap)
    write_plot('Flagged standard deviation', log10, vmax=8.0, vmin=4.0, cmap=cmap)
    write_plot('Flags', log10, vmax=0.0, vmin=-3.0, cmap=cmap)
    write_plot('Zeroes',lambda x: 100*x, vmin=0.0, vmax=100.0, cmap=cmap)
    # write_plot('Fringe SNR 0', log10, vmin=-1.0, vmax=4.0, cmap=cmap)

    # write_plot('Fringe SNR', log10, vmin=-1.0, vmax=4.0, cmap=cmap)
    # write_plot('Delay', lambda x:log10(abs(x)), vmin=-9.0, vmax=-3, cmap=cmap)
    # write_plot('Rate', lambda x:log10(abs(x)), vmin=-4, vmax=0.0, cmap=cmap)

    results_name=os.path.join(output_dir,msname.split('/')[-1][:-3]+'-data.pickle')
    pickle.dump(results, open(results_name, mode='w'), protocol=pickle.HIGHEST_PROTOCOL)

    ms = MeasurementSetSummary(msname)
    time_slots, vis_cube = collect_timeseries_ms(ms, num_points=240)

    good_stations = ['CS002HBA', 'CS002HBA0', 'CS002LBA', 'CS004HBA0', 'CS004LBA', 'CS004HBA',
                     'RS307HBA', 'RS307LBA', 'RS508LBA', 'RS508HBA']
    plot_stations = list(set(good_stations).intersection(set(ant_names)))
    if len(plot_stations) == 0:
        plot_stations = unique([ant_names[0], ant_names[len(ant_names)/2], ant_names[-1]])
    
    for station in plot_stations:
        filename = os.path.join(output_dir,msname.split('/')[-1][:-3]+'-timeseries-'+station.lower()+extension)
        timeseries_station_page(ms, station, time_slots, vis_cube, output_name=filename)
        filename = os.path.join(output_dir,msname.split('/')[-1][:-3]+'-station-gain-'+station.lower()+extension)
        station_gain_bar_chart(ms, station, time_slots, vis_cube, output_name=filename)
    
    return results




def collect_timeseries_ms(ms, num_points=240, subband=0, column_name='DATA'):
    mstab      = tables.table(ms.msname)
    num_ant    = len(ms.tables['antennae'])
    rowincr    = max(1, floor(len(ms.times)/num_points))
    time_slots = ms.times[0::rowincr]
    query_text = 'TIME in '+repr(list(time_slots))+' && DATA_DESC_ID == '+str(subband)

    print('Selecting data from '+ms.msname+' where '+query_text)
    selection  = mstab.query(query_text+' orderby '+(','.join(['TIME','ANTENNA1', 'ANTENNA2'])))
    print('done.')

    ant1       = selection.getcol('ANTENNA1')
    ant2       = selection.getcol('ANTENNA2')
    time_col   = selection.getcol('TIME')
    data_mean  = map_casa_table(bl_mean_no_edges, selection, chunksize=20000, column_name=column_name)
    print(data_mean.shape)
    output     = zeros((num_ant, num_ant, data_mean.shape[1], len(time_slots)), dtype=complex64)
    
    print('Allocated memory')
    time_slot_index = {}
    for i, ts in enumerate(time_slots):
        time_slot_index[repr(ts)] = i

    
    print('Beginning gridding')
    for (i,(a1,a2,mjds, data)) in enumerate(zip(ant1, ant2, time_col, data_mean)):
        if i%100000 == 0:
            print(i)
        ts = time_slot_index[repr(mjds)]
        output[a1,a2,:,ts]=data
        output[a2,a1,:,ts]=conj(data)
        pass
    
    return (time_slots, output)
    

def timeseries_station_page(ms, station_name, time_slots, data, fn=abs, output_name=None):
    dpi=50
    if output_name is None:
        fig = figure(figsize=(32,24), dpi=dpi)
    else:
        fig = Figure(figsize=(32,24), dpi=dpi)

    station_name_list = list(ms.tables['antennae']['NAME'])
    station_id        = station_name_list.index(station_name)
    num_ant           = len(ms.tables['antennae'])
    tsn               = time_slots-time_slots[0]
    pol_names         = corr_type(ms.tables['polarization']['CORR_TYPE'][0])
    ref_freq_mhz      = ms.tables['spectral_windows'][0]['REF_FREQUENCY']/1.e6

    fig.suptitle(ms.msname+': '+fn.__name__+'(vis) with '+station_name+' at %3.2f MHz' % (ref_freq_mhz,), fontsize='large')

    median_amp = ma.median(ma.mean(ma.median(fn(data[station_id,:,0::3,:]), axis=-1), axis=-1), axis=-1)
    
    for id2,name in enumerate(station_name_list):
        ax = fig.add_subplot(ceil(num_ant/4.0),4, id2+1)
        ax.plot(tsn, fn(data[station_id,id2,0,:]), c='blue'  , label=pol_names[0])
        ax.plot(tsn, fn(data[station_id,id2,1,:]), c='green' , label=pol_names[1])
        ax.plot(tsn, fn(data[station_id,id2,2,:]), c='purple', label=pol_names[2])
        ax.plot(tsn, fn(data[station_id,id2,3,:]), c='red'   , label=pol_names[3])
        ax.grid()
        ax.set_ylabel(station_name_list[id2], rotation='horizontal')
        ax.set_ylim(0.0, 3*median_amp)
        ax.set_yticklabels([])
        if id2 < len(station_name_list)-4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time [s]')    
        pass
    fig.subplots_adjust(hspace=0.0, top=0.95, bottom=0.04)
    if output_name is not None:
        canvas = FigureCanvasAgg(fig)
        if output_name[-4:] in ['.jpg', '.JPG']:
            canvas.print_jpg(output_name, dpi=dpi, quality=55)
        else:
            canvas.print_figure(output_name, dpi=dpi)
        pass
    pass




def station_gain_bar_chart(ms, station_name, time_slots, data, output_name= None):
    dpi=50
    if output_name is None:
        fig = figure(figsize=(38,24), dpi=dpi)
    else:
        fig = Figure(figsize=(38,24), dpi=dpi)

    station_name_list = list(ms.tables['antennae']['NAME'])
    num_stations      = len(station_name_list)
    station_id        = station_name_list.index(station_name)
    ref_freq_mhz      = ms.tables['spectral_windows'][0]['REF_FREQUENCY']/1.e6

    
    is_autocorrelation = array([station_name == name for name in ms.tables['antennae']['NAME']])

    noise = ma.array(data[station_id,:, 1:3,:].imag.std(axis=-1).mean(axis=-1),
                     mask = is_autocorrelation)
    sig    = median(abs(data[station_id, :, :, :]),axis=-1)
    signal = ma.array(sig, mask = is_autocorrelation[:, newaxis]*ones((num_stations, 4)))
    ax = fig.add_subplot(1,1,1)
    xx_bars = ax.bar(arange(len(station_name_list))-0.4, signal[:, 0], width=0.2, color='blue', label='xx')
    xy_bars = ax.bar(arange(len(station_name_list))-0.2, signal[:, 1], width=0.2, color='lightblue', label='xy')
    yx_bars = ax.bar(arange(len(station_name_list))    , signal[:, 2], width=0.2, color='lightpink', label='yx')
    yy_bars = ax.bar(arange(len(station_name_list))+0.2, signal[:, 3], width=0.2, color='red', label='yy')
    for x_pos, name  in enumerate(station_name_list):
        if name != station_name:
            ax.text(x_pos, signal[x_pos,:].max()*1.02, name, rotation='vertical',
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=25)
            ax.text(x_pos, signal[x_pos,:].max()*-0.01, name, rotation='vertical',
                    horizontalalignment='center', verticalalignment='top',
                    fontsize=25)
        else:
            ax.text(x_pos, 0.0, ' Reference station: '+name, rotation='vertical',
                    horizontalalignment='center', verticalalignment='bottom',
                    fontsize=25)

    ax.set_xlabel('Station', fontsize=40)
    ax.set_ylabel('Visibility amplitude', fontsize=40)
    ax.set_ylim(0, ma.max(signal)*1.2)
    ax.set_xlim(-1.0, num_stations)
    ax.set_xticklabels([])
    ax.set_title('%s:\nVis. amp. with station %s at %5.2f MHz' %
                 (ms.msname, station_name, ref_freq_mhz),
                 fontsize=40)
    old_legend_fontsize = rcParams['legend.fontsize']
    rcParams.update({'legend.fontsize': 25})
    legend_instance = ax.legend()
    
    
    for label in ax.get_yticklabels():
        label.set_fontsize(40)
    
    if output_name is not None:
        canvas = FigureCanvasAgg(fig)
        if output_name[-4:] in ['.jpg', '.JPG']:
            canvas.print_jpg(output_name, dpi=dpi, quality=55)
        else:
            canvas.print_figure(output_name, dpi=dpi)
        pass

    rcParams.update({'legend.fontsize': old_legend_fontsize})
