from exceptions import *
from pyrap import tables as tables
from pylab import *
from numpy import *
import os,gc
from socket import gethostname
import forkmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import ma

import scipy.ndimage as ndimage
from angle import *
import uvplane


def is_compute_node(name=gethostname()):
    return len(name) == 6 and name[:3]=='lce' and name[3:].isdigit()

def compute_node_number(name=gethostname()):
    return int(name[3:])

def get_subcluster_number(compute_node_name=gethostname()):
    return int(floor((compute_node_number(compute_node_name)-1)/9)+1)

def get_compute_node_number_in_subcluster(compute_node_name=gethostname()):
    return compute_node_number(compute_node_name)-9*(get_subcluster_number(compute_node_name) -1) -1

def get_storage_node_names(subcluster_number):
    return ['lse'+str(i+(subcluster_number-1)*3).rjust(3,'0') for i in [1,2,3]]

def get_data_dirs(subcluster_number):
    storage_nodes=get_storage_node_names(subcluster_number)
    return ['/net/sub%s/%s/data%s/' %(subcluster_number, lse, data) for lse in storage_nodes for data in range(1,5)]

def find_msses(msname):
    result=[]
    for directory in get_data_dirs(get_subcluster_number()):
        result += [s.strip() for s in os.popen("find %s/%s -iname '*.MS'"%(directory,msname), 'r')]
    return result



def find_my_msses(msname):
    msnames=find_msses(msname)
    n_msnames=len(msses)
    n = int(ceil(float(n_msnames)/num_proc))
    proc_id=get_compute_node_number_in_subcluster()
    msses_here = msses[proc_id*n:min((proc_id+1)*n, n_msnames)]
    return msses_here


class NotImplementedError(Exception):
    pass









def split_data_col(data_col):
    """Returns xx, xy, yx, yy, num_pol"""
    flags=logical_or.reduce(data_col.mask, axis=2)
    
    return (ma.array(data_col[:,:,0], mask=flags),
            ma.array(data_col[:,:,1], mask=flags),
            ma.array(data_col[:,:,-2], mask=flags),
            ma.array(data_col[:,:,-1], mask=flags),
            data_col.shape[-1])

def single_correlation_flags(tf_plane, threshold=5.0, max_iter=5, previous_sums=[], verbose=False):
    flags    = tf_plane.mask
    sum_flags=flags.sum()
    if verbose:
        print 'sum(flags): %s' % (sum_flags,)
        print     '%5.3f%s flagged\n' % ((sum_flags*100.0/product(tf_plane.shape)),'%')
    if sum(flags) == product(flags.shape):
        return flags
    if max_iter <= 0:
        return ndimage.binary_dilation(flags,iterations=2)
    med       = ma.median(tf_plane.real) +1j*ma.median(tf_plane.imag)
    sigma     = max(ma.std(tf_plane.real), ma.std(tf_plane.imag))
    bad_data  = abs(tf_plane.data-med) > threshold*sigma
    new_flags = logical_or(flags,bad_data)
    new_data  = ma.array(tf_plane.data, mask=new_flags)
    sum_flags = new_flags.sum()
    if verbose:
        print 'sum_flags: %s' % (sum_flags,)
        print     '%5.3f%s flagged\nstd: %6.4f' % ((sum_flags*100.0/product(tf_plane.shape)),'%', ma.std(new_data))
        print     sum_flags
        print     previous_sums
        print     '------------------------------------------------------------'
    if sum_flags == reduce(max, previous_sums, 0):
        return single_correlation_flags(new_data, threshold=threshold, max_iter=0, previous_sums=previous_sums+[sum_flags])
    else:
        return single_correlation_flags(new_data, threshold=threshold, max_iter=max_iter-1, previous_sums=previous_sums+[sum_flags])
    
    
def bad_data(data_col, threshold=5.0,max_iter=5, fubar_fraction=0.5, verbose=False):
    xx,xy,yx,yy,num_pol = split_data_col(data_col)
    flags = reduce(logical_or,
                   map(lambda x: single_correlation_flags(x,threshold=threshold,max_iter=max_iter, verbose=verbose),
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


def regrid_time_frequency_correlation_cube(cube, time_slots):
    new_cube=ma.array(zeros((max(time_slots)-min(time_slots)+1,cube.shape[1],cube.shape[2]),dtype=complex64))
    new_cube.mask=ones(new_cube.shape,dtype=bool)
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


    def baseline_table(self, ant1, ant2, subband=0):
        print 'MeasurementSetSummary.baseline_table subband: '+str(subband)
        print '(ANTENNA1 == %d && ANTENNA2 == %d || ANTENNA1 == %d && ANTENNA2 == %d) && DATA_DESC_ID == %d' % (ant1,ant2,ant2,ant1, subband)
        return tables.table(self.msname).query('(ANTENNA1 == %d && ANTENNA2 == %d || ANTENNA1 == %d && ANTENNA2 == %d) && DATA_DESC_ID == %d' % (ant1,ant2,ant2,ant1, subband))

    
    def baseline(self, ant1, ant2, column='DATA', subband=0, taper=None, **kwargs):
        """Returns a tuple with the time,frequency,correlation masked
        array cube of the visibility on baseline ant1-ant2 for
        subband, as well as an array containing the time slot number
        of each row/plane in the cube. The mask of the cube is the
        contents of the 'FLAG' column. The second element of the tuple
        is the time slot number of each row in the cube."""
        print 'MeasurementSetSummary.baseline subband: '+str(subband)
        selection = self.baseline_table(ant1, ant2, subband=subband)
        data=selection.getcol(column, **kwargs)
        if self.endian_swap:
            data.byteswap(True)
            pass

        time_centroids = selection.getcol('TIME_CENTROID', **kwargs)
        time_slots     = array((time_centroids - min(self.times))/self.integration_times[0] +0.5, dtype=int64)

        flags=selection.getcol('FLAG', **kwargs)
        raw = regrid_time_frequency_correlation_cube(ma.array(data,mask=flags),time_slots)
        if taper is not None:
            return ma.array(apply_taper(raw.data, taper(raw.shape[1])),mask=raw.mask)
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
            print '%d -- %d / %d' % (chunk*chunksize+1, (chunk+1)*chunksize, nrows)
            results += [function(flag_data(self.baseline(ant1,ant2,startrow=chunk*chunksize, nrow=chunksize),threshold=4.0, max_iter=10))]
            pass
        print '%d -- %d / %d' % (complete_chunks*chunksize+1, nrows, nrows)
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
            print '%d -- %d / %d' % (chunk*chunksize+1, (chunk+1)*chunksize, nrows)
            results += [function(selection.getcol('DATA', startrow=chunk*chunksize, nrow=chunksize))]
            pass
        print '%d -- %d / %d' % (complete_chunks*chunksize+1, nrows, nrows)
        results += [function(selection.getcol('DATA',startrow=complete_chunks*chunksize, nrow=lastset))]
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
    if sum(flags) == product(flags.shape):
        amax=1.0
    else:
        amax=(scale-stddev)*amax_factor
    

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
    padded_plane[:,(padding/2):(padding/2+nf)] = tf_plane.data*logical_not(tf_plane.mask)
    return fftshift(ifft2(padded_plane))





def plot_baseline(ms_summary, baseline, plot_flags=True,padding=1, amax_factor=1.0, num_delay=80, num_fringe_rate=160,cmap=cm.hot, subband=0, taper=None, column='DATA', **kwargs):
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
    print 'plot_baseline subband: '+str(subband)
    data            = ms_summary.baseline(baseline[0], baseline[1], subband=subband, taper=taper, column=column, **kwargs)
    flagged_data    = flag_data(data, threshold=5.0, max_iter=20)
    xx,xy,yx,yy,num_pol = split_data_col(ma.array(flagged_data))
    antenna_names   = array(ms_summary.subtable('ANTENNA').getcol('NAME'))[list(baseline)]
    print antenna_names

    if sum(flagged_data.mask) == product(flagged_data.shape):
        scale=1.0
        stddev=0.1
        amax=1.0*amax_factor
    else:
        means = array([ma.mean(x) for x in [xx,xy,yx,yy]])
        stddevs= [max(ma.std(x.real), ma.std(x.imag)) for x in [xx,xy,yx,yy]]
        scale=max(abs(means))
        stddev = max(stddevs)
        amax=(scale+2.5*stddev)*amax_factor

    
#    print '%f%% of time slots available' % (int((max(ms_summary.times[kwargs['startrow']:kwargs['startrow']+kwargs['nrow']*kwargs['rowincr']:kwargs['rowincr']]) - min(ms_summary.times[kwargs['startrow']:kwargs['startrow']+kwargs['nrow']*kwargs['rowincr']:kwargs['rowincr']]))/ms_summary.integration_times[0]+0.5)*100.0/len(time_slots),)
    print 'scale: %f\nsigma: %f' % (scale, stddev)
    good=logical_not(xx.mask)
    if not plot_flags:
        good = None
    names = ['XX', 'XY', 'YX', 'YY']

    clf()
    t = gcf().text(0.5,
                   0.95, '%s-%s %s: %6.3f MHz' % (antenna_names[0], antenna_names[1], ', '.join(ms_summary.msname.split('/')[-2:]), ms_summary.subtable('SPECTRAL_WINDOW').getcol('REF_FREQUENCY')[subband]/1e6),
                   horizontalalignment='center',
                   fontsize=24)

    for i,name,data in zip(range(4), names, [xx, xy, yx, yy]):
        subplot(241+i)
        plot_complex_image(name, data, good, amin=0.0, amax=amax)
        pass
    
    plots = map(lambda tf: delay_fringe_rate(tf, padding=padding), [xx,xy,yx,yy])
    
    amax = array([abs(d).max() for d in plots]).max()
    width=min(num_delay, xx.shape[1])
    height=min(num_fringe_rate, xx.shape[0])

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
            print col
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
    return ma.mean(array[:,1:-1,:], axis=1)

def bl_median_no_edges(array):
    return ma.median(array.real[:,1:-1,:], axis=1)+1j*ma.median(array.imag[:,1:-1,:], axis=1)

def bl_std_no_edges(array):
    return ma.std(array.real[:,1:-1,:],axis=1) +1j*ma.std(array.imag[:,1:-1,:], axis=1)



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



def printnow (s):
    print s
    sys.stdout.flush()
    pass


def ant_ant_stat_plot(ax, title_text, single_pol_array, station_names, **kwargs):
    img = ax.imshow(single_pol_array, interpolation='nearest', **kwargs)
    ax.axis('equal')
    for y,n in enumerate(station_names):
        ax.text(y, -1,n, rotation=90, verticalalignment='bottom')
        ax.text(len(station_names), y, n, horizontalalignment='left',verticalalignment='center')
        pass
    #colorbar(img, pad=0.2, ax=ax)
    ax.set_xlabel(title_text)
    pass

def ant_ant_stat_frame(title_text, full_pol_array, station_names, output_name=None, **kwargs):
    dpi=50
    if output_name is None:
        fig = figure(figsize=(32,24), dpi=dpi)
    else:
        fig=Figure(figsize=(32,24), dpi=dpi)
    
    ax1=fig.add_subplot(2,2,1)
    ant_ant_stat_plot(ax1, title_text+' XX', full_pol_array[:,:,0], station_names, **kwargs)
    
    ax2=fig.add_subplot(2,2,2)
    ant_ant_stat_plot(ax2, title_text+' XY', full_pol_array[:,:,1], station_names, **kwargs)
    
    ax3=fig.add_subplot(2,2,3)
    ant_ant_stat_plot(ax3, title_text+' YX', full_pol_array[:,:,2], station_names, **kwargs)
    
    ax4=fig.add_subplot(2,2,4)
    ant_ant_stat_plot(ax4, title_text+' YY', full_pol_array[:,:,3], station_names, **kwargs)

    if output_name is not None:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(output_name, dpi=dpi)
        pass
    pass




def collect_stats_ms(msname, max_mem_bytes=4*(2**30)):
    ms = MeasurementSetSummary(msname)
    num_ant=len(ms.tables['antennae'])
    def timeslot(mjd):
        return int((mjd-ms.times[0])/ms.integration_times[0] +0.5)
    num_timeslots = timeslot(ms.times[-1])+1
    num_bl        = num_ant*(num_ant+1)/2
    num_chan      = ms.tables['spectral_windows']['NUM_CHAN'][0]
    num_pol       = 4
    bytes_per_vis = 8
    fudge_factor  = 8
    max_timeslots = max_mem_bytes/(bytes_per_vis*num_pol*num_chan*num_bl*fudge_factor)

    num_ts = min(max_timeslots, num_timeslots)
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


    def calc_bl_stat(fn):
        """
        result must return a complex float
        """
        result = zeros(data_shape[0:3], dtype=complex64)
        for ant1 in range(num_ant):
            for ant2 in range(num_ant):
                for pol in range(num_pol):
                    result[ant1, ant2, pol] = fn(data[ant1,ant2,pol,:,:])
                    pass
                pass
            pass
        gc.collect()
        return result
    
    printnow('computing std')
    bl_std    = calc_bl_stat(ma.std)
    printnow('computing mean')
    bl_mean   = calc_bl_stat(ma.mean)
    printnow('computing zeroes')
    bl_zeroes = calc_bl_stat(lambda x: (x==0.0+0.0j).sum()/product(x.shape))


    printnow('flagging iteration 1/2')
    deviant = abs(data - bl_mean[:,:,:,newaxis,newaxis]) > 4*bl_std[:,:,:,newaxis,newaxis]
    deviant_all_pol = reduce(logical_or, [deviant[:,:,0,:,:], deviant[:,:,1,:,:], deviant[:,:,2,:,:], deviant[:,:,3,:,:]])
    data.mask = logical_or(data.mask, deviant_all_pol[:,:,newaxis,:,:])
    deviant=None
    deviant_all_pol=None
    gc.collect()
    
    printnow('computing flagged mean')
    bl_flagged_mean   = calc_bl_stat(ma.mean)

    printnow('computing flagged std')
    bl_flagged_std    = calc_bl_stat(ma.std)

    printnow('flagging iteration 2/2')
    deviant = abs(data - bl_flagged_mean[:,:,:,newaxis,newaxis]) > 4*bl_flagged_std[:,:,:,newaxis,newaxis]
    deviant_all_pol = reduce(logical_or, [deviant[:,:,0,:,:], deviant[:,:,1,:,:], deviant[:,:,2,:,:], deviant[:,:,3,:,:]])
    data.mask = logical_or(data.mask, deviant_all_pol[:,:,newaxis,:,:])
    deviant=None
    deviant_all_pol=None
    gc.collect()


    printnow('computing flags')
    bl_flags  = calc_bl_stat(lambda x: x.mask.sum()*1.0/product(x.shape))

    printnow('counting unflagged points')
    bl_good   = calc_bl_stat(lambda x: logical_not(x.mask).sum())
    
    printnow('computing flagged mean')
    bl_flagged_mean   = calc_bl_stat(ma.mean)

    printnow('computing flagged std')
    bl_flagged_std    = calc_bl_stat(ma.std)

    bl_sn = bl_flagged_std/sqrt(bl_good-1)

    return {'Meta data'         : ms.tables,
            'Standard deviation': bl_std,
            'Mean'              : bl_mean,
            'Zeroes'            : bl_zeroes,
            'Good points'       : bl_good,
            'Fringe SNR 0'      : bl_sn,
            'Flags'             : bl_flags,
            'Flagged mean'      : bl_flagged_mean,
            'Flagged standard deviation': bl_flagged_std}


def inspect_ms(msname, ms_id, max_mem_bytes=4*(2**30), output_prefix='/globalhome/brentjens/'):
    results = collect_stats_ms(msname, max_mem_bytes)

    ant_names=results['Meta data']['antennae']['NAME']

    output_dir = os.path.join(output_prefix, str(ms_id))
    try:
        os.mkdir(output_dir)
    except Exception:
        pass
    
    def write_plot(quantity_name, scaling_function=lambda x: x, **kwargs):
        return ant_ant_stat_frame(quantity_name, scaling_function(abs(results[quantity_name])), ant_names,
                                  os.path.join(output_dir,msname.split('/')[-1][:-3]+'-'+quantity_name.lower().replace(' ','-')+'.png'),
                                  **kwargs)

    write_plot('Flagged mean', log10, vmax=0.0, vmin=-4.0)
    write_plot('Flagged standard deviation', log10, vmax=0.0, vmin=-4.0)
    write_plot('Flags', log10, vmax=0.0, vmin=-3.0)
    write_plot('Zeroes',lambda x: 100*x, vmin=0.0, vmax=100.0)
    write_plot('Fringe SNR 0', log10, vmin=-1.0, vmax=3.0)
    
    return results
