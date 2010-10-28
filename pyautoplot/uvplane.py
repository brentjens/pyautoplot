from pylab import *
import pyfits as p
import gc


def quadrant(angle_rad):
    """
    Returns the quadrant in which *angle_rad*, an angle in radians,
    resides. The quadrants are numbered 0 [0, pi/2>, 1 [pi/2, pi>, 2
    [pi, 3pi/2>, and 3 [3pi/2, 2pi>.
    """
    a = remainder(angle_rad, 2*pi)
    return int(2.0*a/pi)

def fixup_rgb(rgb_image):
    """
    Ensure that the values in the array *rgb_image* are all in the
    range 0.0 to 1.0 inclusive.
    """
    rgb = copy(rgb_image)
    rgb[rgb > 1.0] = 1.0
    rgb[rgb < 0.0] = 0.0
    return rgb

def color_from_angle(angle_rad):
    a = remainder(angle_rad,2*pi)
    q = quadrant(a)
    red   = [1-2*a/pi,
             2*a/pi - 1, 
             3-2*a/pi,
             2*a/pi-3][q]
    green = [2*a/pi,
             2-2*a/pi,
             3*a/(2*pi) - 1.5,
             3-3*a/(2*pi)][q]
    blue  = [0.0,
             3*a/(2*pi) -0.75,
             0.75,
             3-3*a/(2*pi)][q]
    return array([red,green,blue])
    

def rgb_scale_palette(angle,amp):
    """
    Compensate for a clover-leaf like distortion in brightness
    perception in the color scheme. It returns
    (1+amp-amp*(cos(4*angle)))**0.8, where *amp* is the amplitude of
    the correction, and *angle* the phase in radians.
    """
    return (1+amp-amp*(cos(4*angle)))**0.8


def phase_palette(phase_rad):
    """
    Returns a 2D array of shape (len(*phase_rad*)+1, 3), containing
    the RGB values associated with each angle in *phase_rad*. The last
    element is the colour used in case of phase overflow (which would
    be an error; it should never happen). In this case pure white.
    """
    basic_palette= [color_from_angle(a) for a in phase_rad]+[array([1.0,1.0,1.0])]
    palette=fixup_rgb(array(basic_palette)*array(list(rgb_scale_palette(phase_rad,0.5))+[1.0])[:,newaxis])
    palette[len(phase_rad)]=array([1.0,1.0,1.0])
    return palette


def phase_index(complex_image, points):
    """
    Returns array of the same shape as *complex_image* containing the
    phase of the complex image encoded as an integer between 0 and
    points (exclusive).
    """
    if isnan(complex_image).sum() > 0:
        raise ValueError('*complex_image* contains NaN values. Please sanitize it before plotting using, e.g. complex_image[isnan(complex_image)] == 0.0, or pyautoplot.utilities.set_nan_zero(complex_image)')
    
    normalized_phase = array(floor(remainder(angle(complex_image),2*pi)*points/(2*pi) + 0.5), dtype=int)
    normalized_phase[normalized_phase == points] = 0
    return normalized_phase


def rgb_from_complex_image(complex_image,amin=None, amax=None, angle_points=100, scaling_function=None):
    if isnan(complex_image).sum() > 0:
        raise ValueError('*complex_image* contains NaN values. Please sanitize it before plotting using, e.g. complex_image[isnan(complex_image)] == 0.0, or pyautoplot.utilities.set_nan_zero(complex_image)')
    gc.collect()
    palette=phase_palette(2*pi*arange(angle_points)/angle_points)
    normalized_phase = phase_index(complex_image, angle_points)
    #normalized_phase[normalized_phase >= angle_points] = angle_points
    #normalized_phase[normalized_phase < 0] = angle_points
    if scaling_function:
        amp=scaling_function(abs(complex_image))
    else:
        amp=abs(complex_image)
        pass

    if amax == None:
        max_amp = max(amp)
    else:
        if scaling_function:
            max_amp = scaling_function(amax)
        else:
            max_amp = amax
            pass
        pass

    if amin == None:
        min_amp = amp.min()
    else:
        if scaling_function:
            min_amp=scaling_function(amin)
        else:
            min_amp = amin
            pass
        pass
    normalized_amp = (amp - min_amp)/(max_amp-min_amp)
    amp=None
    gc.collect()
    normalized_amp[normalized_amp <= 0.0] = 0.0
    normalized_phase[normalized_amp > 1.0] = angle_points
    normalized_amp[normalized_amp > 1.0] = 1.0
    #print normalized_phase
    return palette[normalized_phase]*normalized_amp[:,:,newaxis]
    




def plot_complex_image(image, plot_title='',textsize=18, scale=False):

    limit = 6*median(abs(image))

    clf()
    if is_string(plot_title):
        figtext(0.5,0.95,plot_title,size=1.5*textsize,horizontalalignment='center')
        pass
    subplot(221)
    title("Real")
    if scale:
        imshow(image.real,interpolation='nearest',vmin=image.imag.min(),vmax=image.imag.max())
    else:
        imshow(image.real,interpolation='nearest')
        pass
    colorbar()
    
    subplot(222)
    title("Imag")
    imshow(image.imag,interpolation='nearest')
    colorbar()

    subplot(223)
    title("Abs")
    if scale:
        imshow(abs(image),interpolation='nearest',vmax=image.imag.max())
    else:
        imshow(abs(image),interpolation='nearest')
        pass
    colorbar()

    subplot(224)
    title("Phase")
    imshow(angle(image),interpolation='nearest')
    colorbar()
    pass



def read_fits_image(filename):
    hdulist=p.open(fits_filename)
    data= hdulist[0].data.squeeze()
    p.close()
    return data

def plot_uvplane(image, width_pixels=None, **kwargs):
    uvplane= fftshift(fftn(fftshift(image,[0,1])),[0,1])
    gc.collect()
    uvplane_shape = uvplane.shape
    print 'image shape    : '+str(uvplane_shape)
    print 'amplitude_range: '+str(abs(uvplane).min())+'--'+str(abs(uvplane).max())
    if width_pixels:
        uvplane_cropped = uvplane[(uvplane_shape[0]/2-width_pixels/2):(uvplane_shape[0]/2+width_pixels/2),
                                                 (uvplane_shape[1]/2-width_pixels/2):(uvplane_shape[1]/2+width_pixels/2)]
    else:
        uvplane_cropped = uvplane
        pass
    uvplane=None
    gc.collect()
    return imshow(rgb_from_complex_image(uvplane_cropped,
                                         **kwargs),
                  interpolation='nearest')
    


