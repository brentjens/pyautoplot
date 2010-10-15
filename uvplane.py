from pylab import *
import pyfits as p
import gc


def quadrant(angle):
    return int(2.0*angle/pi)

def fixup_rgb(rgb_image):
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
    return (1+amp-amp*(cos(4*angle)))**0.8


def phase_palette(phase_rad):
    basic_palette= [color_from_angle(a) for a in phase_rad]+[array([1.0,1.0,1.0])]
    palette=fixup_rgb(array(basic_palette)*array(list(rgb_scale_palette(phase_rad,0.5))+[1.0])[:,newaxis])
    palette[len(phase_rad)]=array([1.0,1.0,1.0])
    return palette


def rgb_from_complex_image(complex_image,amin=None, amax=None, angle_points=100, scaling_function=None):
    gc.collect()
    palette=phase_palette(2*pi*arange(angle_points)/angle_points)
    #normalized_phase 
    normalized_phase=array(floor(remainder(angle(complex_image),2*pi)*angle_points/(2*pi)), dtype=int)
    normalized_phase[normalized_phase >= angle_points] = 0
    normalized_phase[normalized_phase < 0] = 0
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
    


