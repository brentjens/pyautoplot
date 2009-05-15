from pyrap import tables as tb
import pylab as pl
from pylab import pi,floor,sign
import datetime


class Angle:
    lower_bound = 0.0
    upper_bound = 2*pi
    include_upper_bound = False
    value = None
    
    def __init__(self, value, lower_bound=0, upper_bound=2*pi, include_upper_bound=False,type='rad'):
        """type may be 'rad' 'hms' or 'sdms'"""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_upper_bound = include_upper_bound
        if type == 'rad':
            self.set_rad(value)
        elif type == 'hms':
            self.set_hms(*value)
        elif type == 'sdms':
            self.set_sdms(*value)
        pass

    def adjust(self, x=None):
        v = self.value
        if x is not None:
            v = x
        if self.include_upper_bound and v is self.upper_bound:
            return self.value
        range = self.upper_bound - self.lower_bound
        steps = floor((v - self.lower_bound)/range)
        v -= steps*range
        if x is None:
            self.value = v
        return v
        

    def set_rad(self, new_value):
        self.value = new_value
        return self.adjust()

    def set_hms(self, h,m,s):
        self.value = (h+m/60.0+s/3600.0)*pi/12.0
        return self.adjust()

    def set_sdms(self, sign_char, d, m, s):
        self.value = (d+m/60.0+s/3600.0)*pi/180.0
        if sign_char == '-':
            self.value = -self.value
        return self.adjust()
        

    def as_hms(self, decimals=0):
        h_float   = abs(self.value)*12.0/pi
        h_int     = int(floor(h_float))
        m_float   = 60*(h_float - h_int)
        m_int     = int(floor(m_float))
        s_float   = 60*(m_float - m_int)
        s_int     = int(floor(s_float))
        frac_int    = int(floor(10**decimals*(s_float - s_int)+0.5))
        
        if frac_int >= 10**decimals:
            frac_int -= 10**decimals
            s_int +=1
        if s_int >= 60:
            s_int -= 60
            m_int += 1
        if m_int >= 60:
            m_int -= 60
            h_int += 1
        sign_char=''
        if self.value < 0:
            sign_char = '-'
        base_str = sign_char+str(h_int).rjust(2,'0')+':'+str(m_int).rjust(2,'0')+':'+str(s_int).rjust(2,'0')
        if decimals is 0:
            return base_str
        else:
            return base_str+'.'+str(frac_int).rjust(decimals,'0')

    def as_sdms(self,decimals=0):
        min_val_size = len(str(int(floor(abs(self.lower_bound)*180/pi))))
        max_val_size = len(str(int(floor(abs(self.upper_bound)*180/pi))))
        deg_size=max(min_val_size, max_val_size)
        sign_char = '- +'[int(sign(self.value))+1]
        d_float   = abs(self.value)*180/pi
        d_int     = int(floor(d_float))
        m_float   = 60*(d_float - d_int)
        m_int     = int(floor(m_float))
        s_float   = 60*(m_float - m_int)
        s_int     = int(floor(s_float))
        
        frac_int    = int(floor(10**decimals*(s_float - s_int)+0.5))

        if frac_int >= 10**decimals:
            frac_int -= 10**decimals
            s_int +=1
        if s_int >= 60:
            s_int -= 60
            m_int += 1
        if m_int >= 60:
            m_int -= 60
            d_int += 1
        
        base_str = sign_char+str(d_int).rjust(deg_size,'0')+':'+str(m_int).rjust(2,'0')+':'+str(s_int).rjust(2,'0')
        if decimals is 0:
            return base_str
        else:
            return base_str+'.'+str(frac_int).rjust(decimals,'0')
    pass




class EquatorialDirection:
    ra = Angle(0.0,0.0,2*pi)
    dec = Angle(0.0, -pi/2.0, pi/2.0, True)

    def __init__(self,ra_rad,dec_rad):
        self.ra_rad.set(ra_rad)
        self.dec_rad.set(dec_rad)
        pass
    
    def __str__(self):
        return 'RA: %(ra)s, DEC: %(dec)s' % \
            {'ra': self.ra_rad.as_hms(),
             'dec': self.dec_rad.as_sdms()}


        


def subtable(table, subtable_name):
    return tb.table(table.getkeyword(subtable_name))
    


def sdms_from_rad(rad,lower_bound=-pi/2, upper_bound=pi/2, include_upper_bound=False):
    angle=adjust_angle(rad, lower_bound, upper_bound, include_upper_bound)
    
    return ''




# class Target:
#     name=''
#     direction=

class MeasurementSetSummary:
    msname = ''
    times  = []
    target_directions = []
    target_names      = []

# def summary(msname):
#     return """
# %(msname)s


# """ % { 'msname'            : msname
#         'tab'               : tb.table(msname)
#         'times'             : pl.unique(tab.getcol('TIME'))
#         'integration_times' : pl.unique(tab.getcol('EXPOSURE'))
#         'target_dirs'       : subtable(tab, 'FIELD').getcol('REFERENCE_DIR')}

