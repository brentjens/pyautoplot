from pylab import pi,floor,sign


class Angle:
    lower_bound = 0.0
    upper_bound = 2*pi
    include_upper_bound = False
    cyclical = True
    value = None
    
    def __init__(self, value, lower_bound=0, upper_bound=2*pi, include_upper_bound=False,type='rad', cyclical=True):
        """type may be 'rad' 'hms' or 'sdms'"""
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.include_upper_bound = include_upper_bound
        self.cyclical=cyclical
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
        if self.cyclical:
            if self.include_upper_bound and v == self.upper_bound:
                return self.value
            range = self.upper_bound - self.lower_bound
            steps = floor((v - self.lower_bound)/range)
            v -= steps*range
        else:
            v=max(self.lower_bound, min(v,self.upper_bound))
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
        max_h = int(floor(self.upper_bound*12/pi+0.5))
        min_h = int(floor(self.lower_bound*12/pi+0.5))
        if h_int >= max_h and self.cyclical:
            h_int -= (max_h-min_h)
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
        max_d = int(floor(self.upper_bound*180/pi+0.5))
        min_d = int(floor(self.lower_bound*180/pi+0.5))
        if d_int >= max_d and self.cyclical:
            d_int -= (max_d-min_d)
        
        base_str = sign_char+str(d_int).rjust(deg_size,'0')+':'+str(m_int).rjust(2,'0')+':'+str(s_int).rjust(2,'0')
        if decimals is 0:
            return base_str
        else:
            return base_str+'.'+str(frac_int).rjust(decimals,'0')
    pass


class RightAscension(Angle):
    def __init__(self, value, type='rad'):
        Angle.__init__(self,value, 0.0, 2*pi, cyclical=True, type=type)
        pass
    pass

class Declination(Angle):
    def __init__(self, value, type='rad'):
        Angle.__init__(self, value, -pi/2, pi/2, True, cyclical=False, type=type)
        pass
    pass

class HourAngle(Angle):
    def __init__(self, value, type='rad'):
        Angle.__init__(self, value, -pi, pi, True, type=type)
        pass
    pass
    



class EquatorialDirection:
    ra = RightAscension(0.0)
    dec = Declination(0.0)
    ref_frame = 'J2000'

    def __init__(self,ra,dec, ref_frame='J2000'):
        self.ra.set_rad(ra.value)
        self.dec.set_rad(dec.value)
        self.ref_frame = ref_frame
        pass
    
    def __str__(self):
        return '%(ref_frame)s RA: %(ra)s, DEC: %(dec)s' % \
            {'ra': self.ra.as_hms(),
             'dec': self.dec.as_sdms(),
             'ref_frame': self.ref_frame}
