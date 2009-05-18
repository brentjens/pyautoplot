from pyrap import tables as tables
import pylab as pl
from pylab import pi,floor,sign,unique,is_numlike
from exceptions import *


def is_list(obj):
    return type(obj) == type([])

class NotImplementedError(Exception):
    pass


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


class RightAscension(Angle):
    def __init__(self, value, type='rad'):
        Angle.__init__(self, value, 0.0, 2*pi, type=type)
        pass
    pass

class Declination(Angle):
    def __init__(self, value, type='rad'):
        Angle.__init__(self, value, -pi/2, pi/2, True, type=type)
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



class Target:
    name=''
    direction=None
    def __init__(self, name, direction):
        self.name = name
        self.direction = direction
        pass


class TableFormatter:
    """Derive from this class in order to implent formatters for HTML,
    TXT, LaTeX"""
    header = []
    data   = []
    
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
    
    def __init__(self, table_as_list):
        self.header = table_as_list[0]
        self.data   = table_as_list[1:]
        pass
    
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





class MeasurementSetSummary:
    ms = None
    msname = ''
    times  = []
    mjd_start = 0.0
    mjd_end   = 0.0
    duration_seconds  = 0.0
    integration_times = []

    targets          = None
    antennae         = None
    spectral_windows = None
    
    def __init__(self, msname):
        self.msname = msname
        self.ms = tables.table(msname)
        pass

    def subtable(self, subtable_name):
        return tables.table(self.ms.getkeyword(subtable_name))

    def read_subtable(self, subtable_name, columns=None):
        subtab = self.subtable(subtable_name)
        colnames = subtab.colnames()
        if columns is not None:
            colnames = columns
        cols = [subtab.getcol(col) for col in colnames]
        return [colnames]+[[col[i] for col in cols]  for i in range(subtab.nrows())]

    def read_metadata(self):
        self.times = unique(self.ms.getcol('TIME'))
        self.mjd_start = self.times.min()
        self.mjd_end   = self.times.max()
        self.duration_seconds  = self.mjd_end - self.mjd_start
        self.integration_times = unique(self.ms.getcol('EXPOSURE'))
        
        self.targets  = TableFormatter(self.read_subtable('FIELD', ['NAME','REFERENCE_DIR']))
        self.antennae = TableFormatter(self.read_subtable('ANTENNA',['NAME', 'POSITION']))
        self.spectral_windows = TableFormatter(self.read_subtable('SPECTRAL_WINDOW',
                                                                  ['REF_FREQUENCY',
                                                                   'TOTAL_BANDWIDTH',
                                                                   'NUM_CHAN']))
        pass

    

# def summary(msname):
#     return """
# %(msname)s


# """ % { 'msname'            : msname
#         'tab'               : tb.table(msname)
#         'times'             : pl.unique(tab.getcol('TIME'))
#         'integration_times' : pl.unique(tab.getcol('EXPOSURE'))
#         'target_dirs'       : subtable(tab, 'FIELD').getcol('REFERENCE_DIR')}

