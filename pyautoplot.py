from pyrap import tables as tables
from pylab import pi,floor,sign,unique,is_numlike
from exceptions import *
from angle import *


def is_list(obj):
    return type(obj) == type([])

class NotImplementedError(Exception):
    pass





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

    default_format          = 'txt'
    default_col_widths      = 15
    default_cell_formatters = str
    
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
        
        self.targets  = TableFormatter(
            self.read_subtable('FIELD',
                               ['NAME','REFERENCE_DIR']),
            cell_formatters=[str,
                             lambda x:
                                 str(EquatorialDirection(RightAscension(x[0,0]),
                                                         Declination(x[0,1])))])
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

