from utilities import *



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
        if len(table_as_list) == 0:
            raise ValueError('*table_as_list* is empty. Provide at least a 2D list with one row containing the header as a list of strings.')
        self.header = table_as_list[0]
        if len(table_as_list) == 1:
            self.data=[]
        else:
            self.data   = table_as_list[1:]
        
        self.default_format          = format
        self.default_col_widths      = col_widths
        self.default_cell_formatters = cell_formatters
        pass
    

    def __getitem__(self, id):
        if type(id) == type(''):
            col = self.header.index(id)
            return [row[col] for row in self.data]
        elif type(id) == type(1):
            return dict(zip(self.header, self.data[id]))

    def __repr__(self):
        return self.__str__()


    def __str__(self):
        return self.format(self.default_format, self.default_col_widths, self.default_cell_formatters)

    def __len__(self):
        return len(self.data)


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
