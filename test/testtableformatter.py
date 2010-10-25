import unittest
from pyautoplot.tableformatter import *



class TableFormatterTest(unittest.TestCase):
    def setUp(self):
        self.formatter = TableFormatter([['NAME','POSITION'],
                                         ['a', [1,2,3]],
                                         ['b', [4,5,6]]])
        self.no_data = TableFormatter([['NAME', 'POSITION']])
        pass


    def test_init(self):
        self.assertRaises(ValueError, TableFormatter, ([]))
        self.assertEquals(self.no_data.header, ['NAME', 'POSITION'])
        self.assertEquals(self.no_data.data, [])
        self.assertEquals(self.formatter.header, ['NAME','POSITION'])
        self.assertEquals(self.formatter.data,
                          [['a', [1,2,3]],
                           ['b', [4,5,6]]])
        pass


    def test_getitem(self):
        self.assertEquals(self.formatter['NAME'][0], 'a')
        self.assertEquals(self.formatter['NAME'][1], 'b')
        
        first = self.formatter[0]
        self.assertEquals(first['NAME'], 'a')
        self.assertEquals(first['POSITION'][0], 1)
        self.assertEquals(first['POSITION'][1], 2)
        self.assertEquals(first['POSITION'][2], 3)

        last = self.formatter[-1]
        self.assertEquals(last['NAME'], 'b')
        self.assertEquals(last['POSITION'][0], 4)
        self.assertEquals(last['POSITION'][1], 5)
        self.assertEquals(last['POSITION'][2], 6)
        pass


    def test_len(self):
        self.assertEquals(len(self.no_data), 0)
        self.assertEquals(len(self.formatter), 2)
        pass


    def test_html(self):
        self.assertEquals(self.formatter.format_as_string('html',col_widths=5),
                          """<table>
<th><td>
NAME </td><td>POSITION
</td></th>
<tr><td>a    </td><td>[1, 2, 3]</td></tr>
<tr><td>b    </td><td>[4, 5, 6]</td></tr>
</table>""")
        self.assertEquals(self.formatter.format_as_string('html',col_widths=5,cell_formatters=[lambda x: 'boe'+x, lambda x: 'baa'+str(x[0])]),
                          """<table>
<th><td>
NAME </td><td>POSITION
</td></th>
<tr><td>boea </td><td>baa1 </td></tr>
<tr><td>boeb </td><td>baa4 </td></tr>
</table>""")
        pass


    def test_txt(self):
        self.assertEquals(self.formatter.format_as_string('txt',col_widths=5),
                          """
================================================================================
NAME  POSITION
--------------------------------------------------------------------------------
a     [1, 2, 3]
b     [4, 5, 6]
--------------------------------------------------------------------------------""")
        pass


    def test_str(self):
        self.assertEquals(str(self.formatter),
                          self.formatter.format_as_string('txt', 15, str))
        pass


    def test_repr(self):
        self.assertEquals(repr(self.formatter), str(self.formatter))
        pass



    pass


#
#  M A I N 
#

if __name__ == '__main__':
    unittest.main()

#
#  E O F
#
