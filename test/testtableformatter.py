import unittest
from pyautoplot.tableformatter import *



class TableFormatterTest(unittest.TestCase):
    formatter = TableFormatter([['NAME','POSITION'],
                                ['a', [1,2,3]],
                                ['b', [4,5,6]]])

    def test_init(self):
        self.assertEquals(self.formatter.header, ['NAME','POSITION'])
        self.assertEquals(self.formatter.data,
                          [['a', [1,2,3]],
                           ['b', [4,5,6]]])
        pass

    def test_html(self):
        self.assertEquals(self.formatter.format('html',col_widths=5),
                          """<table>
<th><td>
NAME </td><td>POSITION
</td></th>
<tr><td>a    </td><td>[1, 2, 3]</td></tr>
<tr><td>b    </td><td>[4, 5, 6]</td></tr>
</table>""")
        self.assertEquals(self.formatter.format('html',col_widths=5,cell_formatters=[lambda x: 'boe'+x, lambda x: 'baa'+str(x[0])]),
                          """<table>
<th><td>
NAME </td><td>POSITION
</td></th>
<tr><td>boea </td><td>baa1 </td></tr>
<tr><td>boeb </td><td>baa4 </td></tr>
</table>""")
        pass

    def test_txt(self):
        self.assertEquals(self.formatter.format('txt',col_widths=5),
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
                          self.formatter.format('txt', 15, str))
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
