import unittest
from pyautoplot import *
from math import pi



class TargetTest(unittest.TestCase):
    
    def test_init(self):
        target = Target('Cas A', 
                        EquatorialDirection(RightAscension((23, 58, 12),type='hms'),
                                            Declination(('+', 58, 12,13.500001), type='sdms')))
        self.assertEquals(target.name, 'Cas A')
        self.assertEquals(target.direction.ra.as_hms(), '23:58:12')
        self.assertEquals(target.direction.dec.as_sdms(), '+58:12:14')
        pass
    pass



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



class FlaggerTest(unittest.TestCase):
    pass

#
#  M A I N 
#

if __name__ == '__main__':
    unittest.main()

#
#  E O F
#
