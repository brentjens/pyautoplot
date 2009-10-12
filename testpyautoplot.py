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


class UtilitiesTest(unittest.TestCase):
    def test_is_list(self):
        self.assertFalse(is_list(array([])))
        self.assertFalse(is_list(10))
        self.assertTrue(is_list([]))
        self.assertTrue(is_list([10, 'boe']))
        pass

    def test_is_masked_array(self):
        self.assertFalse(is_masked_array(array([])))
        self.assertFalse(is_masked_array(10))
        self.assertFalse(is_masked_array([]))
        self.assertTrue(is_masked_array(ma.array([10, 'boe'])))
        pass


class FlaggerTest(unittest.TestCase):
    def test_split_data_col(self):
        test_data=array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                         [[1,2,3,4],[5,6,7,8],[9,10,11,12]]])
        test_flags=array([[[True,False,False,False],[False,False,False,False],[False,True,False,False]],
                         [[False,False,False,False],[False,False,False,True],[True,True,True,True]]])
        result4 = split_data_col(ma.array(test_data, mask=test_flags))
        result2 = split_data_col(ma.array(test_data[:,:,::3], mask=test_flags[:,:,::3]))
        self.assertEquals(len(result4), 5)
        self.assertEquals(result4[-1],4)

        self.assertEquals(len(result2), 5)
        self.assertEquals(result2[-1],2)

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
