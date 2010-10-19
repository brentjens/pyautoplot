import unittest
from pyautoplot import *
from math import pi




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

    def test_is_compute_node(self):
        self.assertTrue(is_compute_node('lce024'))
        self.assertFalse(is_compute_node('lce0241'))
        self.assertFalse(is_compute_node('lse024'))
        self.assertFalse(is_compute_node('lce02'))
        self.assertFalse(is_compute_node(''))
        pass

    def test_compute_node_number(self):
        self.assertEquals(compute_node_number('lce012'), 12)
        pass

    def test_get_subcluster_number(self):
        self.assertEquals(get_subcluster_number('lce001'),1)
        self.assertEquals(get_subcluster_number('lce009'),1)
        self.assertEquals(get_subcluster_number('lce010'),2)
        self.assertEquals(get_subcluster_number('lce018'),2)
        self.assertEquals(get_subcluster_number('lce019'),3)
        self.assertEquals(get_subcluster_number('lce027'),3)
        pass

    def test_get_storage_node_names(self):
        [self.assertEquals(x,y) for x,y in zip(get_storage_node_names(1), ['lse001', 'lse002','lse003'])]
        [self.assertEquals(x,y) for x,y in zip(get_storage_node_names(2), ['lse004', 'lse005','lse006'])]
        [self.assertEquals(x,y) for x,y in zip(get_storage_node_names(3), ['lse007', 'lse008','lse009'])]
        [self.assertEquals(x,y) for x,y in zip(get_storage_node_names(4), ['lse010', 'lse011','lse012'])]
        pass

    def test_get_data_dirs(self):
        [self.assertEquals(found, expected) for found,expected in zip(get_data_dirs(1),
                                                                      ['/net/sub1/lse001/data1/',
                                                                       '/net/sub1/lse001/data2/',
                                                                       '/net/sub1/lse001/data3/',
                                                                       '/net/sub1/lse001/data4/',
                                                                       '/net/sub1/lse002/data1/',
                                                                       '/net/sub1/lse002/data2/',
                                                                       '/net/sub1/lse002/data3/',
                                                                       '/net/sub1/lse002/data4/',
                                                                       '/net/sub1/lse003/data1/',
                                                                       '/net/sub1/lse003/data2/',
                                                                       '/net/sub1/lse003/data3/',
                                                                       '/net/sub1/lse003/data4/'])]
        [self.assertEquals(found, expected) for found,expected in zip(get_data_dirs(3),
                                                                      ['/net/sub3/lse007/data1/',
                                                                       '/net/sub3/lse007/data2/',
                                                                       '/net/sub3/lse007/data3/',
                                                                       '/net/sub3/lse007/data4/',
                                                                       '/net/sub3/lse008/data1/',
                                                                       '/net/sub3/lse008/data2/',
                                                                       '/net/sub3/lse008/data3/',
                                                                       '/net/sub3/lse008/data4/',
                                                                       '/net/sub3/lse009/data1/',
                                                                       '/net/sub3/lse009/data2/',
                                                                       '/net/sub3/lse009/data3/',
                                                                       '/net/sub3/lse009/data4/'])]
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
