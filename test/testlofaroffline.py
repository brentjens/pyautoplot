import unittest
from pyautoplot.lofaroffline import *

class LofarOfflineTest(unittest.TestCase):

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






if __name__ == '__main__':
    unittest.main()
    pass
