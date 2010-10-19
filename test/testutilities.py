import unittest
from numpy import array
from pyautoplot.utilities import *

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
    pass


#
#  M A I N 
#

if __name__ == '__main__':
    unittest.main()

#
#  E O F
#
