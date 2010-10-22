import unittest
from pyautoplot.main import *
from math import pi




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
