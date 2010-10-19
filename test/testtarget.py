import unittest
from pyautoplot.target import *
from pyautoplot.angle import *


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

#
#  M A I N 
#

if __name__ == '__main__':
    unittest.main()

#
#  E O F
#
