import unittest
from pyautoplot import *
from math import pi



class AngleTest(unittest.TestCase):
    def test_init_adjust(self):
        self.assertAlmostEquals(Angle(-0.2, 1.0, 3.0).value, 1.8)
        self.assertAlmostEquals(Angle(0.3, -2.0, 5.0).value, 0.3)
        self.assertAlmostEquals(Angle(5.2, -2.0, 5.0).value, -1.8)
        self.assertAlmostEquals(Angle(3.0, 1.0, 3.0).value, 1.0)
        self.assertAlmostEquals(Angle(3.0, 1.0, 3.0,True).value, 3.0)
        self.assertAlmostEquals(Angle(1.0,1.0, 3.0).value, 1.0)
        pass
    
    def test_set_rad(self):
        alpha = Angle(0.0,1.0,3.0)
        self.assertAlmostEquals(alpha.value, 2.0)
        self.assertAlmostEquals(alpha.set_rad(-0.2), 1.8)
        pass

    def test_set_hms(self):
        self.assertEquals(Angle((10, 20, 30.5),type='hms').as_hms(2), '10:20:30.50')
        pass

    def test_set_sdms(self):
        self.assertEquals(Angle(('+', 10, 20, 30.5),type='sdms').as_sdms(2), '+010:20:30.50')
        self.assertEquals(Angle(('-', 10, 20, 30.5),-pi/2,pi/2,type='sdms').as_sdms(2), '-10:20:30.50')
        pass

    def test_as_hms(self):
        self.assertEquals(Angle(-4*pi/36.0).as_hms(), '22:40:00')
        self.assertEquals(Angle(-4*pi/36.0,-pi,pi).as_hms(), '-01:20:00')
        self.assertEquals(Angle(-pi/12.0 -pi/12.0/3600.0 +0.9999996/3600.0/12.0*pi).as_hms(3), '23:00:00.000') 
        self.assertEquals(Angle(-pi/12.0 -pi/12.0/3600.0 +0.9999996/3600.0/12.0*pi).as_hms(7), '22:59:59.9999996') 
        pass

    def test_as_sdms(self):
        self.assertEquals(Angle(pi/40.0,-pi/2,pi/2).as_sdms(),'+04:30:00')
        self.assertEquals(Angle(pi/40.0).as_sdms(), '+004:30:00')
        self.assertEquals(Angle(-pi/4.0).as_sdms(2), '+315:00:00.00')
        self.assertEquals(Angle(-pi/4/3600000,-pi/2,pi/2).as_sdms(2), '-00:00:00.05')
        pass



class EquatorialDirectionTest(unittest.TestCase):
    
    def test__str__(self):
        self.assertEquals(str(EquatorialDirection(6.12348768, 1.024)), 'RA: 23:23:24, DEC: +58:40:15')
        self.assertEquals(str(EquatorialDirection(6.12348768+pi, 1.024)), 'RA: 11:23:24, DEC: +58:40:15')
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
