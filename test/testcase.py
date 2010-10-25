import unittest
    
def make_sequence_assertion(assertion_function, **kwargs):
    def sequence_assertion_function(self, result, expected, **kwargs):
        self.assertEquals(len(result), len(expected))
        for r,e in zip(result,expected):
            try:
                assertion_function(self, r,e,**kwargs)
            except AssertionError, er:
                er.args = ('%s != %s.\n  %s' % \
                               (result, expected, er.args[0]),)
                raise er
            pass
        pass
    return sequence_assertion_function



class TestCase(unittest.TestCase):
    
    def assertLessThan(self, result, limit, **kwargs):
        try:
            self.assertTrue(result < limit, **kwargs)
        except AssertionError, er:
            
            er.args = ('%s !< %s.\n' % (result, limit),)
            raise er
        pass

    pass
    

TestCase.assertSequenceEquals       = make_sequence_assertion(unittest.TestCase.assertEquals)
TestCase.assertSequenceAlmostEquals = make_sequence_assertion(unittest.TestCase.assertAlmostEquals)
TestCase.assertSequenceLessThan     = make_sequence_assertion(TestCase.assertLessThan)

