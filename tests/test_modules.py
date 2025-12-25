import doctest
import unittest

from retnext import modules
from retnext.modules import RetNeXt


class TestPretrainedRetNeXt(unittest.TestCase):
    def test_load_pretrained(self):
        model = RetNeXt(pretrained=True)
        self.assertNotEqual(model.backbone[0].running_mean.item(), 0)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(modules))
    return tests
