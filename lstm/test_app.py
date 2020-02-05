import unittest
import requests


class TestRequest(unittest.TestCase):
    """Base class for testing links"""

    def setUp(self):
        self.domain = 'http://127.0.0.1:5000'
        self.acceptable_codes = [200]

    def tearDown(self):
        self.domain = None
        self.acceptable_codes = None

    def request(self, url=None, arg_dict=None):
        url = self.domain + url
        if arg_dict:
            for arg, val in arg_dict.items():
                url = '{}?{}={}'.format(url, arg, val)
        r = requests.get(url, allow_redirects=False, verify=False)
        return r


class TestPredictEndpoint(TestRequest):
    def runTest(self):
        r = self.request('/predict')
        status = r[1].status_code
        self.assertTrue(
            status in self.acceptable_codes,
            'Found {0} in return codes'.format(status)
        )
