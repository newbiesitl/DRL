from urllib.parse import urlparse
class UrlParing(object):
    # you can pass url to constructor
    def __init__(self, url=None):
        if url is not None:
            url = self._check_url(url)
        self.cur = url
        self.obj = urlparse(url) if url is not None else None

    def _check_url(self,url):
        '''
        simple validation, if protocol is missing, add (default) protocol to the url
        :param url: url in string format
        :return:  sanitized url
        '''
        if url[:4] != 'http':
            return ('//'.join(['http:',url]))

    def parseUrl(self, url):
        '''
        API to add new url to the current object
        :param url:  url in text format
        :return: None
        '''
        if url is not None:
            url = self._check_url(url)
        self.cur = url
        self.obj = urlparse(url)

    def _check(self):
        '''
        Simple validation to make sure the url attribute is not None
        :return:
        '''
        if self.obj is None:
            raise AttributeError('url not set {0}'.format(str(self.cur)))

    def getProtocol(self):
        '''
        :return: protocol
        '''
        self._check()
        ret = self.obj.scheme
        return ret

    def getHost(self):
        '''
        Return host
        :return:  host
        '''
        self._check()
        ret = self.obj.netloc.split(':')[0]
        # print(ret)
        return ret

    def getPort(self):
        '''
        Return port, default 80
        :return: port
        '''
        self._check()
        ret = self.obj.port
        return ret if ret is not None else '80'

    def getPath(self):
        '''
        Return path default '/'
        :return: path
        '''
        self._check()
        ret = self.obj.path
        return ret if ret != '' else '/'


if __name__ == "__main__":
    url = '66.171.121.44'
    test = UrlParing(url)
    print(test.getHost())
    print(test.getPath())
    print(test.getPort())
    print(test.getProtocol())
    test2 = UrlParing()
    try:
        test2.getProtocol()
        assert False
    except AttributeError:
        pass
