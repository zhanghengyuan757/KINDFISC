import os


def get_title(*args):
    path = ''
    args = [str(a) for a in args]
    end = args[-1]
    if end.startswith('.'):
        args.pop()
        path += '_'.join(args) + end
    else:
        path += '_'.join(args)
    return path


class PathGetter:
    def __init__(self, base_path):
        self.file_path = os.path.dirname(os.path.dirname(__file__)) + '/files/'
        self.base_path = self.file_path + base_path
        if not base_path.endswith('/'):
            self.base_path += '/'
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)

    def get_path(self, *args):
        path = self.base_path
        args = [str(a) for a in args]
        end = args[-1]
        if end.startswith('.'):
            args.pop()
            path += '_'.join(args) + end
        else:
            path += '_'.join(args)
        return path


class ChartPathGetter(PathGetter):
    def __init__(self):
        PathGetter.__init__(self, 'charts')


class LogPathGetter(PathGetter):
    def __init__(self):
        PathGetter.__init__(self, 'log')


class TempPathGetter(PathGetter):
    def __init__(self):
        PathGetter.__init__(self, 'temp')


class DataPathGetter(PathGetter):
    def __init__(self, cancer):
        PathGetter.__init__(self, 'dataset/' + cancer + '/')
