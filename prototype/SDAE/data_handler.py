import os


class _folder_sentence_iter(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()

class _file_sentence_iter(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):
        for line in open(os.path.join(self.file_name)):
            yield line.split()


class Sentences(object):
    @staticmethod
    def from_dir(dir_path):
        return _folder_sentence_iter(dir_path)
    @staticmethod
    def from_file(file_path):
        return _file_sentence_iter(file_path)




