import os
import sys
cur_dir = os.path.join(__file__)
project_root = os.path.join(cur_dir)
sys.path.append(project_root)

class AppConfig(object):
    model_types = ('nlp', 'vision', 'sequential', 'recursive')
    model_path_prefix = os.path.join(project_root, 'models')
    def __init__(self, kind):
        if kind.lower() not in AppConfig.model_types:
            raise Exception('{0} is not supported, supported are {1}'.format(kind, ' '.join(AppConfig.model_types)))
        else:
            self._model_path = os.path.join(AppConfig.model_path_prefix, kind)

