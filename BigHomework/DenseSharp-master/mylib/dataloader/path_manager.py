import os
import json


class PathManager:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        import pandas as pd
        #str_json = pd.read_csv(os.path.join(self.base, 'info.csv'))
        #temp = str_json.replace("'", '"') # 将单引号，替换成双引号
        #df = json.loads(temp) # Done！ 完美
        df = pd.read_csv(os.path.join(self.base, 'train_val.csv'))
        return df

    @property
    def nodule_path(self):
        return os.path.join(self.base, 'train_val')
    
    @property
    def test_nodule_path(self):
        return os.path.join(self.base,'test')

    @property
    def test_info(self):
        import pandas as pd
        df2 = pd.read_csv(os.path.join(self.base, 'test_result.csv'))
        return df2
    
    @property
    def test_path(self):
        return os.path.join(self.base, 'test_result.csv')
        
def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path,encoding="utf-8") as f: ###
    #with open(cfg_path) as f: ###
        environ = json.load(f)
    return environ


PATH = PathManager()
