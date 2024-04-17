import torch
import torch.nn as nn
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from myutils import ini_env


class BaseModel:
    def __init__(self, args, model_dict, size=[256, 256], ms=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]):
        self.imgpath = 'data/images'  # 保存图片的文件名
        self.foldname = 'data/static'
        self.size = size
        self.ms = ms
        self.model_dict = model_dict
        self.device, self.gpus, self.num_workers, self.device_name = ini_env()

    def sele_model(self, model):
        return self.model_dict[model]

    def _set_module(self, model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)

    def get_layers_name(self, model):
        layer_list = []
        for layer in model.named_modules():
            layer_list.append(layer)
        return layer_list

    def make_model(self, model, n, path=None):
        model_ft = self.sele_model(model)
        layer_list = self.get_layers_name(model_ft)
        last_layers_name = layer_list[-1][0]
        in_features = layer_list[-1][1].in_features
        layer1 = nn.Linear(in_features, n)
        self._set_module(model_ft, last_layers_name, layer1)
        if path is not None:
            model_ft.load_state_dict(
                {k.replace('module.', ''): v for k, v in torch.load(path).items()})
        if isinstance(model_ft, torch.nn.DataParallel):
            model_ft = model_ft.module
        return model_ft
