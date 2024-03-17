import shutil
import torch.nn as nn
import numpy as np
import json
import torch
import os
import matplotlib.pyplot as plt
import time


if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    device = torch.device('cuda')
    gpus = [i for i in range(gpu_num)]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
else:
    device = torch.device('cpu')
    gpus = []


def remove_file(old_path, new_path):
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)


def make_model(modelinfo, n, path, device):
    model_ft = sele_model(modelinfo)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    model_ft.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(path, map_location=device).items()})
    if isinstance(model_ft, torch.nn.DataParallel):
        model_ft = model_ft.module
    return model_ft


def make_train_model(modelinfo, n):
    model_ft = sele_model(modelinfo, train=True)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    return model_ft


def sele_model(Model, train=False):
    from models import model_dict
    if train:
        return model_dict[Model.model]
    else:
        return model_dict[Model]


def show(c, model, txt_list):
    layer_list = get_layers_name(model)  # 获取模型各层信息
    if c.show_mode == 'All':
        for layers in layer_list:
            txt_list.append(str(layers) + '\r\n')

    elif c.show_mode == 'Simple':
        for layers in layer_list:
            txt_list.append(str(layers[0]) + '\r\n')


def get_label_list(imgpath):
    file_path = f'./{imgpath}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def bar(i, t, start, des, train=True, loss=0, acc=0):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    if train:
        proc = "\r{}({}/{}轮):{:^3.2f}%[{}->{}] 用时:{:.2f}s 验证集上损失:{:.3f} 正确率: {:.2f} %".format(des, i, t,
                                                                                                          progress,
                                                                                                          finsh,
                                                                                                          need_do, dur,
                                                                                                          loss, acc)
    else:
        proc = "\r{}:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(des, progress, finsh, need_do, dur)
    print(proc, end="")


def add_log(txt, txt_list, is_print=True):
    if is_print:
        print(txt)
    txt_list.append(txt + '\r\n')


def write_log(in_path, filename, txt_list):
    os.makedirs(in_path, exist_ok=True)
    path = os.path.join(in_path, filename + '.txt')
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


def train_dir(filename):
    try:
        os.mkdir('log/train_process')
    except:
        pass
    file_path = 'log/train_process/' + filename
    try:
        os.mkdir(file_path)
    except:
        pass


def process(img_paths):
    seq = []
    for i in img_paths:
        train_label = int(i[1])
        print(train_label)
        train_label = torch.IntTensor([train_label])
        print(train_label)
    return seq


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def get_layers_name(model):
    layer_list = []
    for layer in model.named_modules():
        layer_list.append(layer)
    return layer_list


def get_layers(model):
    layer_list = []
    for layer in model.modules():
        layer_list.append(layer)
    return layer_list


def write_json(test_dict, path):
    try:
        os.mkdir('log')
    except:
        pass
    path = os.path.join('log', path)
    json_str = json.dumps(test_dict)
    with open(f'{path}.json', 'w') as json_file:
        json_file.write(json_str)


def make_plot(data, mode, filename, epoch):
    file_path = 'log/train_process/' + filename
    if mode == 'loss':
        title = 'LOSS'
        path = os.path.join(file_path, 'LOSS-' + filename)
    elif mode == 'acc':
        title = 'ACC曲线'
        path = os.path.join(file_path, 'ACC-' + filename)
    plt.figure(figsize=(12.8, 9.6))
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title + '-' + filename, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')
