import shutil
import sys
import pynvml
import os
import matplotlib.pyplot as plt
import time
import yaml


def remove_file(old_path, new_path):
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        shutil.move(src, dst)


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


def get_gpu_usage(gpu_id):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_utilization = util.gpu
        memory_use = '%.2f' % (mem_info.used / (1024 ** 3))
        memory_total = '%.2f' % (mem_info.total / (1024 ** 3))
        return gpu_utilization, memory_use, memory_total
    except pynvml.NVMLError as error:
        print(f"Failed to get GPU usage: {error}")
        return None, None, None


def train_bar(args):
    pynvml.nvmlInit()
    t_epoch, t_batch, start, epoch, batch, train_loss, loss, acc = args
    # 获取GPU句柄，这里假设你只有一个GPU
    l = 30
    f_p = epoch / t_epoch
    f_n = int(f_p * l)
    finsh = "▓" * f_n
    need_do = "-" * (l - f_n)
    e_process = finsh + need_do + '|'

    l2 = 15
    f_p_b = batch / t_batch
    f_n = int(f_p_b * l2)
    finsh_b = "▓" * f_n
    need_b = "-" * (l2 - f_n)
    batch_process = finsh_b + need_b + '|'

    dur = time.perf_counter() - start
    finish_epoch = epoch - 1 + f_p_b
    res = (t_epoch - finish_epoch) / (finish_epoch / dur)

    a = ("%-11s" + "%-15s" + "%-11s" * 2 + "%-17s" + "%-11s" * 2 + "%-33s" + "%-20s") % (
        "Epoch",
        "GPU0_men",
        "GPU0_use",
        "batch_loss",
        "batch_process",
        "val_loss",
        "val_acc",
        "epoch_process",
        "time"
    )
    if epoch <= 1 and batch <= 1:
        print(a)
    # 获取GPU利用率和内存占用率
    gpu_util, gpu_mem_use, gpu_mem_total = get_gpu_usage(0)
    gpu_mem_util = str(gpu_mem_use) + '/' + str(gpu_mem_total) + 'GB'
    gpu_util = str(gpu_util) + '%'
    if loss is not None:
        loss = "%.3f" % loss
    if acc is not None:
        acc = "%.3f" % acc + '%'
    epochs = str(epoch) + '/' + str(t_epoch)
    proc = "\r{:10s} {:14s} {:10s} {:10s} {:16s} {:10s} {:10s} {:31s} {:.2f}s/{:.2f}s".format(
        epochs, gpu_mem_util, gpu_util, str(train_loss), batch_process, str(loss), str(acc), e_process, dur,
        res)
    sys.stdout.write(proc)
    sys.stdout.flush()
    if epoch == t_epoch and batch == t_batch:
        print()


def bar(i, total, start, des):
    l = 30
    f_p = i / total
    n_p = (total - i) / total
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    res = (total - i) / (i / dur)
    proc = "\r{}({}/{}):{:^3.2f}%[{}->{}] time:{:.2f}s/{:.2f}s".format(
        des, i, total, progress, finsh, need_do, dur, res)
    sys.stdout.write(proc)
    sys.stdout.flush()
    if i == total:
        print()


def ini_env():
    import torch
    import platform
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device_name = []
        for i in range(gpu_num):
            device_name.append(str(i) + '  ' + torch.cuda.get_device_name(i))
        device = torch.device('cuda')
        gpus = [i for i in range(gpu_num)]
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        device_name = ['cpu']
        device = torch.device('cpu')
        gpus = []
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32
    return device, gpus, num_workers, device_name


def make_dir(path):
    import os
    base_path = os.path.dirname(path)
    if base_path:
        os.makedirs(base_path, exist_ok=True)


def jy_deal(path, mode, dic=None):
    make_dir(path)
    jy = path.split('.')[-1]
    if jy == 'json':
        import json
        if mode == 'w':
            with open(path, mode, encoding="utf-8") as f:
                json.dump(dic, f)
        elif mode == 'r':
            with open(path, mode, encoding='UTF-8') as f:
                res = json.load(f)
            return res
        else:
            raise ValueError(f'mode {mode} not supported')
    elif jy == 'yaml':

        if mode == 'w':
            with open(path, mode, encoding="utf-8") as f:
                yaml.dump(dic, f)
        elif mode == 'r':
            with open(path, mode, encoding='utf-8') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
            return res
        else:
            raise ValueError(f'mode {mode} not supported')
    else:
        raise ValueError('invalid file')


def make_plot(title, data, mode, path, epoch):
    if mode == 'loss':
        title = 'LOSS-' + title
    elif mode == 'acc':
        title = 'ACC-' + title
    else:
        raise ValueError('Invalid plotting mode')
    plt.figure(figsize=(12.8, 9.6))
    x = [i + 1 for i in range(epoch)]
    if mode == 'loss':
        plt.plot(x, data[0])
        plt.plot(x, data[1])
        plt.legend(['train', 'val'])
    else:
        plt.plot(x, data)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')
