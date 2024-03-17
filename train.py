# coding:utf-8

import platform
import traceback
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, base_dir)
from myutils import *


def pre_model(Train, txt_list, modelinfo):
    model_name = 'model-' + modelinfo.model
    lr, step_size, gamma, val_present = Train.lr, Train.step_size, Train.gamma, Train.val_present  # 获取模型参数
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32

    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])

    divide_path = os.path.join(Train.imgpath, 'train')
    if os.path.exists(divide_path):
        data_path = divide_path
    else:
        data_path = Train.foldname
    dataset = ImageFolder(data_path, transform=transform)  # 训练数据集
    class_dict = dataset.class_to_idx
    class_to_id_dict = {class_dict[k]: k for k in class_dict.keys()}

    # torch自带的标准数据集加载函数

    n_label = len(list(class_to_id_dict.keys()))
    write_json(class_to_id_dict, 'class_id')
    val_size = int(Train.val_present * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataloader_val = DataLoader(val_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=False,
                                pin_memory=True)
    dataloader_train = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=False, pin_memory=True)

    epoch = Train.epoch  # 迭代次数
    model = make_train_model(modelinfo, n_label)
    add_log('-' * 40 + '本次训练准备中，准备过程如下' + '-' * 40, txt_list)
    add_log("是否使用GPU训练：{}".format(torch.cuda.is_available()), txt_list)
    if torch.cuda.is_available():
        add_log("使用GPU做训练", txt_list)
        add_log("GPU名称为：{}".format(torch.cuda.get_device_name()), txt_list)

    else:
        add_log("使用CPU做训练", txt_list)
    add_log(
        f'训练集图片数为:{len(train_dataset)},验证集图片数为:{len(val_dataset)},拆分比为{int(10 - val_present * 10)}:{int(val_present * 10)}',
        txt_list)
    loss_fn = nn.CrossEntropyLoss().to(device)  # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 可调超参数
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    model = model.to(device)  # 将模型迁移到gpu
    if len(gpus) > 1:
        # model = nn.parallel.DistributedDataParallel(model, device_ids=gpus)
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # DP

    return [model_name, model, epoch, dataloader_train, dataloader_val, loss_fn, optimizer, scheduler, txt_list,
            device]


def train_model(train_args):
    (model_name, model, epoch, dataloader_train, dataloader_val, loss_fn, optimizer,
     scheduler, txt_list, device) = train_args
    os.makedirs('temp', exist_ok=True)
    loss_list = []
    acc_list = []
    best_acc = 0  # 起步正确率
    ba_epoch = 1  # 标记最高正确率的迭代数
    min_lost = np.Inf  # 起步loss
    ml_epoch = 1  # 标记最低损失的迭代数
    ss_time = time.time()  # 开始时间
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    train_txt = []
    start_time = time.perf_counter()
    for i in range(epoch):
        model.train()
        for j, [imgs, targets] in enumerate(dataloader_train):
            # if j == 1:
            #     os.system("nvidia-smi")
            imgs, targets = imgs.to(device), targets.to(device)

            # imgs.float()
            # imgs=imgs.float() #为将上述改为半精度操作，在这里不需要用
            # imgs=imgs.half()
            # print(imgs.shape)
            outputs = model(imgs)
            # print(outputs.shape)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()  # 梯度归零
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 梯度优化

        scheduler.step()
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            n_total_val = 0
            for j, data in enumerate(dataloader_val):
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                n_total_val += len(targets)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_val_loss = total_val_loss + loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy

            acc = float(total_accuracy * 100 / n_total_val)
            total_test_loss = float(total_val_loss)
            add_log("验证集上的loss：{:.3f}".format(total_val_loss), train_txt, is_print=False)
            add_log("验证集上的正确率：{:.3f}%".format(acc), train_txt, is_print=False)
            acc_list.append(round(acc, 5))
            loss_list.append(round(total_val_loss, 5))

            if acc > best_acc:  # 保存迭代次数中最好的模型
                add_log("根据正确率，已修改模型", train_txt, is_print=False)
                best_acc = acc
                ba_epoch = i + 1
                torch.save(model.state_dict(), f'./temp/{model_name}.pth')
            if total_val_loss < min_lost:
                add_log("根据损失，已修改模型", train_txt, is_print=False)
                ml_epoch = i + 1
                min_lost = total_test_loss
                torch.save(model.state_dict(), f'./temp/{model_name}.pth')

            bar(i + 1, epoch, start_time, '训练进度', train=True, loss=total_test_loss, acc=acc)

    print()
    ee_time = time.time()
    edate_time = time.strftime('%Y-%m-%d-%H时%M分', time.localtime())
    total_time = ee_time - ss_time

    add_log('-' * 35 + f'本次训练结束于{edate_time}，训练结果与报告如下' + '-' * 40, txt_list)
    add_log(f'本次训练开始时间：{date_time}', txt_list)
    add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                     int(total_time % 60)), txt_list)
    add_log(
        f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc), 5)}%',
        txt_list)
    add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(float(min_lost), 5)}', txt_list)

    return model_name, loss_list, acc_list


def train_process():
    from configs import TrainImg, ModelInfo
    Train = TrainImg()
    modelinfo = ModelInfo()
    txt_list = []
    add_log('*' * 40 + f' 训练开始 ' + '*' * 40, txt_list)
    train_args = pre_model(Train, txt_list, modelinfo)
    model_name, loss_list, acc_list = train_model(train_args)
    add_log(40 * '*' + f'模型训练参数如下' + 40 * '*', txt_list)
    add_log(f'预训练模型为:{modelinfo.model}', txt_list)
    add_log(
        f'训练迭代数为:{Train.epoch},初始学习率为:{Train.lr},衰减步长和系数为{Train.step_size},{Train.gamma}',
        txt_list)
    add_log(f'数据压缩量为:{Train.batch_size}', txt_list)
    date_time = time.strftime('%Y_%m_%d-%Hh_%Mm', time.localtime())
    filename = model_name + '-' + str(date_time)
    model_path = f"log/train_process/{filename}/{model_name}.pth"
    train_dir(filename)
    shutil.move(f'./temp/{model_name}.pth', model_path)
    shutil.rmtree('temp', ignore_errors=True)
    write_log(f"log/train_process/{filename}", filename, txt_list)
    make_plot(loss_list, 'loss', filename, Train.epoch)
    make_plot(acc_list, 'acc', filename, Train.epoch)


if __name__ == '__main__':
    try:
        train_process()
    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/train_err_log.txt', 'w+'))
