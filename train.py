# coding:utf-8
import traceback
from models import model_dict
from configs import *
from myutils import jy_deal, add_log, train_bar, make_plot, write_log
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np
import pynvml

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_present', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.95)
args = parser.parse_args()


class TrainModel(BaseModel):
    def __init__(self, args, model_dict):
        super().__init__(args, model_dict)
        self.model = args.model
        self.model_name = 'model-' + self.model
        self.val_present = args.val_present  # 拆分验证集比例
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoch = args.epoch  # 设置迭代次数

    def prepare_training(self, txt_list):
        transform = transforms.Compose([
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])
        ])

        dataset = ImageFolder(os.path.join(self.imgpath, 'train'), transform=transform)  # 训练数据集
        class_dict = dataset.class_to_idx
        class_to_id_dict = {class_dict[k]: k for k in class_dict.keys()}
        n_label = len(list(class_to_id_dict.keys()))
        model = self.make_model(self.model, n_label)
        jy_deal('log/class_id.json', 'w', class_to_id_dict)
        train_size = int((1 - self.val_present) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        dataloader_val = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                    drop_last=False,
                                    pin_memory=True)
        dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=False, pin_memory=True)

        add_log('{0:-^60}'.format('训练准备中'), txt_list)
        add_log('{0:-^60}'.format('模型与训练参数如下'), txt_list)
        add_log(f'预训练模型为:{self.model_name}', txt_list)
        add_log(
            f'训练迭代数为:{self.epoch},初始学习率为:{self.lr},衰减步长和系数为{self.step_size},{self.gamma}',
            txt_list)
        add_log(f'数据压缩量为:{self.batch_size}', txt_list)
        add_log(
            f'训练集数据量为:{len(train_dataset)},验证集数据量为:{len(val_dataset)},拆分比为{int(10 - self.val_present * 10)}:{int(self.val_present * 10)}',
            txt_list)
        add_log('计算平台: ', txt_list)
        for i in self.device_name:
            add_log(i, txt_list)

        loss_fn = nn.CrossEntropyLoss().to(self.device)  # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)  # 可调超参数
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        model = model.to(self.device)  # 将模型迁移到gpu
        if len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus, output_device=self.gpus[0])
        return [model, dataloader_train, dataloader_val, loss_fn, optimizer, scheduler,
                txt_list]

    def process_training(self, train_args):
        model, dataloader_train, dataloader_val, loss_fn, optimizer, scheduler, txt_list = train_args
        loss_list = [[], []]
        acc_list = []
        best_acc = 0  # 起步正确率
        ba_epoch = 1  # 标记最高正确率的迭代数
        min_lost = np.Inf  # 起步loss
        ml_epoch = 1  # 标记最低损失的迭代数
        ss_time = time.time()  # 开始时间
        pynvml.nvmlInit()
        date_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        filename = self.model_name + '-' + str(date_time)
        model_path = f"log/train_process/{filename}"
        os.makedirs(model_path, exist_ok=True)
        model_file = f"{model_path}/{self.model_name}.pth"
        add_log('{0:-^60}'.format('训练开始'), txt_list)
        acc = None
        total_test_loss = None
        t_batch = len(dataloader_train) + len(dataloader_val)
        st = time.perf_counter()
        for i in range(self.epoch):
            train_loss = 0
            model.train()
            for j, [imgs, targets] in enumerate(dataloader_train):
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                optimizer.zero_grad()  # 梯度归零
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 梯度优化
                loss_round = round(float(loss.item()), 3)
                train_loss += loss_round
                data = [self.epoch, t_batch, st, i + 1, j + 1, loss_round, total_test_loss, acc]
                train_bar(data)
            loss_list[0].append(train_loss)
            scheduler.step()
            model.eval()
            total_val_loss = 0
            total_accuracy = 0
            with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
                n_total_val = 0
                for k, data in enumerate(dataloader_val):
                    imgs, targets = data
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                    n_total_val += len(targets)
                    outputs = model(imgs)
                    loss = loss_fn(outputs, targets)
                    total_val_loss = total_val_loss + loss.item()
                    accuracy = (outputs.argmax(1) == targets).sum()
                    total_accuracy = total_accuracy + accuracy
                    acc = float(total_accuracy * 100 / n_total_val)
                    total_test_loss = float(total_val_loss)
                    data = [self.epoch, t_batch, st, i + 1, k + j + 2, loss_round, total_test_loss, acc]
                    train_bar(data)
                acc_list.append(round(acc, 3))
                loss_list[1].append(round(total_val_loss, 3))

                if acc > best_acc:  # 保存迭代次数中最好的模型
                    best_acc = acc
                    ba_epoch = i + 1
                    torch.save(model.state_dict(), f'{model_file}')
                if total_val_loss < min_lost:
                    ml_epoch = i + 1
                    min_lost = total_test_loss
                    torch.save(model.state_dict(), f'{model_file}')

        ee_time = time.time()
        edate_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        total_time = ee_time - ss_time

        add_log('{0:-^60}'.format(f'本次训练结束于{edate_time}，训练结果如下'), txt_list)
        add_log(f'本次训练开始时间：{date_time}', txt_list)
        add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                         int(total_time % 60)), txt_list)
        add_log(
            f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc), 3)}%',
            txt_list)
        add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(float(min_lost), 3)}', txt_list)

        return loss_list, acc_list, filename, txt_list

    def end_train(self, result):
        loss_list, acc_list, filename, txt_list = result
        make_plot(filename, loss_list, 'loss', f"log/train_process/{filename}/LOSS-{filename}", self.epoch)
        make_plot(filename, acc_list, 'acc', f"log/train_process/{filename}/ACC-{filename}", self.epoch)
        write_log(f"log/train_process/{filename}", filename, txt_list)


def train_process(args):
    txt_list = []
    model = TrainModel(args, model_dict)
    train_args = model.prepare_training(txt_list)
    result = model.process_training(train_args)
    model.end_train(result)


if __name__ == '__main__':

    try:
        train_process(args)
    except Exception as e:
        print(e)
        try:
            os.mkdir('log')
        except:
            pass
        traceback.print_exc(file=open('log/train_err_log.txt', 'w+'))
