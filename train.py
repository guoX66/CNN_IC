# coding:utf-8
import traceback
from models import model_dict
from configs import *
from myutils import jy_deal, add_log, train_bar, make_plot, write_log, save_csv
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import argparse
import os
import time
import numpy as np
import pynvml
from torch.cuda.amp import autocast
from torch.utils.data import Subset

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='googlenet')
parser.add_argument('--data', type=str, default='C:/Users/Mikky/Desktop/昆虫爬虫文件下载/insect_data/train')
parser.add_argument('--val_present', type=float, default=0.2)
# parser.add_argument('--name', type=str, default='2024-04-30 13h 19m 50s')
parser.add_argument('--name', type=str)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--loss_mode', type=str, default='change', choices=('change', 'hard', 'soft'))
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()

from PIL import PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10  # this is the current value


# 标签软化
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TrainModel(BaseModel):
    def __init__(self, args, model_dict):
        super().__init__(args, model_dict, size=[224, 224])
        self.data = args.data
        self.val_present = args.val_present
        self.model = args.model
        self.model_name = 'model-' + self.model
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoch = args.epoch  # 设置迭代次数
        self.loss_mode = args.loss_mode
        self.num_workers = args.num_workers
        if args.name is None:
            self.name = os.path.join('log', 'train_process', time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime()))
            self.is_continue = False
        else:
            self.name = os.path.join('log', 'train_process', args.name)
            self.is_continue = True
        if self.is_continue:
            if not os.path.exists(self.name):
                raise FileNotFoundError(f'{self.name} not found')
            else:
                train_cfg = jy_deal(os.path.join(self.name, 'train_cfg.yaml'), 'r')
                self.epoch = train_cfg['epoch']
                self.batch_size = train_cfg['batch_size']
                self.lr = train_cfg['lr']
                self.step_size = train_cfg['step_size']
                self.gamma = train_cfg['gamma']
                self.model = train_cfg['model']
                self.data = train_cfg['data']
                self.val_present = train_cfg['val_present']
                self.name = train_cfg['name']
                self.ms = train_cfg['ms']
                self.size = train_cfg['size']
                self.loss_mode = train_cfg['loss_mode']
        else:
            train_cfg = {
                'epoch': self.epoch,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'step_size': self.step_size,
                'gamma': self.gamma,
                'model': self.model,
                'data': self.data,
                'val_present': self.val_present,
                'name': self.name,
                'ms': self.ms,
                'size': self.size,
                'loss_mode': self.loss_mode
            }
            jy_deal(os.path.join(self.name, 'train_cfg.yaml'), 'w', train_cfg)
        os.makedirs(self.name, exist_ok=True)

    def prepare_training(self, txt_list):
        transform = transforms.Compose([
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])
        ])
        dataset = ImageFolder(self.data, transform=transform)  # 训练数据集
        if not self.is_continue:
            class_dict = dataset.class_to_idx
            class_to_id_dict = {class_dict[k]: k for k in class_dict.keys()}
            n_label = len(list(class_to_id_dict.keys()))

            jy_deal(f'{self.name}/class_id.json', 'w', class_to_id_dict)

            train_size = int((1 - self.val_present) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            torch.save(train_dataset.indices, f'{self.name}/train_indices.pth')
            torch.save(val_dataset.indices, f'{self.name}/val_indices.pth')
            model = self.make_model(self.model, n_label)
        else:
            train_indices = torch.load(f'{self.name}/train_indices.pth')
            val_indices = torch.load(f'{self.name}/val_indices.pth')

            # 使用加载的索引重建数据集
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)
            class_to_id_dict = jy_deal(f'{self.name}/class_id.json', 'r')
            n_label = len(list(class_to_id_dict.keys()))
            model = self.make_model(self.model, n_label, f"{self.name}/last_{self.model_name}.pth")

        t_n = len(train_dataset)
        v_n = len(val_dataset)

        rate = f"{round(t_n * 10 / (t_n + v_n))}:{round(v_n * 10 / (t_n + v_n))}"

        dataloader_val = DataLoader(val_dataset, batch_size=2 * self.batch_size, shuffle=False,
                                    num_workers=self.num_workers,
                                    drop_last=False,
                                    pin_memory=True)
        dataloader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers,
                                      drop_last=True, pin_memory=True)

        add_log('{0:-^60}'.format('训练准备中'), txt_list)
        add_log('{0:-^60}'.format('模型与训练参数如下'), txt_list)
        add_log(f'预训练模型为:{self.model_name}', txt_list)
        add_log(
            f'训练迭代数为:{self.epoch},初始学习率为:{self.lr},衰减步长和系数为{self.step_size},{self.gamma}',
            txt_list)
        add_log(f'数据压缩量为:{self.batch_size}', txt_list)
        add_log(
            f'训练集数据量为:{t_n},验证集数据量为:{v_n},拆分比为{rate}',
            txt_list)
        add_log('计算平台: ', txt_list)
        for i in self.device_name:
            add_log(i, txt_list)

        loss_fn_hard = nn.CrossEntropyLoss().to(self.device)  # 优化器
        loss_fn_soft = LabelSmoothingLoss(classes=len(class_to_id_dict), smoothing=0.1).to(self.device)

        model = model.to(self.device)  # 将模型迁移到gpu
        if len(self.gpus) > 1:
            model = nn.DataParallel(model, device_ids=self.gpus, output_device=self.gpus[0])

        del train_dataset, val_dataset

        return [model, dataloader_train, dataloader_val, loss_fn_soft, loss_fn_hard, txt_list, v_n]

    def process_training(self, train_args):
        model, dataloader_train, dataloader_val, loss_fn_soft, loss_fn_hard, txt_list, v_n = train_args
        pynvml.nvmlInit()
        last_model_file = f"{self.name}/last_{self.model_name}.pth"
        best_model_file = f"{self.name}/best_{self.model_name}.pth"
        add_log('{0:-^60}'.format('训练开始'), txt_list)
        t_batch = len(dataloader_train) + len(dataloader_val)
        if self.is_continue:
            data = np.loadtxt(f"{self.name}/{self.model_name}.csv", delimiter=',')
            t_list = data[:, 0].tolist()
            train_list = data[:, 1].tolist()
            val_list = data[:, 2].tolist()
            acc_list = data[:, 3].tolist()
            lr_list = data[:, 4].tolist()
            loss_list = [train_list, val_list]
            start_epoch = len(train_list)
            acc = acc_list[-1]
            lr = lr_list[-1]
            val_loss = loss_list[1][-1]
            ml_epoch = np.argmin(val_list) + 1
            ba_epoch = np.argmax(acc_list) + 1
            best_acc = max(acc_list)
            min_lost = min(val_list)
        else:
            loss_list = [[], []]
            acc_list = []
            t_list = []
            best_acc = 0  # 起步正确率
            ba_epoch = 1  # 标记最高正确率的迭代数
            min_lost = np.Inf  # 起步loss
            ml_epoch = 1  # 标记最低损失的迭代数
            start_epoch = 0
            acc = None
            val_loss = None
            lr = self.lr
        # print(lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 可调超参数
        scheduler = StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        date_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        st = time.perf_counter()
        ss_time = time.time()  # 开始时间
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
        print(a)
        if self.loss_mode == 'soft':
            train_loss_fn = loss_fn_soft
        else:
            train_loss_fn = loss_fn_hard
        for i in range(start_epoch, self.epoch):
            if self.loss_mode == 'change':
                if sorted(loss_list[1], reverse=True) != loss_list[1] or (i + 1) / self.epoch >= 0.5:
                    train_loss_fn = loss_fn_soft
                    # print('change epoch'+str(i+1))

            train_loss = 0
            model.train()
            for j, [imgs, targets] in enumerate(dataloader_train):
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                with autocast():
                    outputs = model(imgs)
                    loss = train_loss_fn(outputs, targets)
                optimizer.zero_grad()  # 梯度归零
                loss.backward()  # 反向传播计算梯度
                optimizer.step()  # 梯度优化
                loss_round = round(float(loss.item()), 3)
                del loss, imgs, targets
                train_loss += loss_round
                data = [self.epoch, t_batch, st, i + 1, j + 1, loss_round, val_loss, acc, start_epoch]
                train_bar(data)
            loss_list[0].append(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            model.eval()
            total_val_loss = 0
            total_accuracy = 0
            with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
                for k, data in enumerate(dataloader_val):
                    imgs, targets = data
                    imgs = imgs.to(self.device)
                    targets = targets.to(self.device)
                    with autocast():
                        outputs = model(imgs)
                        loss = loss_fn_hard(outputs, targets)
                    accuracy = (outputs.argmax(1) == targets).sum()
                    total_val_loss = total_val_loss + loss.item()
                    total_accuracy = total_accuracy + accuracy
                    del loss, imgs, targets
                    # val_loss = float(total_val_loss / n_total_val)
                    val_loss = float(total_val_loss)
                    acc = float(total_accuracy * 100 / v_n)
                    data = [self.epoch, t_batch, st, i + 1, j + k + 2, loss_round, val_loss, acc, start_epoch]
                    train_bar(data)
                acc_list.append(round(acc, 5))
                loss_list[1].append(round(val_loss, 5))
                torch.save(model.state_dict(), f'{last_model_file}')
                if acc > best_acc:  # 保存迭代次数中最好的模型
                    best_acc = acc
                    ba_epoch = i + 1
                    torch.save(model.state_dict(), f'{best_model_file}')
                if total_val_loss < min_lost:
                    ml_epoch = i + 1
                    min_lost = val_loss
                    torch.save(model.state_dict(), f'{best_model_file}')
            ee_time = time.time()
            save_csv(f"{self.name}/{self.model_name}.csv", ee_time - ss_time, train_loss, val_loss, acc, current_lr)
            t_list.append(ee_time - ss_time)
            ss_time = ee_time
        edate_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        total_time = sum(t_list)

        add_log('{0:-^60}'.format(f'本次训练结束于{edate_time}，训练结果如下'), txt_list)
        add_log(f'本次训练开始时间：{date_time}', txt_list)
        add_log("本次训练用时:{}小时:{}分钟:{}秒".format(int(total_time // 3600), int((total_time % 3600) // 60),
                                                         int(total_time % 60)), txt_list)
        add_log(
            f'验证集上在第{ba_epoch}次迭代达到最高正确率，最高的正确率为{round(float(best_acc), 3)}%',
            txt_list)
        add_log(f'验证集上在第{ml_epoch}次迭代达到最小损失，最小的损失为{round(float(min_lost), 3)}', txt_list)

        make_plot(f'LOSS-{self.model_name}', loss_list, 'loss', f"{self.name}/LOSS-{self.model_name}", self.epoch)
        make_plot(f'ACC-{self.model_name}', acc_list, 'acc', f"{self.name}/ACC-{self.model_name}", self.epoch)
        write_log(f"{self.name}", self.model_name, txt_list)


def train_process(args):
    txt_list = []
    model = TrainModel(args, model_dict)
    train_args = model.prepare_training(txt_list)
    model.process_training(train_args)


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
