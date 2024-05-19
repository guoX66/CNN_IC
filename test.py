import argparse
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from configs import BaseModel
from models import model_dict
from myutils import *
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import PngImagePlugin

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10  # this is the current value

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='2024-05-03 12h 40m 51s')
parser.add_argument('--test', type=str, default='data/static/test')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()


class TestModel(BaseModel):
    def __init__(self, args, model_dict):
        super().__init__(args, model_dict)
        self.img_path = args.test
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.name = os.path.join('log', 'train_process', args.name)
        files = os.listdir(self.name)
        pattern = r"best_model-(.*?).pth"
        is_match = False
        for filename in files:
            match = re.match(pattern, filename)
            if match:
                self.model = match.group(1)
                self.model_name = 'model-' + self.model
                is_match = True
                break
        if not is_match:
            raise ValueError(f'{args.name} has no model!')

    def t_acc(self, txt_list):
        transform = transforms.Compose([
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])
        ])
        datasets = ImageFolder(self.img_path, transform=transform)
        total = len(datasets)
        datasets = DataLoader(datasets, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                              drop_last=False,
                              pin_memory=False)
        with open(f"{self.name}/class_id.json", 'r', encoding='UTF-8') as f:
            class_dict = json.load(f)
        class_dict = {int(k): class_dict[k] for k in class_dict.keys()}

        t = time.strftime('%Y %m %d  %Hh:%Mm:%Ss', time.localtime())
        add_log(f'Time:{t}', txt_list)
        add_log(f'model:{self.model}', txt_list)
        right_num = 0
        test_num = 0
        model = self.make_model(self.model, len(class_dict.keys()), f"{self.name}/best_{self.model_name}.pth")
        model.to(self.device)
        model.eval()
        top_1 = 0
        top_5 = 0
        top_10 = 0

        with torch.no_grad():  # 验证数据集时禁止反向传播优化权重
            for data in tqdm(datasets):
                imgs, targets = data
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(imgs)
                top_1_tensor, top_1_indices = outputs.topk(k=1, dim=1, largest=True, sorted=True)
                top_5_tensor, top_5_indices = outputs.topk(k=5, dim=1, largest=True, sorted=True)
                top_10_tensor, top_10_indices = outputs.topk(k=10, dim=1, largest=True, sorted=True)
                top_1 += (top_1_indices == targets.unsqueeze(1)).sum().item()
                top_5 += (top_5_indices == targets.unsqueeze(1).expand_as(top_5_indices)).sum().item()
                top_10 += (top_10_indices == targets.unsqueeze(1).expand_as(top_10_indices)).sum().item()
                del imgs, targets

        top_1_accuracy = top_1 / total
        top_5_accuracy = top_5 / total
        top_10_accuracy = top_10 / total

        add_log(f'Top-1 Accuracy: {top_1_accuracy * 100:.2f}%', txt_list)
        add_log(f'Top-5 Accuracy: {top_5_accuracy * 100:.2f}%', txt_list)
        add_log(f'Top-10 Accuracy: {top_10_accuracy * 100:.2f}%', txt_list)
        write_log(f'{self.name}', f'{self.model_name}_test', txt_list)
        print(f'测试完成,记录保存在{self.name}/{self.model_name}/test.txt中')


if __name__ == '__main__':
    txt_list = []
    model = TestModel(args, model_dict)
    model.t_acc(txt_list)
