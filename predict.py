import argparse
import json
import torch
from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from myutils import *
from configs import BaseModel
import openpyxl as op
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='t_model')
parser.add_argument('--predict_path', type=str, default='predict_img')
parser.add_argument('--min_pr', type=float, default=4 / 5)
args = parser.parse_args()


class PredictModel(BaseModel):
    def __init__(self, args, model_dict):
        super().__init__(args, model_dict)
        self.min_pr = args.min_pr
        self.predict_path = args.predict_path
        self.model_path = args.model_path
        self.txt_list = []

    def predict_class(self, image, model, data_class):
        model.eval()
        model = model.to(self.device)
        image = torch.reshape(image, (1, 3, self.size[0], self.size[1]))  # 修改待预测图片尺寸，需要与训练时一致
        image = image.to(self.device)
        with torch.no_grad():
            output = model(image)
        predictclass = data_class[int(output.argmax(1))]
        return predictclass

    def ensemble_predict(self, model_list, class_dict, img):
        transform = transforms.Compose([
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])
        ])
        img = transform(img)
        result_dict = {class_dict[i]: 0 for i in class_dict}
        for model in model_list:
            result = self.predict_class(img, model, class_dict)
            result_dict[result] += 1
        max_key = max(result_dict, key=lambda k: result_dict[k])  # 股价最高的股票名
        if result_dict[max_key] / len(model_list) >= self.min_pr:
            ans = max_key
        else:
            ans = 'unknown'
        return ans

    def imgs_predict(self):
        with open("log/class_id.json", 'r', encoding='UTF-8') as f:
            class_dict = json.load(f)
        class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
        img_list = [[i, os.path.join(self.predict_path, i)] for i in list(os.walk(self.predict_path))[0][-1]
                    if i.split('.')[0] != '']  # 获取[图片名,图片路径] 列表
        model_list = [[i.split('.')[0], os.path.join(self.model_path, i)] for i in list(os.walk(self.model_path))[0][-1]
                      if
                      i.split('.')[0] != '']  # 获取[模型名,模型路径] 列表
        model_list = [self.make_model(i[0].split('-')[-1], len(class_dict.keys()), i[1]) for i in model_list]
        # 获取[模型,模型路径]列表
        for img in img_list:
            file, source = img
            image = Image.open(source).convert('RGB')
            ans = self.ensemble_predict(model_list, class_dict, image)
            self.txt_list.append([img[0], ans])
            print(f'{img[0]}的预测类别为{ans}')

    def write_csv(self):
        date_time = time.strftime('%Y-%m-%d %Hh %Mm %Ss', time.localtime())
        wb = op.Workbook()
        ws = wb.create_sheet('预测结果', 0)
        ws.append(['图片名称', '预测类别'])
        for i in self.txt_list:
            ws.append(i)
        ws_ = wb['Sheet']
        wb.remove(ws_)
        wb.save(f'log/Result-{date_time}.xlsx')
        print(f'预测结果保存在log/Result-{date_time}.xlsx中')


if __name__ == '__main__':
    from models import model_dict

    model = PredictModel(args, model_dict)
    model.imgs_predict()
    model.write_csv()
