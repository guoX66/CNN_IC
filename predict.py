import argparse
from PIL import Image
from models import model_dict
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

    def predict_class(self, path, model, data_class):
        transform = transforms.Compose([
            transforms.Resize([self.size[0], self.size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(self.ms[0], self.ms[1])
        ])
        model.eval()
        model = model.to(self.device)
        image = Image.open(path).convert('RGB')
        image = transform(image)
        image = torch.reshape(image, (1, 3, self.size[0], self.size[1]))  # 修改待预测图片尺寸，需要与训练时一致
        image = image.to(self.device)
        with torch.no_grad():
            output = model(image)
        predictclass = data_class[int(output.argmax(1))]
        return predictclass

    def ensemble_predict(self):
        with open("log/class_id.json", 'r', encoding='UTF-8') as f:
            class_dict = json.load(f)
        class_dict = {int(k): class_dict[k] for k in class_dict.keys()}

        img_list = [[i, os.path.join(args.predict_path, i)] for i in list(os.walk(args.predict_path))[0][-1]
                    if i.split('.')[0] != '']
        model_list = [[i.split('.')[0], os.path.join(self.model_path, i)] for i in list(os.walk(self.model_path))[0][-1]
                      if
                      i.split('.')[0] != '']
        model_list = [self.make_model(i[0].split('-')[-1], len(class_dict.keys()), i[1]) for i in model_list]
        for img in img_list:
            result_dict = {class_dict[i]: 0 for i in class_dict}
            file, path = img
            for model in model_list:
                result = self.predict_class(path, model, class_dict)
                result_dict[result] += 1
            max_key = max(result_dict, key=lambda k: result_dict[k])  # 股价最高的股票名
            if result_dict[max_key] / len(model_list) >= self.min_pr:
                ans = max_key
                self.txt_list.append([file, ans])
                print(f'{file}的预测类别为{ans}')
            else:
                self.txt_list.append([file, '未知类别'])
                print(f'{file}的预测类别为未知类别')

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
    model = PredictModel(args, model_dict)
    model.ensemble_predict()
    model.write_csv()
