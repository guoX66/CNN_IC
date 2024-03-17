from PIL import Image
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, base_dir)
from myutils import *
from configs import TrainImg, ModelInfo, args
import openpyxl as op
from torchvision import transforms


def predict_class(path, model, modelinfo, data_class):
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    model.eval()
    image = Image.open(path).convert('RGB')
    image = transform(image)
    image = torch.reshape(image, (1, 3, size[0], size[1]))  # 修改待预测图片尺寸，需要与训练时一致
    with torch.no_grad():
        output = model(image)
        # output = torch.squeeze(model(image)).cpu()  # 压缩batch维度
        # predict = torch.softmax(output, dim=0)
        # predict_cla = torch.argmax(predict).numpy()
    predictclass = data_class[int(output.argmax(1))]
    return predictclass


def file_list(filename):
    list1 = []
    for i in os.walk(filename):
        for file in i[2]:
            path = os.path.join(filename, file)
            list1.append([path, file])
        return list1


if __name__ == '__main__':
    date_time = time.strftime('%Y-%m-%d-%Hh %Mm', time.localtime())
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    modelinfo = ModelInfo()
    min_pr = args.min_pr
    size = modelinfo.size  # 获取原模型对图像的转换
    model_path = 't_model'
    img_list = [[i, os.path.join(args.predict_path, i)] for i in list(os.walk(args.predict_path))[0][-1]
                if i.split('.')[0] != '']
    model_list = [[i.split('.')[0], os.path.join(model_path, i)] for i in list(os.walk(model_path))[0][-1] if
                  i.split('.')[0] != '']
    model_list = [make_model(i[0].split('-')[-1], len(class_dict.keys()), i[1], device) for i in model_list]

    txt_list = []
    for img in img_list:
        result_dict = {class_dict[i]: 0 for i in class_dict}
        file, path = img
        for model in model_list:
            result = predict_class(path, model, modelinfo, class_dict)
            result_dict[result] += 1
        max_key = max(result_dict, key=lambda k: result_dict[k])  # 股价最高的股票名
        if result_dict[max_key] / len(model_list) >= min_pr:
            ans = max_key
            txt_list.append([file, ans])
            print(f'{file}的预测类别为{ans}')
        else:
            txt_list.append([file, '未知类别'])
            print(f'{file}的预测类别为未知类别')

    wb = op.Workbook()
    ws = wb.create_sheet('预测结果', 0)
    ws.append(['图片名称', '预测类别'])
    for i in txt_list:
        ws.append(i)
    ws_ = wb['Sheet']
    wb.remove(ws_)
    wb.save(f'log/Result-{date_time}.xlsx')
    print(f'预测结果保存在log/Result-{date_time}.xlsx中')
