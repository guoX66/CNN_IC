import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, base_dir)
from configs import TrainImg, ModelInfo
from myutils import *
from torchvision import transforms

def t_img(txt_list, model_name):
    test_imgs = os.path.join(TrainImg().imgpath, 'test')

    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}
    real_classlist, _, _ = get_label_list(test_imgs)
    t = time.strftime('%Y年%m月%d日 %Hh:%Mm:%Ss', time.localtime())
    add_log(f'测试时间：{t}', txt_list)

    add_log('-' * 43 + '测试错误信息如下' + '-' * 43, txt_list)
    print()
    modelinfo = ModelInfo()
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    size = modelinfo.size  # 获取原模型对图像的转换
    right_num = 0
    test_num = 0
    model = make_model(modelinfo.model, len(class_dict.keys()), f'{model_name}.pth', device)
    model.eval()
    for real_classes in real_classlist:
        real_class = real_classes[0]
        test_path = f'{test_imgs}/{real_class}/{real_classes[1]}'
        image = Image.open(test_path).convert('RGB')
        image = transform(image)
        image = torch.reshape(image, (1, 3, size[0], size[1]))  # 修改待预测图片尺寸，需要与训练时一致
        with torch.no_grad():
            output = model(image)
        predict_class = class_dict[int(output.argmax(1))]

        if predict_class == real_class:
            right_num = right_num + 1
        else:
            is_right = '错误'
            add_log(f'图片{real_classes[1]}预测类别为{predict_class}，真实类别为{real_class},预测{is_right}',
                    txt_list)

        test_num = test_num + 1
    add_log(f'模型：{modelinfo.model}', txt_list)
    acc = right_num / test_num * 100
    add_log(f'测试总数量为{test_num}，错误数量为{test_num - right_num}', txt_list)
    add_log(f'总预测正确率为{acc}%', txt_list)
    write_log('log', 'test_log.txt', txt_list)
    print(f'测试完成,记录保存在log/test_log.txt中')

    return acc


if __name__ == '__main__':
    txt_list = []
    model = ModelInfo().model
    model_path = f't_model/model-{model}'
    # model_path = 'train_process\\model-googlenet-2023-06-11-13h 21m\\model-googlenet-2023-06-11-13h 21m'
    t_img(txt_list, model_path)
