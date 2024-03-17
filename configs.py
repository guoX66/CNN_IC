import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--val_present', type=float, default=0.2)
parser.add_argument('--test_present', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.95)
args = parser.parse_args()


class ModelInfo:
    def __init__(self):
        self.model = args.model  # 选择模型，可选googlenet,resnet18，resnet34，resnet50，resnet101
        self.modelname = 'model-' + self.model
        self.size = [256, 256]  # 输入模型的图片大小
        # self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        self.ms = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class TrainImg:
    def __init__(self):
        self.imgpath = 'data/images'  # 保存图片的文件名
        self.foldname = 'data/static'
        self.val_present = args.val_present  # 拆分验证集比例
        self.test_present = args.test_present
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.epoch = args.epoch  # 设置迭代次数



