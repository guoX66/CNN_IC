import os
import random
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
sys.path.insert(0, base_dir)
from myutils import *
from configs import TrainImg


def divide_test(path, o_path, divide_num):
    print('正在拆分数据集......')
    shutil.rmtree(o_path, ignore_errors=True)
    train_path = os.path.join(o_path, 'train')
    test_path = os.path.join(o_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for i in os.walk(path):
        for c_fold in i[1]:
            i_fold_path = os.path.join(path, c_fold)
            o_fold_path_train = os.path.join(train_path, c_fold)
            o_fold_path_test = os.path.join(test_path, c_fold)
            os.makedirs(o_fold_path_train, exist_ok=True)
            os.makedirs(o_fold_path_test, exist_ok=True)

            file_list = os.listdir(i_fold_path)
            random.shuffle(file_list)
            train_num = len(file_list) - int(len(file_list) * divide_num)
            train_list = file_list[:train_num]
            test_list = file_list[train_num:]
            for file in train_list:
                in_path = os.path.join(i_fold_path, file)
                # o_file_path_train = os.path.join(o_fold_path_train, file)
                shutil.copy(in_path, o_fold_path_train)
            for file in test_list:
                in_path = os.path.join(i_fold_path, file)
                # o_file_path_test = os.path.join(o_fold_path_test, file)
                shutil.copy(in_path, o_fold_path_test)
    print(f'拆分完成！拆分比为{int(10 - divide_num * 10)}:{int(divide_num * 10)}')


if __name__ == '__main__':
    Train = TrainImg()
    divide_test(Train.foldname, Train.imgpath, Train.test_present)
