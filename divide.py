import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil
from myutils import bar
from configs import BaseModel

# 创建一个锁对象
lock = threading.Lock()
parser = argparse.ArgumentParser()
parser.add_argument('--test_present', type=float, default=0.1)
args = parser.parse_args()


def copy_files(source, destination, index, total, start_time):
    # 复制文件
    shutil.copy(source, destination)
    # 使用锁来确保线程安全的调用进度条更新
    with lock:
        bar(index, total, start=start_time, des='拆分进度')


def divide_test(path, o_path, divide_num, thread_workers):
    import shutil
    import os
    import random
    import time

    print('正在拆分数据集......')
    shutil.rmtree(o_path, ignore_errors=True)
    train_path = os.path.join(o_path, 'train')
    test_path = os.path.join(o_path, 'test')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    path_pair = []
    start_time = time.perf_counter()
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
            for count in range(len(file_list)):
                if count < train_num:
                    in_path = os.path.join(i_fold_path, file_list[count])
                    # o_file_path_train = os.path.join(o_fold_path_train, file)
                    path_pair.append([in_path, o_fold_path_train])
                else:
                    in_path = os.path.join(i_fold_path, file_list[count])
                    # o_file_path_test = os.path.join(o_fold_path_test, file)
                    path_pair.append([in_path, o_fold_path_test])
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        # 提交任务到线程池
        futures = [executor.submit(copy_files, pair[0], pair[1], i + 1, len(path_pair), start_time) for i, pair in
                   enumerate(path_pair)]
        # 等待所有线程任务完成
        executor.shutdown(wait=True)

    print(f'拆分完成！拆分比为{int(10 - divide_num * 10)}:{int(divide_num * 10)}')


if __name__ == '__main__':
    model = BaseModel(args, None)
    divide_test(model.foldname, model.imgpath, args.test_present, 4)
