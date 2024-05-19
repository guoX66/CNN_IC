import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil
from myutils import bar

# 创建一个锁对象
lock = threading.Lock()
parser = argparse.ArgumentParser()
parser.add_argument('--test_present', type=int, default=0.1)
parser.add_argument('--mode', type=str, default='soft')
parser.add_argument('--input_path', type=str, default='data/static')
parser.add_argument('--output_path', type=str, default='data/images')
args = parser.parse_args()
num_workers = min(8, max(1, os.cpu_count() - 1))


def copy_files(source, destination, index, total, start_time):
    # 复制文件
    shutil.copy(source, destination)
    # 使用锁来确保线程安全的调用进度条更新
    with lock:
        bar(index, total, start=start_time, des='拆分进度')


def divide_test(path, o_path, test):
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
    divide_num = f'{int(10 - test * 10)}:{int(test * 10)}'
    for i in os.walk(path):
        for c_fold in i[1]:
            i_fold_path = os.path.join(path, c_fold)
            o_fold_path_train = os.path.join(train_path, c_fold)
            o_fold_path_test = os.path.join(test_path, c_fold)

            os.makedirs(o_fold_path_train, exist_ok=True)
            os.makedirs(o_fold_path_test, exist_ok=True)

            file_list = os.listdir(i_fold_path)
            random.shuffle(file_list)
            train_num = len(file_list) - int(len(file_list) * test)

            for count in range(len(file_list)):
                if count < train_num:
                    in_path = os.path.join(i_fold_path, file_list[count])
                    # o_file_path_train = os.path.join(o_fold_path_train, file)
                    path_pair.append([in_path, o_fold_path_train])
                else:
                    in_path = os.path.join(i_fold_path, file_list[count])
                    # o_file_path_test = os.path.join(o_fold_path_test, file)
                    path_pair.append([in_path, o_fold_path_test])
    return path_pair, start_time, divide_num


def hard_divide(path, o_path, test, thread_workers):
    path_pair, start_time, divide_num = divide_test(path, o_path, test)
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        # 提交任务到线程池
        futures = [executor.submit(copy_files, pair[0], pair[1], i + 1, len(path_pair), start_time) for i, pair in
                   enumerate(path_pair)]
        # 等待所有线程任务完成
        executor.shutdown(wait=True)
    print(f'拆分完成！拆分比为{divide_num}')


def soft_divide(path, o_path, test):
    path_pair, start_time, divide_num = divide_test(path, o_path, test)
    for i, j in path_pair:
        out_path = os.path.join(j, os.path.basename(i))
        os.symlink(os.path.abspath(i), os.path.abspath(out_path))

    print(f'拆分完成！拆分比为{divide_num}')


if __name__ == '__main__':
    test = args.test_present
    if args.mode == 'default':
        hard_divide(args.input_path, args.output_path, test, num_workers)
    else:
        soft_divide(args.input_path, args.output_path, test)
