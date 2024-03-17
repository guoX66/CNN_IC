# CNN_IC（CNN Image Classification,基于CNN网络的图像分类)



# 一、环境配置

首先需安装 python>=3.10.2，然后将项目移至全英文路径下,安装以下依赖

## pytorch

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
```

## 其他依赖

```bash
pip install -r requirements.txt
```



# 二、数据集准备

将需要分类的图像存入data/static文件夹中，按类放入各个子文件夹中，并以类的名称命名子文件夹。结构示例如下：

```
--data
    --static
        --class1
            --img1.jpg
            --img2.jpg
                ...
        --class2
            --img1.jpg
            --img2.jpg
                ...
        --class3
            ...
```

若需要对模型进行测试，可以运行divide.py脚本对数据集进行拆分：

```bash
python divide.py --test_present 0.1
```

# 三、模型训练

在命令行输入以下命令开始训练

```bash
python train.py --model resnet18 --epoch 100 --batch_size 4 --val_present 0.2 --lr 0.001 --step_size 1 --gamma 0.95
```

训练好的模型和过程记录保存在log/train_process文件夹下

# **四、模型测试

按照二中步骤运行divide.py脚本对数据集进行拆分后，可以对训练好模型进行测试

将模型移到t_model文件夹下，运行test.py脚本

```bash
python test.py --model resnet18
```

测试结果保存在log文件夹下
