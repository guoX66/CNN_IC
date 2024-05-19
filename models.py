from torchvision import models as tm

model_dict = {
    'resnet18': tm.resnet18(weights=tm.ResNet18_Weights.DEFAULT),  # 残差网络
    'resnet34': tm.resnet34(weights=tm.ResNet34_Weights.DEFAULT),
    'resnet50': tm.resnet50(weights=tm.ResNet50_Weights.DEFAULT),
    'resnet101': tm.resnet101(weights=tm.ResNet101_Weights.DEFAULT),
    'resnet152': tm.resnet152(weights=tm.ResNet152_Weights.DEFAULT),
    'googlenet': tm.googlenet(weights=tm.GoogLeNet_Weights.DEFAULT),
    'mobilenet_v3': tm.mobilenet_v3_small(weights=tm.MobileNet_V3_Small_Weights.DEFAULT),
    'densenet121': tm.densenet121(weights=tm.DenseNet121_Weights.DEFAULT),
    'densenet161': tm.densenet161(weights=tm.DenseNet161_Weights.DEFAULT),
    'efficientnet_v2_s': tm.efficientnet_v2_s(weights=tm.EfficientNet_V2_S_Weights.DEFAULT),
    'efficientnet_v2_m': tm.efficientnet_v2_m(weights=tm.EfficientNet_V2_M_Weights.DEFAULT),
    # 'vit_l_32': tm.vit_l_32(weights=tm.ViT_L_32_Weights.DEFAULT),
}
