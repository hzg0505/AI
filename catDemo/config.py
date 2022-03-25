# --coding:utf-8--
# ----------------------- CONFIG CLASS ----------------------- #
import torch.cuda


class Config(object):
    """
    Holds the configuration for anything you want it to.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    I made this because doing cfg['x'] all the time is dumb.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))

        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


# 配置文件，基本设置
# --------------------------------------------- #
# 数据集设置
Dataset = Config({
    'ImgPath': "./data_raw",
    'TrainTxtPath': r"./data_raw/train_list.txt"
})


# --------------------------------------------- #
# 模型设置
VGG = Config({
    "ModelName": "VGG",
    "ModelWeightPath": r"./weight/vgg.pth",
    "InputShape": [224, 224, 3],
    "NumClasses": 12
})

Resnet50 = Config({
    "ModelName": "Resnet50",
    "ModelWeightPath": r"./weight/Resnet50.pth",
    "InputShape": [224, 224, 3],
    "NumClasses": 12
})

# --------------------------------------------- #
# 训练参数设置
class Train():
    def __init__(self):
        self.Dataset = Dataset
        self.Model = VGG
        self.TrainPercent = 0.8
        self.BatchSize = 8
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LearningRate = 0.000011
        self.Epoches = 50
        self.Transform = None



