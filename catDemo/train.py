# --coding:utf-8--
import torch
from torch import nn, optim
from backbone import ResNet50, ResNet101, ResNet152, VGG
from config import Train
from utils import get_dataset
from torchvision import transforms

# -------------------------------------------------- #
# 训练参数
config = Train()
# 设备
device = config.Device
# 学习率
lr = config.LearningRate
# batch
batchsize = config.BatchSize
# 轮次
epochs = config.Epoches
# 权重存储路径
weight_path = config.Model.ModelWeightPath

# 模型加载
model_name = config.Model.ModelName
num_classes = config.Model.NumClasses
input_shape = config.Model.InputShape
print(input_shape[0], input_shape[1])
if model_name=="VGG":
    model = VGG(num_classes=num_classes, weight_path=weight_path).to(device)

# 数据集加载
dataset = config.Dataset.ImgPath  # image
root = config.Dataset.ImgPath
txt_path = config.Dataset.TrainTxtPath
train_percent = config.TrainPercent


# print(model)

# 模型训练
loss_f = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=lr)

my_transform = transforms.Compose([
    transforms.Resize(250),  # 图片短边缩放至x，长宽比保持不变
    transforms.RandomCrop((input_shape[0], input_shape[1])),
    transforms.ToTensor()
])


# 训练函数
def train(train_data):
    model.train()
    loss_sum = 0
    for i, data in enumerate(train_data):
        # 获得一个批次的数据和标签
        inputs, labels = data
        # print(inputs.dtype)
        # inputs = torch.tensor(inputs)
        # labels = torch.tensor(labels)
        inputs = inputs.type(torch.FloatTensor)
        inputs = inputs.to(device)
        labels = labels.type(torch.FloatTensor)
        labels = labels.to(device)
        optimiser.zero_grad()
        target = model(inputs)
        loss = loss_f(target, labels)
        loss_sum += loss.item()
        loss.backward()
        optimiser.step()
    i += 1
    print("loss:{}".format(loss_sum/i))

def test(test_data):
    model.eval()
    correct = 0
    test_sample_num = 0
    for i, data in enumerate(test_data):
        inputs, labels = data
        test_sample_num += inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = model(inputs)
        # predicted = torch.max(out, 1)
        # correct += (predicted == labels).sum()
        predicted = out.argmax(dim=1)
        correct += predicted.eq(labels.view_as(predicted)).sum()
    print("Test acc:{0}".format(correct.item() / test_sample_num))

# 训练
for epoch in range(epochs):
    D_TRAIN, D_TEST = get_dataset(root, txt_path, train_percent, batchsize, my_transform)
    print('epoch:', epoch)
    train(D_TRAIN)
    test(D_TEST)

# 保存模型
torch.save(model, weight_path)

# data = get_dataset(batchsize=32)
# for i, (img, label) in enumerate(data):
#     print(img.shape)
#     val = img[0]
#     print(val.shape)
#     unloader = transforms.ToPILImage()
#     val = torch.tensor([item.cpu().detach().numpy() for item in val]).cuda()
#     I = unloader(val*255)
#     I.show()
#     print(label.shape)
