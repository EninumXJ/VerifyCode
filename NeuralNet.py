# -*- coding: UTF-8 -*-
#@Time : 2021/7/8 9:12
#@File : NeuralNet.py
#@Software: PyCharm
import torch
from torch import nn
from torch.nn import init
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
from ReadLabel import ReadPic
from Normalization import Normalize

# normMean = [0.69690245, 0.69690245, 0.69690245]
# normStd = [0.38857305, 0.38857305, 0.38857305]

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class AlexNet(nn.Module):
    def __init__(self,num_input,num_output,BATCH_SIZE):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(1, 96, 11, 4),
            nn.Conv2d(1,16,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # nn.Conv2d(96, 256, 5, 1, 2),
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            # nn.Conv2d(256, 384, 3, 1, 1),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            # nn.Conv2d(384, 384, 3, 1, 1),
            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            # nn.Conv2d(384, 256, 3, 1, 1),
            nn.Conv2d(128,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )

        self.fc = nn.Sequential(
            # nn.Linear(64*5*5, 4096),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096,4096),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(4096,num_output)
            nn.Linear(64 * 5 * 5, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, num_output)
        )
        # 迭代循环初始化参数
        for m in self.children():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, -100)
            # 也可以判断是否为conv2d，使用相应的初始化方式
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0)

    def forward(self,img):
        x = img.view((img.shape[0],1) + img.shape[1:3])
        # print(x.shape)
        feature = self.conv(x)
        # print(feature.shape)
        output = self.fc(feature.view(feature.shape[0], -1))
        return output

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i ==0 :
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)

def VGG(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1),
        vgg_block(num_convs, in_channels, out_channels))

    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features,fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units,fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units,36)
                                       ))
    return net

def IfAccurate(y_hat, target):
    if(int(target.item())>=10 and int(target.item())<=35):
        if(y_hat.item()==int(target.item()) or y_hat.item()==(int(target.item())+26)):
            return 1
        else:
            return 0
    elif(int(target.item())>35):
        if(y_hat.item()==int(target.item()) or y_hat.item()==(int(target.item())-26)):
            return 1
        else:
            return 0
    else:
        if(y_hat.item()==int(target.item())):
            return 1
        else:
            return 0

def evaluate_acc(dataset, net):
    acc_sum, n = 0.0, 0
    if (torch.cuda.is_available()):
        X = dataset[0]
        X = torch.from_numpy(X).cuda()
        X = X[0]
        label = dataset[1]
        y = torch.from_numpy(label).cuda()
        y = y[0]
    else:
        X = dataset[0]
        X = torch.from_numpy(X)
        y = dataset[1]
        y = torch.Tensor([y])

    torch_dataset = Data.TensorDataset(X, y)
    # for i in range(X.shape[0]):
    test_loader = Data.DataLoader(
        dataset=torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
        batch_size=4,  # 每块的大小
        shuffle=True,  # 要不要打乱数据
    )
    for step, (batch_x, batch_y) in enumerate(test_loader):
        # xx = X[i,:,:]
        xx = batch_x.reshape([4,224,224])
        y_hat = net(xx)
        # y_hat = net(xx).argmin(dim=1)
        # print(y_hat.argmax(dim=1), batch_y)
        # for i in range(len(batch_y)):
        #     IfAcc = IfAccurate(y_hat[i], batch_y[i])
            # acc_sum += (IfAcc).float().sum().item()
        acc_sum += (y_hat.argmax(dim=1) == batch_y).float().sum().item()
            # acc_sum += IfAcc
        n += batch_y.shape[0]
    return acc_sum/n

def l1_regularization(model, l1_alpha):
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            module.weight.grad.data.add_(l1_alpha * torch.sign(module.weight.data))

def train(net,traindata,testdata,loss,num_epochs,
          BATCH_SIZE ,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, l = 0.0, 0.0, 0, 0.0
        if(torch.cuda.is_available()):
            X = traindata[0]
            X = torch.from_numpy(X).cuda()
            X = X[0]
            label = traindata[1]
            y = torch.from_numpy(label).cuda()
            y = y[0]
            net.cuda()
        else:
            X = traindata[0]
            X = torch.from_numpy(X)
            X = X[0]
            label = traindata[1]
            y = torch.from_numpy(label)
            y = y[0]

        # y = torch.Tensor([label])
        torch_dataset = Data.TensorDataset(X, y)

        loader = Data.DataLoader(
            dataset = torch_dataset,  # 数据，封装进Data.TensorDataset()类的数据
            batch_size = BATCH_SIZE,  # 每块的大小
            shuffle = True,  # 要不要打乱数据
        )

        for step, (batch_x, batch_y) in enumerate(loader):
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            y_hat = net(batch_x)
            l = loss(y_hat, batch_y.long()).sum()
            # print(l.item())
            # if(l.item()>1e29):
            #     l = 0
            l.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 1e10)
            l1_regularization(net,2)
            if optimizer is None:
                optimizer=optim.SGD(net.parameters(), lr=0.1)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == batch_y).float().sum().item()
            # print(y_hat.argmax(dim=1), batch_y)
            n += batch_y.shape[0]

            # print('iteration %d, loss %.4f, train acc %.3f' % (iter, train_l_sum/n, train_acc_sum /n))
        test_acc = evaluate_acc(testdata, net)
        print('epoch %d, loss %.4f, train acc %.3f,test acc %.3f'
              % (epoch + 1, train_l_sum/n, train_acc_sum /n, test_acc))

def mix(data_, label_):
    N = data_.shape[1]
    data = np.zeros((1, 8000, 224, 224)).astype(np.float32)
    label = np.zeros((1, 8000)).astype(np.float32)
    DATA = []
    for i in range(N):
        x = data_[0,i,:,:]
        y = label_[0,i]
        DATA.append((x,y))
    random.shuffle(DATA)
    for i in range(N):
        data[0,i,:,:] = DATA[i][0]
        label[0,i] = DATA[i][1]
    return data, label

if __name__=='__main__':
    # 准备训练集和测试集数据
    img_path = "cut"
    label_path = "label.txt"
    data, label = ReadPic(img_path, label_path)
    # data = Normalize(data)
    data, label = mix(data, label)
    TrainData = [data[:,0:7000,:,:], label[:,0:7000]]
    TestData = [data[:,7000:8000,:,:], label[:,7000:8000]]

    num_input,  num_output = 224*224,  36
    batch_size = 5
    num_epochs = 200

    # AlexNet
    net = AlexNet(num_input, num_output, batch_size)
    # VGGNet
    # conv_arch = ((1,1,64), (1,64,128), (2,128,256), (2,256,512), (2,512,512))
    # fc_features =512*7*7
    # fc_hidden_units = 4096
    # net = VGG(conv_arch,fc_features,fc_hidden_units)

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    train(net, TrainData, TestData, loss, num_epochs, BATCH_SIZE = batch_size, optimizer=optimizer)

    SAVE_PATH = "improved_model.pth"
    torch.save(net.state_dict(), SAVE_PATH)

