import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pathlib
import torch.utils.data
from sklearn.preprocessing import MultiLabelBinarizer

import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import PIL

PATH_TO_IMAGES = 'input/train/'
PATH_TO_TEST_IMAGES = 'input/test/'
PATH_TO_META = 'input/train.csv'
SAMPLE_SUBMI = 'input/sample_submission.csv'

LABEL_MAP = {
0: "Nucleoplasm" ,
1: "Nuclear membrane"   ,
2: "Nucleoli"   ,
3: "Nucleoli fibrillar center",   
4: "Nuclear speckles"   ,
5: "Nuclear bodies"   ,
6: "Endoplasmic reticulum"   ,
7: "Golgi apparatus"  ,
8: "Peroxisomes"   ,
9:  "Endosomes"   ,
10: "Lysosomes"   ,
11: "Intermediate filaments"  , 
12: "Actin filaments"   ,
13: "Focal adhesion sites"  ,
14: "Microtubules"   ,
15: "Microtubule ends"   ,
16: "Cytokinetic bridge"   ,
17: "Mitotic spindle"  ,
18: "Microtubule organizing center",  
19: "Centrosome",
20: "Lipid droplets"   ,
21: "Plasma membrane"  ,
22: "Cell junctions"   ,
23: "Mitochondria"   ,
24: "Aggresome"   ,
25: "Cytosol" ,
26: "Cytoplasmic bodies",
27: "Rods & rings"}

cuda = torch.device('cuda')

def predict_submission(model, submission_load):
    all_preds = []
    # model.eval()
    for i, b in enumerate(submission_load, 0):
        if i % 100: print('processing batch {}/{}'.format(i, len(submission_load)))
        X, labels = b
        
        # if torch.cuda.is_available():
        #     X = X.cuda()
        pred = model(X)
        all_preds.append(pred.sigmoid().cpu().data.numpy())
    return np.concatenate(all_preds)
        
         
def make_submission_file(sample_submission_df, predictions):
    submissions = []
    for row in predictions:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('submission.csv', index=None)
    
    return sample_submission_df

class MultiBandMultiLabelDataset(Dataset):
    # BANDS_NAMES = ['_red.png','_green.png','_blue.png','_yellow.png']
    BANDS_NAMES = ['_green.png']
        
    def __len__(self):
        return len(self.images_df)
    
    def __init__(self, images_df, 
                 base_path, 
                 image_transform, 
                 augmentator=None,
                 train_mode=True    
                ):
        if not isinstance(base_path, pathlib.Path):
            base_path = pathlib.Path(base_path)
            
        self.images_df = images_df.copy()
        self.image_transform = image_transform
        self.augmentator = augmentator
        self.images_df.Id = self.images_df.Id.apply(lambda x: base_path / x)
        self.mlb = MultiLabelBinarizer(classes=list(LABEL_MAP.keys()))
        self.train_mode = train_mode

                                      
        
    def __getitem__(self, index):
        y = None
        X = self._load_multiband_image(index)
        if self.train_mode:
            y = self._load_multilabel_target(index)
        
        # augmentator can be for instance imgaug augmentation object
        if self.augmentator is not None:
            X = self.augmentator(X)
            
        X = self.image_transform(X)
            
        return X, y 
        
    def _load_multiband_image(self, index):
        row = self.images_df.iloc[index]
        image_bands = []
        for band_name in self.BANDS_NAMES:
            p = str(row.Id.absolute()) + band_name
            pil_channel = PIL.Image.open(p)
            image_bands.append(pil_channel)
                    
        # lets pretend its a RBGA image to support 4 channels
        # band4image = PIL.Image.merge('RGBA', bands=image_bands)
        # return band4image
        return image_bands[0]
    
    
    def _load_multilabel_target(self, index):
        return list(map(int, self.images_df.iloc[index].Target.split(' ')))
    
        
    def collate_func(self, batch):
        labels = None
        images = [x[0] for x in batch]
        
        if self.train_mode:
            labels = [x[1] for x in batch]
            labels_one_hot  = self.mlb.fit_transform(labels)
            labels = torch.FloatTensor(labels_one_hot)
            
        
        return torch.stack(images)[:,:4,:,:], labels

SEED = 666
DEV_MODE = True

df = pd.read_csv(PATH_TO_META)
df_train, df_test  = train_test_split(df, test_size=0.2, random_state=SEED)
df_submission = pd.read_csv(SAMPLE_SUBMI)

#if DEV_MODE:
df_train = df_train[:1000]
# df_test = df_test[:1000]
# df_submission = df_submission[:200]

image_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=4)

trainset = MultiBandMultiLabelDataset(df_train, base_path=PATH_TO_IMAGES, image_transform=image_transform)
testset = MultiBandMultiLabelDataset(df_test, base_path=PATH_TO_IMAGES, image_transform=image_transform)
subset = MultiBandMultiLabelDataset(df_submission, base_path=PATH_TO_TEST_IMAGES, train_mode=False, image_transform=image_transform)

trainloader = DataLoader(trainset, collate_fn=trainset.collate_func, batch_size=256, num_workers=6)
testloader = DataLoader(testset, collate_fn=testset.collate_func, batch_size=256, num_workers=6)
submissionloader = DataLoader(subset, collate_fn=subset.collate_func, batch_size=256, num_workers=6)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=28):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet34()
    
    # net.load_state_dict(torch.load('./resnet2.pth'))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    
    for epoch in range(1) :
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0) :
            inputs, labels = data
            inputs.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs)
            # loss = criterion(outputs, torch.max(labels, 1)[1])
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('[%d, %5d] loss : %.3f' %(epoch + 1, i + 1, running_loss / 1000))
            if i % 30 == 29 :
                print('[%d, %5d] loss : %.3f' %(epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')
    torch.save(net.state_dict(), './resnet2.pth')
    # correct = 0
    # total = 0
    
    # all_preds = []
    # true = []
    # THRESHOLD = 0.2
    # net.eval()

    # for data in testloader :
    #     images, labels = data
    #     outputs = net(images)
    #     # _, predicted = torch.max(outputs.data, 1)
    #     all_preds.append(outputs.sigmoid().cpu().data.numpy())
    #     true.append(labels.cpu().data.numpy())
        
    # predicted = np.concatenate(all_preds)
    # labels = np.concatenate(true)

    # f1 = f1_score(predicted>THRESHOLD, labels, average='macro')

    # print(f1)

    # total += labels.size(0)
    # correct += (predicted ==  torch.max(labels, 1)[1]).sum().item()
    # correct += (predicted>THRESHOLD == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images : %d %%' % (100 * correct / total))

    # submission_predictions =predict_submission(net, submissionloader)
    # THRESHOLD = 0.03
    # # print(submission_predictions)
    # p = submission_predictions>THRESHOLD
    # # print(p)
    # submission_file = make_submission_file(sample_submission_df=df_submission, predictions=p)

    #submission_file.head()

    
test()