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
from tqdm import tqdm

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
    sample_submission_df.to_csv('submission_ver2.csv', index=None)
    
    return sample_submission_df

class MultiBandMultiLabelDataset(Dataset):
    BANDS_NAMES = ['_red.png','_green.png','_blue.png','_yellow.png']
    # BANDS_NAMES = ['_green.png']
    # BANDS_NAMES = ['_red.png','_green.png','_blue.png']
        
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
        band4image = PIL.Image.merge('RGBA', bands=image_bands)
        return band4image
        # return image_bands
    
    
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
# df_train = df_train[:30]
# df_test = df_test[:1000]
# df_submission = df_submission[:200]

image_transform = transforms.Compose([
            transforms.Resize(224),
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

trainloader = DataLoader(trainset, collate_fn=trainset.collate_func, batch_size=100, num_workers=4)
testloader = DataLoader(testset, collate_fn=testset.collate_func, batch_size=100, num_workers=4)
submissionloader = DataLoader(subset, collate_fn=subset.collate_func, batch_size=100, num_workers=4)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()

        # self.resnet = torchvision.models.resnet34(pretrained=True)
        # self.fc = nn.Linear(512, 28)


        self.model = nn.Sequential(
            nn.BatchNorm2d(4),
            nn.Conv2d(4, 3, kernel_size=1, stride=1),
            torchvision.models.resnet34(pretrained=True),
            nn.Linear(1000, 512),
            nn.Linear(512, 128),
            nn.Linear(128, 28),
        ) # 파이팅!!!

    def forward(self, x):
        # x = self.resnet(x)
        # print(x.size())
        # x = self.fc(x)
        # print(x.size())
        # return x 
        return self.model(x)

def test():
    net = SampleNet()
    
    net.load_state_dict(torch.load('./resnet3.pth'))

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.MultiLabelSoftMarginLoss()
    # optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
    optimizer = optim.Adam(net.parameters(), lr = 0.001)
    
    for epoch in range(5) :
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainset)) :
            inputs, labels = data
            # print(inputs.size())
            # break
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
            # if i % 30 == 29 :
            #     print('[%d, %5d] loss : %.3f' %(epoch + 1, i + 1, running_loss / 1000))
            #     running_loss = 0.0
        torch.save(net.state_dict(), './resnet3.pth')
    print('Finished Training')
    



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

    submission_predictions =predict_submission(net, submissionloader)
    THRESHOLD = 0.03
    # print(submission_predictions)
    p = submission_predictions>THRESHOLD
    # print(p)
    submission_file = make_submission_file(sample_submission_df=df_submission, predictions=p)

    #submission_file.head()

    
test()