import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
from scipy.io import loadmat
from scipy.fftpack import fft

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def pu_folders(CLASS_NUM):
    root = "D:\Study\Data\公开\轴承-PU"
    labels = ['KA01', 'KA03', 'KA05', 'KA07', 'KA08', 'KI01', 'KI03', 'KI07'] + ['K001'] + ['KA04', 'KB23',
                                                                                            'KB27', 'KI04']
    random.seed(1)
    # random.shuffle(labels)

    folds = [os.path.join(root, label) for label in labels]

    samples = dict()
    train_files = []
    test_files = []
    train_labels = []
    test_labels = []
    for c in folds[:-CLASS_NUM]:

        name0 = 'N09_M07_F10_'
        name1 = c.split('\\')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)) for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])

        train_labels.append(get_class(temp))
        train_files.append(file)
    for c in folds[-CLASS_NUM:]:

        name0 = 'N09_M07_F10_'
        name1 = c.split('\\')[-1] + '_'
        temps = [os.path.join(c, name0 + name1 + str(x)) for x in range(1, 21)]
        samples[c] = random.sample(temps, len(temps))
        part = samples[c]

        for i in range(part.__len__()):
            temp = part[i]
            data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
            data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
            if i == 0:
                file = data1
            else:
                file = np.vstack(
                    [file, data1])

        test_labels.append(get_class(temp))
        test_files.append(file)

    metatrain_folders = list(zip(train_files, train_labels))
    metatest_folders = list(zip(test_files, test_labels))
    return metatrain_folders, metatest_folders


def get_class(sample):
    return sample.split('\\')[-2]


class puTask(object):
    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num

        self.train_files = []
        self.test_files = []
        self.train_labels = []
        self.test_labels = []

        class_folders = random.sample(self.character_folders, self.num_classes)
        index = 0
        for class_folder in class_folders:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file[:self.train_num, :])
            self.test_files += list(file[self.train_num:self.train_num + self.test_num, :])
            self.train_labels += [index for i in range(train_num)]
            self.test_labels += [index for i in range(test_num)]
            index += 1


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None,dt='t',mt='1d'):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.dt = dt
        self.mt = mt
        self.image_files = self.task.train_files if self.split == 'train' else self.task.test_files
        self.labels = np.array(self.task.train_labels if self.split == 'train' else self.task.test_labels).reshape(-1)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class pu(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(pu, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_files[idx]
        if self.dt == 'fft':
            image = abs(fft(image - np.mean(image)))[0:1024]

        if self.mt == '2d':
            image = image[0:1024].reshape([1, 32, 32])
        elif self.mt == '1d':
            image = image[0:1024].reshape([1, 1024])
        label = self.labels[idx]
        return image, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train', shuffle=True,dt='t',mt='1d'):

    dataset = pu(task, split=split,
                 transform=transforms.ToTensor(),dt=dt,mt=mt)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader
