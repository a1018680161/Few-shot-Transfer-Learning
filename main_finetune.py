import torch
import torch.nn as nn
from utlis.models import CNN1d
from torch.autograd import Variable
import numpy as np
import os
import utlis.finetune_generator as tg
import argparse
import pickle
from utlis.logger import setlogger
import logging
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import Iterable

parser = argparse.ArgumentParser(description="fine-tuning")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=200)
parser.add_argument("-t", "--test_episode", type=int, default=1)
parser.add_argument("-ft", "--finetune_episode", type=int, default=50)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-d", "--datatype", type=str, default='fft')
parser.add_argument("-m", "--modeltype", type=str, default='1d')
parser.add_argument("-u", "--unfrozen", type=int, default=0)
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
FINETUNE_EPISODE = args.finetune_episode
CLASS_NUM_train = 8
CLASS_NUM_test = 5
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
DATATYPE = args.datatype
MODELTYPE = args.modeltype
UNFROZEN = args.unfrozen
batchsize = 1000
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# -----------------------------path---------------------------
root = '.\\result_finetune'
path1 = 'episode' + str(EPISODE) + '_' + str(SAMPLE_NUM_PER_CLASS) + 'shot_unfrozen' + str(UNFROZEN)
path2 = 'models_save'
modelname = 'episode' + str(EPISODE) + '.pkl'
model_path = os.path.join(root, path2, modelname)
path = os.path.join(root, path1)
if not os.path.exists("%s" % path):
    os.makedirs("%s" % path)
if not os.path.exists("%s" % os.path.join(root, path2)):
    os.makedirs("%s" % os.path.join(root, path2))

# ----------------------------logging-----------------------
setlogger(os.path.join("%s" % path, 'train.log'))
f = open("%s\\train.log" % path, 'w')
f.truncate()
f.close()


def set_freeze_by_id(model, layer_num_last):
    for param in model.parameters():
        param.requires_grad = False
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        for param in child.parameters():
            param.requires_grad = True


def mytrain(metatrain_character_folders):
    logging.info("init neural networks")
    feature_encoder = CNN1d(FEATURE_DIM)
    feature_encoder.cuda(GPU)
    classifier_train = nn.Linear(FEATURE_DIM * 25, CLASS_NUM_train)
    classifier_train.bias.data.fill_(0)
    feature_encoder_optim = torch.optim.Adam(list(feature_encoder.parameters())+list(classifier_train.parameters()), lr=LEARNING_RATE)
    logging.info("Training...")
    losses = []
    lepochs = []
    train_dataloader = tg.get_train_loader(metatrain_character_folders, batchsize, CLASS_NUM_train)
    bce = nn.CrossEntropyLoss().cuda(GPU)
    for episode in range(EPISODE):
        train_accuracy = 0
        for batch, (samples, sample_labels) in enumerate(train_dataloader):
            # calculate features
            sample_features_o = feature_encoder(Variable(samples).cuda(GPU).float())  # 1600
            relations = classifier_train(sample_features_o)
            loss = bce(relations, sample_labels)
            pred = relations.argmax(dim=1)
            correct = torch.eq(pred, sample_labels).float().sum().item()
            train_accuracy += correct
            feature_encoder.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
        train_accuracy = train_accuracy / len(train_dataloader.dataset)
        logging.info("episode:" + str(episode + 1) + "   loss: " + str(loss.item()) + 'acc:' + str(train_accuracy))
        losses.append(loss.item())
        lepochs.append(episode)
    torch.save(feature_encoder.state_dict(), os.path.join(model_path))
    return feature_encoder


def mytest(metatest_character_folders, feature_encoder):
    classifier_test = nn.Linear(FEATURE_DIM*25, CLASS_NUM_test)
    classifier_test.bias.data.fill_(0)
    set_freeze_by_id(feature_encoder, UNFROZEN+1)
    para = list(filter(lambda p: p.requires_grad, feature_encoder.parameters()))+list(classifier_test.parameters())
    finetune_optim = torch.optim.SGD(para, lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
    logging.info("testing...")
    bce = nn.CrossEntropyLoss().cuda(GPU)
    for episode in range(TEST_EPISODE):
        task = tg.finetuneTask(metatest_character_folders, SAMPLE_NUM_PER_CLASS)
        support_dataloader = tg.get_test_loader(task, SAMPLE_NUM_PER_CLASS, 'support', CLASS_NUM_test)
        test_dataloader = tg.get_test_loader(task, batchsize, 'test', CLASS_NUM_test)
        accuracys = []
        for i in range(FINETUNE_EPISODE):
            for batch, (sample_images, sample_labels) in enumerate(support_dataloader):
                sample_features_o = feature_encoder(Variable(sample_images).cuda(GPU).float())
                relations = classifier_test(sample_features_o)
                classifier_test.zero_grad()
                loss = bce(relations, sample_labels)
                loss.backward()
                finetune_optim.step()
            test_accuracy = 0
            for batch, (test_images, test_labels) in enumerate(test_dataloader):
                # calculate features
                sample_features_o = feature_encoder(Variable(test_images).cuda(GPU).float())  # 1600
                relations = classifier_test(sample_features_o)
                pred = relations.argmax(dim=1)
                correct = torch.eq(pred, test_labels).float().sum().item()
                test_accuracy += correct
            test_accuracy = test_accuracy / len(test_dataloader.dataset)
            logging.info("test accuracy:" + str(test_accuracy))
            accuracys.append(test_accuracy)
    plt.figure()
    plt.plot(accuracys)
    plt.savefig(path + '/test accuracy.jpg')
    # acc = np.mean(accuracys)
    # acc_std = np.std(accuracys)
    # fepisode = np.arange(FINETUNE_EPISODE)
    # plt.figure()
    # plt.plot(fepisode,accuracys)
    # plt.savefig(path + '/test accuracy.jpg')
    output = open('%s\\accuracy.pkl' % path, 'wb')
    pickle.dump(accuracys, output)
    output.close()
    logging.info('final accuracy:' + str(np.mean(accuracys[-10:])))
    return accuracys


def main():
    # Step 1: init data folders
    logging.info("init data folders")
    # init character folders for dataset construction
    datapath = '.\\tempdata\\' + '5way' + '.pkl'
    with open(datapath, 'rb') as pkl:
        metatrain_character_folders, metatest_character_folders = pickle.load(pkl)
    if not os.path.exists(model_path):
        feature_encoder = mytrain(metatrain_character_folders)
    else:
        feature_encoder = CNN1d(FEATURE_DIM)
        feature_encoder.cuda(GPU)
        feature_encoder.load_state_dict(torch.load(model_path))
    acc = mytest(metatest_character_folders, feature_encoder)


if __name__ == '__main__':
    main()
