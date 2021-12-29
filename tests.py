# %% md

## Skeleton Code

# The
# code
# below
# provides
# a
# skeleton
# for the model building & training component of your project.You can add / remove / build on code however you see fit, this is meant as a starting point.

# %%
import math

import numpy as np# linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import time
import copy

##Import any other stats/DL/ML packages you may need here. E.g. Keras, scikit-learn, etc.
from itertools import chain
from sklearn.model_selection import train_test_split
from PIL import Image

## pytorch libs
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# %%

print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.cuda.current_device()", torch.cuda.current_device())
print("torch.cuda.device(0)", torch.cuda.device(0))
print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

if True and torch.cuda.is_available():
    device = torch.device('cuda')
    use_gpu = True
else:
    device = torch.device('cpu')
    use_gpu = False

# %% md

## Do some early processing of your metadata for easier model training:

# %%
## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation


def load_xray_data():
    """ load data and preprocess required fields"""
    df = pd.read_csv('data/Data_Entry_2017.csv')
    df = df.iloc[0:math.floor(len(df)/4), :]
    def convert(x):
        """assume that the first number is incorrect for ages above 110"""
        if x > 110:
            x = int(str(x)[1:-1])
        return x

    # convert ages above 110 to lower ages by assuming and non-random consisting error in the data-collection process.
    df['Patient Age'] = df['Patient Age'].apply(lambda x: convert(x))

    disease_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

    for label in disease_labels:
        df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)

    all_image_paths = {os.path.basename(x): x for x in
                       glob(os.path.join('data', 'images*', '*', '*.png'))}
    print('Scans found:', len(all_image_paths), ', Total Headers', df.shape[0])
    df['path'] = df['Image Index'].map(all_image_paths.get)

    print(df)
    return df, disease_labels


xray_df, disease_labels = load_xray_data()
# %%

## Here you may want to create some extra columns in your table with binary indicators of certain diseases
## rather than working directly with the 'Finding Labels' column

# Todo
# this is already done in the function load_xray_data() is lines:
# disease_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))

# for label in disease_labels:
#     df[label] = df['Finding Labels'].map(lambda x: 1 if label in x else 0)


# %%

## Here we can create a new column called 'pneumonia_class' that will allow us to look at
## images with or without pneumonia for binary classification

# Todo
# this is already done in the function load_xray_data(), of of the labels corresponds to the 'pneumonia' class.


# %% md

## Create your training and testing data:

# %%

def create_splits(df, test_prop, class_name):
    ## Either build your own or use a built-in library to split your original dataframe into two sets
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia

    # Todo
    # train_df, valid_df, test_df = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    # train_df, test_df = train_test_split(df, shuffle = True, test_size=0.1, random_state = 0)
    # test_df, valid_df = train_test_split(test_d, shuffle = True, test_size=0.5, random_state = 0)

    # split data by patients:
    patient_df = df.groupby(['Patient ID']).first()

    train_patient_df, valid_ptient_df = train_test_split(patient_df, stratify=patient_df[class_name],
                                                         test_size=test_prop)

    train_patient_df = df[df['Patient ID'].isin(train_patient_df.index.values)]
    valid_ptient_df = df[df['Patient ID'].isin(valid_ptient_df.index.values)]

    return train_patient_df, valid_ptient_df


train_data, val_data = create_splits(xray_df, 0.2, 'Pneumonia')

print('train data, n = {}({}% of the data)'.format(len(train_data), round(len(train_data) / len(xray_df) * 100, 2)))
print('validation, n = {}({}% of the data)'.format(len(val_data), round(len(val_data) / len(xray_df) * 100, 2)))

print('prop. pneumonia train data: ' + str(round(train_data['Pneumonia'].sum() / len(train_data), 4)))
print('prop. pneumonia validation data: ' + str(round(val_data['Pneumonia'].sum() / len(val_data), 4)))

# %% md
# Now we can begin our model-building & training

# %% md
#### First suggestion: perform some image augmentation on your data

# %%


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        self.data_frame = df
        self.transforms = transforms
        self.len = df.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]

        address = row['path']

        x = Image.open(address).convert('RGB')

        vec = np.array(row['Pneumonia'], dtype=float)
        y = torch.LongTensor(vec)

        if self.transforms:
            x = self.transforms(x)
        return x, y


def my_image_augmentation(vargs):
    ## recommendation here to implement a package like Keras' ImageDataGenerator
    ## with some of the built-in augmentations

    ## keep an eye out for types of augmentation that are or are not appropriate for medical imaging data
    ## Also keep in mind what sort of augmentation is or is not appropriate for testing vs validation data

    ## STAND-OUT SUGGESTION: implement some of your own custom augmentation that's *not*
    ## built into something like a Keras package

    # Todo

    transformations = transforms.Compose([  # transforms.ToPILImage(),
        # transforms.CenterCrop(224),  #
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8,
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transformations


def make_train_gen(trainset, batch_size, transformations):
    ## Create the actual generators using the output of my_image_augmentation for your training data
    ## Suggestion here to use the flow_from_dataframe library, e.g.:

    #     train_gen = my_train_idg.flow_from_dataframe(dataframe=train_df,
    #                                          directory=None,
    #                                          x_col = ,
    #                                          y_col = ,
    #                                          class_mode = 'binary',
    #                                          target_size = ,
    #                                          batch_size =
    #                                          )
    # Todo
    train_gen = ImageDataset(trainset, transformations)
    trainloader = DataLoader(dataset=train_gen, shuffle=True, batch_size=batch_size)
    return trainloader


def make_val_gen(valset, batch_size):
    #     val_gen = my_val_idg.flow_from_dataframe(dataframe = val_data,
    #                                              directory=None,
    #                                              x_col = ,
    #                                              y_col = ',
    #                                              class_mode = 'binary',
    #                                              target_size = ,
    #                                              batch_size = )

    # Todo
    transformations = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_gen = ImageDataset(valset, transformations)
    valloader = DataLoader(dataset=val_gen, batch_size=batch_size, shuffle=False)

    return valloader


# %%
## May want to pull a single large batch of random validation data for testing after each epoch:
val_gen = make_val_gen(val_data, 20)
# valX, valY = next(iter(val_gen))
# %%
train_gen = make_train_gen(train_data, 20, my_image_augmentation(True))
# trainX, trainY = next(iter(train_gen))

# %% md
## Build your model:

# %%

class PneumoNet(nn.Module):
    def __init__(self, out_size):
        super(PneumoNet, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        for param in self.vgg16.parameters():
            param.requires_grad = False

        num_features = self.vgg16.classifier[6].in_features
        features = list(self.vgg16.classifier.children())[:-1] # Remove last layer
        # #    The VGG-16 is able to classify 1000 different labels; we just need 2 instead. In order to do that we are going replace the last fully connected layer of the model with a new one with 4 output features instead of 1000.
        # #    In PyTorch, we can access the VGG-16 classifier with model.classifier, which is an 6-layer array. We will replace the last entry.
        features.extend([nn.Linear(num_features, out_size)])# add Linear layer
        self.vgg16.classifier = nn.Sequential(*features)

    def forward(self, x):
        x = self.vgg16(x)
        features = self.vgg16.classifier[6]
        return x, features

model = PneumoNet(2).to(device)

criterion = nn.CrossEntropyLoss()# this includes a LogSoftmax layer added after the Linear layer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decays the learning rate of each parameter group by gamma every step_size epochs. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# %%

## STAND-OUT Suggestion: choose another output layer besides just the last classification layer of your modele
## to output class activation maps to aid in clinical interpretation of your model's results


# %%

## Below is some helper code that will allow you to add checkpoints to your model,
## This will save the 'best' version of your model by comparing it to previous epochs of training

## Note that you need to choose which metric to monitor for your model's 'best' performance if using this code.
## The 'patience' parameter is set to 10, meaning that your model will train for ten epochs without seeing
## improvement before quitting

# Todo

# weight_path="{}_my_model.best.hdf5".format('xray_class')

# checkpoint = ModelCheckpoint(weight_path,
#                              monitor= CHOOSE_METRIC_TO_MONITOR_FOR_PERFORMANCE,
#                              verbose=1,
#                              save_best_only=True,
#                              mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC,
#                              save_weights_only = True)

# early = EarlyStopping(monitor= SAME_AS_METRIC_CHOSEN_ABOVE,
#                       mode= CHOOSE_MIN_OR_MAX_FOR_YOUR_METRIC,
#                       patience=10)

# callbacks_list = [checkpoint, early]



def train_model(vgg, model_criterion, model_optimizer, num_epochs=10, patience=5):

    history_aux = {}
    since = time.time()
    best_model_wts = copy.deepcopy(vgg.state_dict())
    best_acc = 0.0

    train_avg_loss_list = []
    train_avg_acc_list = []
    val_avg_loss_list = []
    val_avg_acc_list = []

    train_batches = len(train_gen)
    val_batches = len(val_gen)

    train_size = train_gen.dataset.len
    val_size = val_gen.dataset.len

    patience_aux = 0

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        loss_train = 0
        loss_val = 0
        acc_train = 0
        acc_val = 0
        avg_loss_aux = 0
        avg_acc_aux = 0
        avg_loss_val_aux = 0
        avg_acc_val_aux = 0

        vgg.train(True)

        for i, data in enumerate(train_gen):

            if i % 100 == 0:

                if i > 0:
                    avg_loss_aux = loss_train / ((i+1)*len(data[0]))
                    avg_acc_aux = acc_train / ((i+1)*len(data[0]))
                print("\rTraining batch {}/{}; mean_loss {}, mean acc {} ".format(i, train_batches, avg_loss_aux, avg_acc_aux), end='', flush=True)

            # # Use half training dataset
            # if i >= train_batches / 2:
            #     break

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs[0], 1)
            loss = model_criterion(outputs[0], labels)

            loss.backward()
            model_optimizer.step()

            loss_train += loss
            acc_train += torch.sum(preds == labels)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        print()
        # * 2 as we only used half of the dataset
        train_avg_loss = loss_train / train_size
        train_avg_acc = acc_train / train_size

        train_avg_loss_list.append(train_avg_loss)
        train_avg_acc_list.append(train_avg_acc)

        vgg.train(False)
        vgg.eval()

        for i, data in enumerate(val_gen):
            if i % 100 == 0:
                if i > 0:
                    avg_loss_val_aux = loss_val / ((i+1)*len(data[0]))
                    avg_acc_val_aux = acc_val / ((i+1)*len(data[0]))
                print("\rValidation batch {}/{}; avg loss {}; avg acc {}".format(i, val_batches, avg_loss_val_aux, avg_acc_val_aux), end='', flush=True)

            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            optimizer.zero_grad()

            outputs = vgg(inputs)

            _, preds = torch.max(outputs[0], 1)
            loss = criterion(outputs[0], labels)

            loss_val += loss
            acc_val += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        val_avg_loss = loss_val / val_size
        val_avg_acc = acc_val / val_size
        val_avg_loss_list.append(val_avg_loss)
        val_avg_acc_list.append(val_avg_acc)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(train_avg_loss))
        print("Avg acc (train): {:.4f}".format(train_avg_acc))
        print("Avg loss (val): {:.4f}".format(val_avg_loss))
        print("Avg acc (val): {:.4f}".format(val_avg_acc))
        print('-' * 10)
        print()

        if val_avg_acc > best_acc:
            print("val_binary_accuracy improved from {}".format(best_acc))
            best_acc = val_avg_acc
            best_model_wts = copy.deepcopy(vgg.state_dict())
            torch.save(vgg.state_dict(), 'PneumoVGG16_checkpoint.pt')

        else:
            patience_aux = patience_aux+1
            print("val_binary_accuracy did not improve from {}".format(best_acc))
            if patience_aux > patience:
                break

    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    vgg.load_state_dict(best_model_wts)

    history_aux['train_avg_loss'] = train_avg_loss_list
    history_aux['train_avg_acc']  = train_avg_acc_list
    history_aux['val_avg_loss']   = val_avg_loss_list
    history_aux['val_avg_acc']    = val_avg_acc_list

    return vgg, history_aux


# %% md

### Start training!

vgg16, history = train_model(model, criterion, optimizer, num_epochs=10)
torch.save(vgg16.state_dict(), 'PneumoVGG16_weights.pt')
history_df = pd.DataFrame(history)
history_df.to_csv('PneumoVGG16_history.csv')

