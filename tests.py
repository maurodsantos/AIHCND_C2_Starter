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

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve, auc
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


torch.manual_seed(0)
np.random.seed(0)
BATCH_SIZE = 256
NUM_EPOCHS = 30
LOG_SIZE = 50
EXP_NAME = 'tests_balance'
TL_MODEL = 'resnet50_v2'
MODEL_NAME = 'PneumoNet'
BALANCE_TRAINING = True
# %%

print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.cuda.current_device()", torch.cuda.current_device())
print("torch.cuda.device(0)", torch.cuda.device(0))
print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
if( torch.cuda.device_count()>1):
    print("torch.cuda.device(1)", torch.cuda.device(1))
    print("torch.cuda.get_device_name(1)", torch.cuda.get_device_name(1))
else:
    print("only 1 cuda enabled device")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% md

## Do some early processing of your metadata for easier model training:

# %%
## Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation


def load_xray_data():
    """ load data and preprocess required fields"""
    df = pd.read_csv('data/Data_Entry_2017.csv')
    #df = df.iloc[0:math.floor(len(df)/4), :]

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

def create_splits_test(df, val_prop, test_prop, class_name, balanced=True):
    ## Either build your own or use a built-in library to split your original dataframe into two sets
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia

    # split data by patients:
    patient_df = df.groupby(['Patient ID']).first()

    train_patient_df, test_patient_df = train_test_split(patient_df, stratify=patient_df[class_name],
                                                         test_size=test_prop, random_state=0)

    train_patient_df, validation_patient_df = train_test_split(train_patient_df, stratify=train_patient_df[class_name],
                                                         test_size=val_prop, random_state=0)

    train_patient_df = df[df['Patient ID'].isin(train_patient_df.index.values)]

    if balanced:
        train_patient_df = train_patient_df.groupby(class_name).apply(lambda data: data.sample(n=len(train_patient_df) // 2, replace=True))

    validation_patient_df = df[df['Patient ID'].isin(validation_patient_df.index.values)]

    test_patient_df = df[df['Patient ID'].isin(test_patient_df.index.values)]

    return train_patient_df, validation_patient_df, test_patient_df


train_data, val_data, test_data = create_splits_test(xray_df, 0.2, 0.1, 'Pneumonia')


def create_splits(df, val_prop, class_name):
    ## Either build your own or use a built-in library to split your original dataframe into two sets
    ## that can be used for training and testing your model
    ## It's important to consider here how balanced or imbalanced you want each of those sets to be
    ## for the presence of pneumonia

    # split data by patients:
    patient_df = df.groupby(['Patient ID']).first()

    train_patient_df, validation_patient_df = train_test_split(patient_df, stratify=patient_df[class_name],
                                                         test_size=val_prop, random_state=0)

    train_patient_df = df[df['Patient ID'].isin(train_patient_df.index.values)]
    validation_patient_df = df[df['Patient ID'].isin(validation_patient_df.index.values)]

    return train_patient_df, validation_patient_df


#train_data, val_data = create_splits(xray_df, 0.3, 'Pneumonia')

print('train data, n = {}({}% of the data)'.format(len(train_data), round(len(train_data) / len(xray_df) * 100, 2)))
print('validation, n = {}({}% of the data)'.format(len(val_data), round(len(val_data) / len(xray_df) * 100, 2)))
print('test, n = {}({}% of the data)'.format(len(test_data), round(len(test_data) / len(xray_df) * 100, 2)))

print('prop. pneumonia train data: ' + str(round(train_data['Pneumonia'].sum() / len(train_data), 4)))
print('prop. pneumonia validation data: ' + str(round(val_data['Pneumonia'].sum() / len(val_data), 4)))
print('prop. pneumonia test data: ' + str(round(test_data['Pneumonia'].sum() / len(test_data), 4)))


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


def my_image_augmentation():
    ## recommendation here to implement a package like Keras' ImageDataGenerator
    ## with some of the built-in augmentations

    ## keep an eye out for types of augmentation that are or are not appropriate for medical imaging data
    ## Also keep in mind what sort of augmentation is or is not appropriate for testing vs validation data

    ## STAND-OUT SUGGESTION: implement some of your own custom augmentation that's *not*
    ## built into something like a Keras package
    transformations = transforms.Compose([  # transforms.ToPILImage(),
        # transforms.CenterCrop(224),  #
        transforms.Resize(224),
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
    train_gen = ImageDataset(trainset, transformations)
    trainloader = DataLoader(dataset=train_gen, batch_size=batch_size, shuffle=True)

    return trainloader


def make_val_gen(valset, batch_size, shuffle=True):
    #     val_gen = my_val_idg.flow_from_dataframe(dataframe = val_data,
    #                                              directory=None,
    #                                              x_col = ,
    #                                              y_col = ',
    #                                              class_mode = 'binary',
    #                                              target_size = ,
    #                                              batch_size = )

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    val_gen = ImageDataset(valset, transformations)
    valloader = DataLoader(dataset=val_gen, batch_size=batch_size, shuffle=shuffle)

    return valloader


# %%
## May want to pull a single large batch of random validation data for testing after each epoch:
val_gen = make_val_gen(val_data, BATCH_SIZE)
# valX, valY = next(iter(val_gen))
# %%
train_gen = make_train_gen(train_data, BATCH_SIZE, my_image_augmentation())
# trainX, trainY = next(iter(train_gen))

# %% md
## Build your model:
def load_pretrained_model():
    # model = VGG16(include_top=True, weights='imagenet')
    # transfer_layer = model.get_layer(lay_of_interest)
    # vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)

    # Todo
    model = models.wide_resnet50_2(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model
# %%

class PneumoNet(nn.Module):
    def __init__(self, out_size):
        super(PneumoNet, self).__init__()
        self.model = load_pretrained_model()

        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
                      nn.BatchNorm1d(num_features),
                      nn.Linear(num_features, 256),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(256, out_size))

    def forward(self, x):
        x = self.model(x)
        return x


def build_my_model():
    # ....add your pre-trained model, and then whatever additional layers you think you might
    # want for fine-tuning (Flatteen, Dense, Dropout, etc.)

    # if you want to compile your model within this function, consider which layers of your pre-trained model,
    # you want to freeze before you compile

    # also make sure you set your optimizer, loss function, and metrics to monitor

    return PneumoNet(1).to(device)

model = build_my_model()
## give different weights to each class, up-weighting the minor class such that it balances the numbers in the majors class
# class_weights = [train_data['Pneumonia'].sum()/train_data['Pneumonia'].size,
#                  (train_data['Pneumonia'] == 0).sum()/train_data['Pneumonia'].size]
# class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
# criterion = nn.CrossEntropyLoss(weight=class_weights)# this includes a LogSoftmax layer added after the Linear layer
criterion = nn.BCEWithLogitsLoss()
#
# Decays the learning rate of each parameter group by gamma every step_size epochs. Notice that such decay can happen simultaneously with other changes to the learning rate from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
#
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#optimizer = optim.RMSprop(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=4)

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


def accuracy(preds, labels):
    # _, preds = torch.max(outputs, dim=1)
    return np.sum(preds == labels) / len(preds)


def print_metrics(phase, batch_number, total_batches, loss, acc, auc, prauc, f1score):
    print("\r{} batch {}/{}; loss {:.4f}; acc {:.4f}; auc {:.4f}; prauc {:.4f}, f1score {:.4f}".format(phase, batch_number, total_batches,
                                                                                     loss, acc,
                                                                                     auc, prauc,
                                                                                     f1score))


def train_model(dnn_model, model_criterion, model_optimizer, scheduler, num_epochs=10, log_size=100, patience=10):


    since = time.time()
    best_model_wts = copy.deepcopy(dnn_model.state_dict())
    best_acc = 0.0

    val_epoch_list = []
    val_iteration_list = []
    val_iteration_acc_list = []
    train_epoch_list = []
    train_iteration_list = []
    train_iteration_acc_list = []

    train_avg_loss_list = []
    train_avg_acc_list = []
    train_avg_auc_list = []
    train_avg_prauc_list = []
    train_avg_f1score_list = []

    val_avg_loss_list = []
    val_avg_acc_list = []
    val_avg_auc_list = []
    val_avg_prauc_list = []
    val_avg_f1score_list = []

    train_batches_size = len(train_gen)
    val_batches_size = len(val_gen)

    patience_aux = 0

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 40)

        loss_train = 0
        acc_train = 0
        prauc_train = 0
        f1score_train = 0
        auc_train = 0
        train_auc_count = 0

        dnn_model.train(True)

        for i, data in enumerate(train_gen):

            if i % log_size == 0:

                if i > 0:
                    avg_loss_aux = loss_train / (i+1)
                    avg_acc_aux = acc_train / (i+1)
                    avg_auc_aux = auc_train / train_auc_count
                    avg_prauc_aux = prauc_train / train_auc_count
                    avg_f1score_aux = f1score_train / train_auc_count
                    print_metrics('Train', i, train_batches_size, avg_loss_aux, avg_acc_aux, avg_auc_aux, avg_prauc_aux, avg_f1score_aux)
                    train_avg_loss_list.append(avg_loss_aux)
                    train_avg_acc_list.append(avg_acc_aux)
                    train_avg_prauc_list.append(avg_prauc_aux)
                    train_avg_f1score_list.append(avg_f1score_aux)
                    train_avg_auc_list.append(avg_auc_aux)
                    train_epoch_list.append(epoch)
                    train_iteration_list.append(i)
                    train_iteration_acc_list.append(epoch*train_batches_size+i)

            # Use half training dataset
            # if i >= 101:
            #     break

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.float()
            #print("\rTrain class0 {}; class1 {} ".format(sum(labels == 0), sum(labels)))
            optimizer.zero_grad()
            outputs = dnn_model(inputs)
            loss = model_criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            model_optimizer.step()

            loss_train += loss.item()
            labels_cpu = labels.cpu().detach().numpy()
            prob_cpu = torch.sigmoid(outputs).cpu().detach().numpy()
            preds_cpu = np.round(prob_cpu.reshape(-1))
            acc_train += accuracy(preds_cpu, labels_cpu)

            if (len(np.unique(labels_cpu)) > 1):
                train_auc_count = train_auc_count + 1
                f1score_train += f1_score(labels_cpu, preds_cpu)
                # calculate the precision-recall auc
                precision, recall, _ = precision_recall_curve(labels_cpu, prob_cpu)
                prauc_train += auc(recall, precision)
                auc_train += roc_auc_score(labels_cpu, prob_cpu)

            scheduler.step()

        del inputs, labels, outputs
        torch.cuda.empty_cache()

        train_avg_loss = loss_train / train_batches_size
        train_avg_acc = acc_train / train_batches_size
        train_avg_auc = auc_train / train_auc_count
        train_avg_prauc = prauc_train / train_auc_count
        train_avg_f1score = f1score_train / train_auc_count

        train_avg_loss_list.append(train_avg_loss)
        train_avg_acc_list.append(train_avg_acc)
        train_avg_prauc_list.append(train_avg_prauc)
        train_avg_f1score_list.append(train_avg_f1score)
        train_avg_auc_list.append(train_avg_auc)
        train_epoch_list.append(epoch)
        train_iteration_list.append(i)
        train_iteration_acc_list.append(epoch * train_batches_size + i)
        # validation phase
        dnn_model.train(False)
        dnn_model.eval()
        loss_val = 0
        acc_val = 0
        prauc_val = 0
        f1score_val = 0
        auc_val = 0
        val_auc_count = 0

        with torch.no_grad():
            for i, data in enumerate(val_gen):
                if i % log_size == 0:
                    if i > 0:
                        avg_loss_val_aux = loss_val / (i+1)
                        avg_acc_val_aux = acc_val / (i+1)
                        avg_auc_val_aux = auc_val / (val_auc_count)
                        avg_prauc_val_aux = prauc_val / val_auc_count
                        avg_f1score_val_aux = f1score_train / val_auc_count
                        print_metrics('Validation', i, val_batches_size, avg_loss_val_aux, avg_acc_val_aux, avg_auc_val_aux, avg_prauc_val_aux, avg_f1score_val_aux)

                        val_avg_loss_list.append(avg_loss_val_aux)
                        val_avg_acc_list.append(avg_acc_val_aux)
                        val_avg_prauc_list.append(avg_prauc_val_aux)
                        val_avg_f1score_list.append(avg_f1score_val_aux)
                        val_avg_auc_list.append(avg_f1score_val_aux)
                        val_epoch_list.append(epoch)
                        val_iteration_list.append(i)
                        val_iteration_acc_list.append(epoch * val_batches_size + i)
                # if i >= 101:
                #     break

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.float()
                #print("\rValidation class0 {}; class1 {} ".format(sum(labels == 0), sum(labels)))

                optimizer.zero_grad()
                outputs = dnn_model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))

                loss_val += loss.item()
                labels_cpu = labels.cpu().detach().numpy()
                prob_cpu = torch.sigmoid(outputs).cpu().detach().numpy()
                preds_cpu = np.round(prob_cpu.reshape(-1))
                acc_val += accuracy(preds_cpu, labels_cpu)

                if(len(np.unique(labels_cpu))>1):
                    val_auc_count = val_auc_count+1
                    auc_val += roc_auc_score(labels_cpu, prob_cpu)
                    f1score_val += f1_score(labels_cpu, preds_cpu)
                    precision, recall, _ = precision_recall_curve(labels_cpu, prob_cpu)
                    prauc_val += auc(recall, precision)

            del inputs, labels, outputs
            torch.cuda.empty_cache()

        val_avg_loss = loss_val / val_batches_size
        val_avg_acc = acc_val / val_batches_size
        val_avg_auc = auc_val / val_auc_count
        val_avg_prauc = prauc_val / val_auc_count
        val_avg_f1score = f1score_val / val_auc_count

        val_avg_loss_list.append(val_avg_loss)
        val_avg_acc_list.append(val_avg_acc)
        val_avg_prauc_list.append(val_avg_prauc)
        val_avg_f1score_list.append(val_avg_f1score)
        val_avg_auc_list.append(val_avg_auc)
        val_epoch_list.append(epoch)
        val_iteration_list.append(i)
        val_iteration_acc_list.append(epoch * val_batches_size + i)

        print("\rEpoch {}, Training loss/acc/auc/prauc/f1score: {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}; "
              "Validation loss/acc/auc/prauc/f1score: {:.4f} / {:.4f} / {:.4f} / {:.4f} / {:.4f}".
              format(epoch, train_avg_loss, train_avg_acc, train_avg_auc, train_avg_prauc, train_avg_f1score,
                     val_avg_loss, val_avg_acc, val_avg_auc, val_avg_prauc, val_avg_f1score))

        if val_avg_acc > best_acc:
            print("val_binary_accuracy improved from {}".format(best_acc))
            print('-' * 40)
            best_acc = val_avg_acc
            best_model_wts = copy.deepcopy(dnn_model.state_dict())
            torch.save(dnn_model.state_dict(), '{}_{}_weights_checkpoint_{}_{}_{}.h5'.format(MODEL_NAME,TL_MODEL,EXP_NAME,BATCH_SIZE, epoch))

        else:
            patience_aux = patience_aux+1
            print("val_binary_accuracy did not improve from {}".format(best_acc))
            print('-' * 40)
            if patience_aux > patience:
                break

    elapsed_time = time.time() - since
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))

    dnn_model.load_state_dict(best_model_wts)

    history_train = {}
    history_train['epoch'] = train_epoch_list
    history_train['iteration'] = train_iteration_list
    history_train['iteration_acc'] = train_iteration_acc_list
    history_train['avg_loss'] = train_avg_loss_list
    history_train['avg_acc']  = train_avg_acc_list
    history_train['avg_auc'] = train_avg_auc_list
    history_train['avg_prauc'] = train_avg_prauc_list
    history_train['avg_f1score'] = train_avg_f1score_list

    history_val = {}
    history_val['epoch'] = val_epoch_list
    history_val['iteration'] = val_iteration_list
    history_val['iteration_acc'] = val_iteration_acc_list
    history_val['avg_loss']   = val_avg_loss_list
    history_val['avg_acc']    = val_avg_acc_list
    history_val['avg_auc'] = val_avg_auc_list
    history_val['avg_prauc'] = val_avg_prauc_list
    history_val['avg_f1score'] = val_avg_f1score_list

    return dnn_model, history_train, history_val


# %% md

### Start training!

final_model, history_train, history_val = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS, log_size=LOG_SIZE,patience=5)
torch.save(final_model.state_dict(), '{}_{}_weights_{}_{}_{}.h5'.format(MODEL_NAME,TL_MODEL,EXP_NAME,BATCH_SIZE, NUM_EPOCHS))
history_val_df = pd.DataFrame(history_val)
history_val_df.to_csv('{}_{}_history_val_{}_{}_{}.csv'.format(MODEL_NAME,TL_MODEL,EXP_NAME,BATCH_SIZE, NUM_EPOCHS))
history_train_df = pd.DataFrame(history_train)
history_train_df.to_csv('{}_{}_history_train_{}_{}_{}.csv'.format(MODEL_NAME,TL_MODEL,EXP_NAME,BATCH_SIZE, NUM_EPOCHS))