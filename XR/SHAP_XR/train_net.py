from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.autograd import Variable
from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

torch.manual_seed(1)


def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

img = get_image('./../data_alternate_baseline/train/pneumonia/00000193_019.png')
# plt.imshow(img)
# plt.show()

plt.ion()   # interactive mode
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './../data_alternate_baseline/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)
model_ft.eval()

###############################################################################
#####################  SHAP CODE ##############################################
###############################################################################
batch = next(iter(dataloaders['train']))
images, _ = batch
background = images[:30]
test_images = images[30:]

e = shap.DeepExplainer(model_ft, background)
shap_values = e.shap_values(test_images)


shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)*255

plot = shap.image_plot(shap_numpy, test_numpy, show = True)
plt.show()
plt.savefig("test.png")

# #############################################################################
# ###################  LIME CODE ##############################################
# #############################################################################
# # resize and take the center part of image to what our model expects
# def get_input_transform():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     transf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     return transf
#
# def get_input_tensors(img):
#     transf = get_input_transform()
#     # unsqeeze converts single image to batch of 1
#     return transf(img).unsqueeze(0)
#
#
# def get_pil_transform():
#     transf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.CenterCrop(224)
#     ])
#
#     return transf
#
# def get_preprocess_transform():
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                     std=[0.229, 0.224, 0.225])
#     transf = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     return transf
#
# pill_transf = get_pil_transform()
# preprocess_transform = get_preprocess_transform()
#
# def batch_predict(images):
#     model_ft.eval()
#     batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_ft.to(device)
#     batch = batch.to(device)
#
#     #logits =
#     probs = model_ft(batch)
#     return probs.detach().cpu().numpy()
#
#
# # Actual running of LIME
# # for im_name in os.listdir("./../data_alternate_baseline/val/normal"):
# #     img = get_image("./../data_alternate_baseline/val/normal/"+im_name)
# #     test_pred = batch_predict([pill_transf(img)])
# #     print(test_pred.squeeze().argmax())
# #
# #     explainer = lime_image.LimeImageExplainer()
# #     explanation = explainer.explain_instance(np.array(pill_transf(img)),
# #                                              batch_predict, # classification function
# #                                              top_labels=2,
# #                                              hide_color=0,
# #                                              num_samples=1000) # number of images that will be sent to classification function
# #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
# #     img_boundry2 = mark_boundaries(temp/255.0, mask)
# #     plt.imsave("./LIME_normal/"+ im_name,img_boundry2)
#
# for im_name in os.listdir("./../data_alternate_baseline/val/pneumonia"):
#     img = get_image("./../data_alternate_baseline/val/pneumonia/"+im_name)
#     test_pred = batch_predict([pill_transf(img)])
#     print(test_pred.squeeze().argmax())
#
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(np.array(pill_transf(img)),
#                                              batch_predict, # classification function
#                                              top_labels=2,
#                                              hide_color=0,
#                                              num_samples=1000) # number of images that will be sent to classification function
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
#     img_boundry2 = mark_boundaries(temp/255.0, mask)
#     plt.imsave("./LIME_pneumonia/"+ im_name,img_boundry2)
#
#
# # Evaluate performance of trained models
# # correct = 0
# # total = 0
# # for image in os.listdir("./../data_alternate_baseline/val/normal/"):
# #     im = Image.open("./../data_alternate_baseline/val/normal/"+ image)
# #     im = im.convert('RGB')
# #     im = data_transforms['train'](im).float()
# #     im = Variable(im, requires_grad = True)
# #     im = im.unsqueeze(0)
# #     #im = im.cuda()
# #     outputs = model_ft(im)
# #     _, preds = torch.max(outputs, 1)
# #     total +=1
# #     if int(preds.data[0]) == 0:
# #         correct +=1
# #
# # for image in os.listdir("./../data_alternate_baseline/val/pneumonia/"):
# #     im = Image.open("./../data_alternate_baseline/val/pneumonia/"+ image)
# #     im = im.convert('RGB')
# #     im = data_transforms['train'](im).float()
# #     im = Variable(im, requires_grad = True)
# #     im = im.unsqueeze(0)
# #     #im = im.cuda()
# #     outputs = model_ft(im)
# #     _, preds = torch.max(outputs, 1)
# #     total +=1
# #     if int(preds.data[0]) == 1:
# #         correct +=1
# #
# # print(correct/float(total))
# #
# # correct = 0
# # total = 0
# #     # outfile.write(image + "," + str(int(preds.data[0]))+ "\n")
# # # Use trained model to generate predictions
# # with open("test_baseline.txt", "w+") as outfile:
# #     for image in os.listdir("./../data_alternate_baseline/val/normal/"):
# #         im = Image.open("./../data_alternate_baseline/val/normal/"+ image)
# #         im = im.convert('RGB')
# #         im = data_transforms['train'](im).float()
# #         im = Variable(im, requires_grad = True)
# #         im = im.unsqueeze(0)
# #         #im = im.cuda()
# #         outputs = model_ft(im)
# #         _, preds = torch.max(outputs, 1)
# #         if int(preds.data[0]) == 0:
# #             correct +=1
# #         total += 1
# #         #outfile.write(image + "," + str(int(preds.data[0]))+ "\n")
# #
# #     for image in os.listdir("./../data_alternate_baseline/val/pneumonia/"):
# #         im = Image.open("./../data_alternate_baseline/val/pneumonia/"+ image)
# #         im = im.convert('RGB')
# #         im = data_transforms['train'](im).float()
# #         im = Variable(im, requires_grad = True)
# #         im = im.unsqueeze(0)
# #         #im = im.cuda()
# #         outputs = model_ft(im)
# #         _, preds = torch.max(outputs, 1)
# #         if int(preds.data[0]) == 1:
# #             correct +=1
# #         total += 1
# #         #outfile.write(image + "," + str(int(preds.data[0]))+ "\n")
# #
# #     for image in os.listdir("./../data_alternate_baseline/train/normal/"):
# #         im = Image.open("./../data_alternate_baseline/train/normal/"+ image)
# #         im = im.convert('RGB')
# #         im = data_transforms['train'](im).float()
# #         im = Variable(im, requires_grad = True)
# #         im = im.unsqueeze(0)
# #         #im = im.cuda()
# #         outputs = model_ft(im)
# #         _, preds = torch.max(outputs, 1)
# #         if int(preds.data[0]) == 0:
# #             correct +=1
# #         total += 1
# #         #outfile.write(image + "," + str(int(preds.data[0]))+ "\n")
# #
# #     for image in os.listdir("./../data_alternate_baseline/train/pneumonia/"):
# #         im = Image.open("./../data_alternate_baseline/train/pneumonia/"+ image)
# #         im = im.convert('RGB')
# #         im = data_transforms['train'](im).float()
# #         im = Variable(im, requires_grad = True)
# #         im = im.unsqueeze(0)
# #         #im = im.cuda()
# #         outputs = model_ft(im)
# #         _, preds = torch.max(outputs, 1)
# #         if int(preds.data[0]) == 1:
# #             correct +=1
# #         total += 1
# #         #outfile.write(image + "," + str(int(preds.data[0]))+ "\n")
# # print(correct/float(total))
