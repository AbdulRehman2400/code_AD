
import torch

from torch.utils.data import DataLoader

from torchvision.utils import make_grid

import matplotlib.pyplot as plt

import torch.nn as nn

from torch import optim

from Fat_models import build_model

import numpy as np

import torchvision

import time

import datetime

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision.transforms as transforms

from torchsummary import summary



# Congigurations

batch_size = 8 # You can adjust this as needed

lr = 0.001

epochs = 150


# # Normalize data with mean=0.5, std=1.0

cifar_transform_train = transforms.Compose([

transforms.ToTensor(), 

transforms.Resize((224,224)),

transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),

transforms.ColorJitter(brightness=0.2, contrast=0.2),

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])


cifar_transform_validation = transforms.Compose([

transforms.ToTensor(), 

transforms.Resize((224,224)),

transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])



download_root = './CIFAT10_DATASET'

train_dataset = torchvision.datasets.CIFAR10(root=download_root, train=True, download=True, transform= cifar_transform_train)

valid_dataset = torchvision.datasets.CIFAR10(root=download_root, train=False, download=True, transform= cifar_transform_validation)


batch_size = 32


train_loader = DataLoader(dataset=train_dataset, 

batch_size=batch_size,

shuffle=True)


valid_loader = DataLoader(dataset=valid_dataset, 

batch_size=batch_size,

shuffle=False)




# plotting batch

for batch, label in train_loader:
    
    print("Batch shape:", (batch.shape))
    
    print("label shape:", (label.shape))
    
    print("label shape:", (label))
    
    concatenated_images = make_grid(batch, nrow=int(batch_size ** 0.5), normalize=True)
    
    # Convert the tensor to a NumPy array and transpose the dimensions
    
    concatenated_images = concatenated_images.permute(1, 2, 0).numpy()
    
    plt.figure()
    
    # Display the concatenated image
    
    plt.imshow(concatenated_images)
    
    plt.axis('off')
    
    plt.show()
    
    break


for batch, label in valid_loader:


    print("Batch shape:", (batch.shape))
    
    print("label shape:", (label.shape))
    
    print("label shape:", (label))
    
    concatenated_images = make_grid(batch, nrow=int(batch_size ** 0.5), normalize=True)
    
    # Convert the tensor to a NumPy array and transpose the dimensions
    
    concatenated_images = concatenated_images.permute(1, 2, 0).numpy()
    
    plt.figure()
    
    # Display the concatenated image
    
    plt.imshow(concatenated_images)
    
    plt.axis('off')
    
    plt.show()
    
    break



# Loading Model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


#### For FAT_B1

in_chans = 3 

num_classes = 10 # when sigmoid is used it will be 1

embed_dims= [48, 96, 192, 384]

depths= [2, 2, 6, 2]

kernel_sizes= [3, 5, 7, 9]

num_heads= [3, 6, 12, 24]

window_sizes= [8, 4, 2, 1]

mlp_kernel_sizes= [5, 5, 5, 5]

mlp_ratios= [4, 4, 4, 4]

drop_path_rate= 0.5

use_checkpoint= False


# Model

model = build_model(in_chans, num_classes, embed_dims, depths, kernel_sizes,

num_heads, window_sizes, mlp_kernel_sizes, mlp_ratios,

drop_path_rate, use_checkpoint).to(device)


# print("model: ", summary(model, (1, 224, 224)))


criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-6)

scheduler = CosineAnnealingWarmRestarts(optimizer,

T_0=8, # Number of iterations for the first restart

T_mult=1, # A factor increases TiTi after a restart

eta_min=1e-4) # Minimum learning rate


iters = len(train_loader)


train_loss = []

val_loss = []


start_time = time.time()


learning_rates = []


for epoch in range(epochs):

    print("epoch {}/{}".format(epoch + 1, epochs))
    
    running_loss = 0.0
    
    running_score = 0.0
    
    model.train()
    
    for i, sample in enumerate(train_loader):
    
        image, label = sample
        
        image = image.to(device)
        
        label = label.to(device)
        
        optimizer.zero_grad()
        
        
        y_pred = model.forward(image.float())
        
        # print(y_pred.cpu().numpy())
        
        loss = criterion(y_pred, label.long())
        
        val, index_ = torch.max(y_pred, axis=1)
        
        running_score += torch.sum(index_ == label.data).item()
        
        running_loss += loss.item()
        
        loss.backward() # calculate derivatives
        
        optimizer.step() # update parameters
        
        scheduler.step(epoch + i / iters)
        
        learning_rates.append(optimizer.param_groups[0]["lr"])
    
    
    
    epoch_score = running_score / len(train_loader.dataset)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    
    train_loss.append(epoch_loss)
    
    print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
    
    
    model.eval()
    
    with torch.no_grad():
    
        running_loss = 0.0
        
        running_score = 0.0
        
        for image, label in valid_loader:
        
            image = image.to(device)
            
            label = label.to(device)
            
            y_pred_val = model.forward(image.float())
            
            loss_val = criterion(y_pred_val, label.long())
            
            optimizer.zero_grad()
            
            
            running_loss += loss_val.item()
            
            
            _, index_val = torch.max(y_pred_val, axis=1)
            
            running_score += torch.sum(index_val == label.data).item()
        
        
        epoch_score = running_score / len(valid_loader.dataset)
        
        epoch_loss = running_loss / len(valid_loader.dataset)
        
        val_loss.append(epoch_loss)
        
        print("Validation loss: {}, accuracy: {}".format(epoch_loss, epoch_score))


end_time = time.time()


save_path = 'scratch_model_FAT_b1_schedulelr_model_flip0_rotate45_cifar10.pth'

torch.save(model.state_dict(), save_path)


total_time = end_time - start_time


total_time_formatted = str(datetime.timedelta(seconds=total_time))


print(f"Total training time: {total_time_formatted}")


print("TRain_loss : ", train_loss)

print("Val loss :", val_loss)

plt.plot(train_loss, label='train loss')

plt.plot(val_loss, label='test loss')

plt.legend()

plt.show()


plt.plot(learning_rates)

plt.legend()

plt.show()









# '''