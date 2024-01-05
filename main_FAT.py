from dataset_load import data_load # Loading Data set
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from Augmentation import TrainAugment, ValidationAugment
from Fat_models import build_model
import numpy as np
import time
import datetime
# Congigurations

img_size = 79 # not used yet 23-11-02
batch_size = 8  # You can adjust this as needed
lr= 0.0001
epochs = 100


# Loading Dataset
Class_1_path_train = 'D:\\pythonProject\\Alzhmier\\Contrastive Learning_v3\\Datafile\\original\\Train_EMCI_ori.npy'
Class_2_path_train = 'D:\\pythonProject\\Alzhmier\\Contrastive Learning_v3\\Datafile\\original\\Train_LMCI_ori.npy'
Class_1_path_val = 'D:\\pythonProject\\Alzhmier\\Contrastive Learning_v3\\Datafile\\original\\Val_EMCI_ori.npy'
Class_2_path_val = 'D:\\pythonProject\\Alzhmier\\Contrastive Learning_v3\\Datafile\\original\\Val_LMCI_ori.npy'

X_train, y_train, X_val,y_val  = data_load(Class_1_path_train, Class_2_path_train , Class_1_path_val, Class_2_path_val ) 

# Converting into tensor 

# X_train = torch.DoubleTensor(X_train) # For float64 dtype
# y_train = torch.DoubleTensor(y_train) 


# pad_width = ((0, 0), (17, 16), (25, 24))
# pad_width = ((0,0), (57,58),(66,67))
# # # pad the array with zeros
# X_train = np.pad(X_train, pad_width, mode='constant')
# X_val   = np.pad(X_val, pad_width, mode='constant')
#

X_train = torch.Tensor(X_train)
y_train = torch.Tensor(y_train)

X_val   = torch.Tensor(X_val)
y_val   = torch.Tensor(y_val)


train_transform = TrainAugment(img_size)
val_Transform = ValidationAugment(img_size)

X_train = train_transform(X_train)
X_val = val_Transform(X_val)

X_train = X_train.unsqueeze(1)
X_val = X_val.unsqueeze(1)


train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


# Loading data into batches using dataloader

# Set a random seed for reproducibility
torch.manual_seed(24)  # You can use any integer as the seed

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# plotting batch
for batch, label in train_loader:
    # print("shape batch:", (torch.stack(batch)).shape)
    print("Batch shape:", (batch.shape))
    print("label shape:", (label.shape))
    # Image1 = batch[1]
    Label1 = label
    # print(Image1)
    print(Label1)
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


#### For FAT_Bo 4.5M parameters

in_chans = 1
num_classes = 2
embed_dims= [48, 96, 192, 384]
depths= [2, 2, 6, 2]
kernel_sizes= [3, 5, 7, 9]
num_heads= [3, 6, 12, 24]
window_sizes= [8, 4, 2, 1]
mlp_kernel_sizes= [5, 5, 5, 5]
mlp_ratios= [4, 4, 4, 4]
drop_path_rate= 0.1
use_checkpoint= False

# Model
model = build_model(in_chans, num_classes, embed_dims, depths,kernel_sizes,
                    num_heads,window_sizes, mlp_kernel_sizes, mlp_ratios,
                    drop_path_rate, use_checkpoint).to(device)

print("model: ", model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr)

train_loss = []
val_loss = []



start_time = time.time()


for epoch in range(epochs):
    print("epoch {}/{}".format(epoch + 1, epochs))
    running_loss = 0.0
    running_score = 0.0
    model.train()
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        y_pred = model.forward(image)

        loss = criterion(y_pred, label.long())
        val, index_ = torch.max(y_pred, axis=1)
        running_score += torch.sum(index_ == label.data).item()
        running_loss += loss.item()
        loss.backward()  # calculate derivatives
        optimizer.step()  # update parameters

    epoch_score = running_score / len(train_loader.dataset)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_score = 0.0
        for image, label in val_loader:
            image = image.to(device)
            label = label.to(device)
            y_pred_val = model.forward(image.float())
            loss_val = criterion(y_pred_val, label.long())
            optimizer.zero_grad()

            running_loss += loss_val.item()

            _, index_val = torch.max(y_pred_val, axis=1)
            running_score += torch.sum(index_val == label.data).item()

        epoch_score = running_score / len(val_loader.dataset)
        epoch_loss = running_loss / len(val_loader.dataset)
        val_loss.append(epoch_loss)
        print("Validation loss: {}, accuracy: {}".format(epoch_loss, epoch_score))

end_time = time.time()

save_path = 'scratch_model_FAT_b1_lr0_001_model.pth'
torch.save(model.state_dict(), save_path)

total_time = end_time - start_time


total_time_formatted = str(datetime.timedelta(seconds=total_time))

print(f"Total training time: {total_time_formatted}")



print("TRain_loss : ", train_loss)
print("Val loss :", val_loss)
plt.plot(train_loss,label='train loss')
plt.plot(val_loss,label='test loss')
plt.legend()
plt.show()






