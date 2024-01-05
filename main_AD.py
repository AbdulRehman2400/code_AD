from dataset_load import data_load  # Loading Data set
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from Fat_models import build_model
import numpy as np
import time
import datetime
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torchsummary import summary
import monai.transforms as T

# Configurations
batch_size = 8  # You can adjust this as needed
lr = 0.001
epochs = 150

Class_1_path_train = 'smci_samples.npy'
Class_2_path_train = 'ad_samples.npy'
Class_1_path_val = 'smci_samples.npy'
Class_2_path_val = 'ad_samples.npy'

X_train, y_train, _, _ = data_load(Class_1_path_train, Class_2_path_train, Class_1_path_val, Class_2_path_val)

print("Numpy loaded train data shape: ", X_train.shape)
print("Numpy loaded train label shape: ", y_train.shape)


# data
class train_ADDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, X, labels):
        'Initialization'
        self.X = X
        self.labels = labels  #

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        labels = self.labels[index]

        return X, labels

    transform = T.Compose([
        T.ToTensor(),
        T.EnsureChannelFirst(channel_dim='no_channel'),
        T.Resize((224,224)),
        T.RandRotate(range_x=45, prob=0.7),
        T.RandFlip(prob=0.4, spatial_axis=0),
        # T.RandGaussianNoise(prob=0.5, mean=0.1, std=0.2),
    ])


class val_ADDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, X, labels):
        'Initialization'
        self.X = X
        self.labels = labels  #

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(image)
        labels = self.labels[index]

        return X, labels

    transform = T.Compose([
        T.ToTensor(),
        T.EnsureChannelFirst(channel_dim='no_channel'),
        T.Resize((224, 224)),

    ])


train_dataset = train_ADDataset(X=X_train, labels=y_train)
# val_dataset = val_ADDataset(X=X_val, labels=y_val)


batch_size = 8
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
'''
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

for batch, label in val_loader:
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
'''
# Loading Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#### For FAT_B1

in_chans = 1
num_classes = 2  # when sigmoid is used it will be 1
embed_dims = [48, 96, 192, 384]
depths = [2, 2, 6, 2]
kernel_sizes = [3, 5, 7, 9]
num_heads = [3, 6, 12, 24]
window_sizes = [8, 4, 2, 1]
mlp_kernel_sizes = [5, 5, 5, 5]
mlp_ratios = [4, 4, 4, 4]
drop_path_rate = 0.1
use_checkpoint = False


# Model
model = build_model(in_chans, num_classes, embed_dims, depths, kernel_sizes,
                    num_heads, window_sizes, mlp_kernel_sizes, mlp_ratios,
                    drop_path_rate, use_checkpoint).to(device)

print("model: ", summary(model, (1, 224, 244)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-6)


# scheduler
scheduler_name = "exponential"

# Parameters for each scheduler
lr_decay_rate = 0.95  # For Exponential Decay
T_0 = 8  # Initial number of iterations for SGDR
T_mult = 1  # Multiplicative factor for SGDR
step_size = 30  # For StepLR
gamma = 0.1  # For StepLR and Exponential Decay

# Initialize the chosen scheduler
if scheduler_name == "exponential":
    scheduler = ExponentialLR(optimizer, gamma=lr_decay_rate)
elif scheduler_name == "cosine_annealing":
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-4)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

iters = len(train_loader)

train_loss = []
val_loss = []

start_time = time.time()

learning_rates = []
best_train_accuracy = 0.0

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
        loss = criterion(y_pred, label.long())
        val, index_ = torch.max(y_pred, axis=1)
        running_score += torch.sum(index_ == label.data).item()
        running_loss += loss.item()
        loss.backward()  # calculate derivatives

        optimizer.step()  # update parameters

        if scheduler_name =="CosineAnnealingWarmRestarts":
            scheduler.step(epoch + i / iters)
        
        
        if scheduler_name != "CosineAnnealingWarmRestarts":  # this updates after every epoch
            scheduler.step()

        learning_rates.append(optimizer.param_groups[0]["lr"])


    epoch_score = running_score / len(train_loader.dataset)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_loss.append(epoch_loss)
    print("Training loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
    # Checkpointing based on training accuracy
    if epoch_score > best_train_accuracy and epoch_score > 0.70:
        print(f"Saving checkpoint at epoch {epoch + 1} with training accuracy: {epoch_score}")
        best_train_accuracy = epoch_score
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracy': epoch_score,
        }, filename=f"checkpoint_epoch_{epoch + 1}_{epoch_score:.2f}.pth.tar")

    # model.eval()
    # with torch.no_grad():

    #     running_loss = 0.0
    #     running_score = 0.0
    #     for image, label in val_loader:
    #         image = image.to(device)
    #         label = label.to(device)
    #         y_pred_val = model.forward(image.float())
    #         loss_val = criterion(y_pred_val, label.long())
    #         optimizer.zero_grad()

    #         running_loss += loss_val.item()

    #         _, index_val = torch.max(y_pred_val, axis=1)
    #         running_score += torch.sum(index_val == label.data).item()

    #     epoch_score = running_score / len(val_loader.dataset)
    #     epoch_loss = running_loss / len(val_loader.dataset)
    #     val_loss.append(epoch_loss)
    #     print("Validation loss: {}, accuracy: {}".format(epoch_loss, epoch_score))
    #     # this schedular update on the basis of validation loss
   
end_time = time.time()

save_path = 'model_FAT_b1_exp_schedulelr_model.pth'
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




