# Data import, handling
import pandas as pd

# Data plotting
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Image pre-processing
import matplotlib.image as img

# PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models


# Scikit modules
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from skimage import io, transform

# numpy
import numpy as np

# Standard Library Modules
import argparse
import os

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Credit goes to ptrblack for this class: https://discuss.pytorch.org/t/why-do-we-need-subsets-at-all/49391/7
# This class allows for easier application of different transforms to datasets. Used in the load_split_train_valid_test method
class MyLazyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y
    
    def __len__(self):
        return len(self.dataset)

# Define helper methods
'''Displays an image alongside its bounding box and part locations. Defaults to not showing hidden parts.
Note that the first piece of data in both box_data and parts_data will be the image id, which is irrelevant for
plotting single images at a time.'''
def display_image(image, box_data, parts_data, show_hidden_parts=False):
    # Create base figure and axis, label plot and axes
    fig, ax = plt.subplots()
    ax.set_title("Bird Image with Bounding Box and Labeled Parts")


    ax.imshow(image)

    # Then create red rectangle representing bounding box
    rect = patches.Rectangle((box_data[1], box_data[2]), box_data[3], box_data[4], ec='r', fc='None')
    ax.add_patch(rect)

    # Then plot each provided part (including hidden parts if specified). Part names must be retrieved from a
    # separate file since the parts_data itself only contains part IDs, not names
    part_names = pd.read_csv("../data/CUB_200_2011/parts/parts.txt", sep="-", index_col=False)

    for inx in range(len(parts_data)):
        part = parts_data.iloc[inx]     # Grab current part
        name = part_names[part_names["part_id"]==part[1]]["part_name"].to_string(index=False)   # Minus one to account for the starting index being 0 for part_names
        x = part[2]
        y = part[3]
        hidden = part[4]
        if(hidden == 1) or show_hidden_parts:    # If the part is visible (or hidden but we want to see it regardless), plot its location
            ax.scatter(x, y)
            ax.annotate(name, xy = (x, y), color="black")
    plt.show()


# Note: The default normalize mean and std values are those for the RGB of ImageNet
def load_split_train_test_valid(datadir, test_size=.15, valid_size=.15, normalize_mean=[0.485, 0.456, 0.406], normalize_std=[0.229, 0.224, 0.225]):
    try:
        if(test_size < 0 or valid_size < 0):
            raise ValueError

        percentage_sum = test_size+valid_size
        if(percentage_sum >= 1 or percentage_sum <= 0):
            raise ValueError
    except ValueError:
        print("test_size and valid_size must be non-negative, less than 1, and their sum must be less than 1.")
        exit(1)
    
    train_transforms = transforms.Compose([transforms.Resize(224),  # Transform for training
                                        transforms.ToTensor(),
                                        transforms.Normalize(normalize_mean, normalize_std)])

    test_valid_transforms = transforms.Compose([transforms.Resize(224), # Transform for test/validation
                                        transforms.ToTensor(),
                                        transforms.Normalize(normalize_mean, normalize_std)])


    all_data = datasets.ImageFolder(datadir)

    train_data = MyLazyDataset(all_data,transform=train_transforms)
    test_data = MyLazyDataset(all_data,transform=test_valid_transforms)
    valid_data = MyLazyDataset(all_data,transform=test_valid_transforms)

    total_count = len(all_data)
    #test_count = int(test_size*total_count)
    #valid_count = int(valid_size*total_count)
    train_size = 1 - test_size - valid_size

    indices = np.arange(0,total_count)
    first_split = int(np.floor(train_size*total_count))
    second_split = int(np.floor((train_size+(test_size))*total_count))
    np.random.shuffle(indices)  # Shuffle the indices
    train_inx = indices[:first_split]
    test_inx = indices[first_split:second_split]
    valid_inx = indices[second_split:]
    
    train_data = Subset(train_data, indices=train_inx)
    test_data = Subset(test_data, indices=test_inx)
    valid_data = Subset(valid_data, indices=valid_inx)

    # Define parameters for the dataloaders
    batch_size = 64
    num_workers = 4

    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, drop_last=True)


    """    train_data, test_data, valid_data = random_split(
                                            all_data, 
                                            (train_count, test_count, valid_count),
                                            generator=torch.Generator().manual_seed(572)) """

    return train_loader, test_loader, valid_loader


def main():
    """  n = int(input("Which bird image would you like to look at: "))
    boxing_data = pd.read_csv("data/CUB_200_2011/bounding_boxes.txt", sep=" ", index_col=False)

    boxing_data_n= boxing_data.iloc[n]
    parts_data = pd.read_csv("data/CUB_200_2011/parts/part_locs.txt", sep=" ", index_col=False)
    image_names = pd.read_csv("data/CUB_200_2011/images.txt", sep=" ", index_col=False)
    parts_data_n = parts_data.iloc[(15*n):(15*n + 15)] """
    
    """ 
    # Test plot with second bird image
    image_name = image_names[image_names["image_id"]==n+1]["image_name"].to_string(index=False)   # Get local file name
    display_image(io.imread(os.path.join("data\CUB_200_2011\images", image_name)), boxing_data_n, parts_data_n, False)
    """


    # Import and process data
    data_dir = '../data/CUB_200_2011/cropped_images'
    
    # Build training/test sets randomly, applying the appropriate transformations to each set of images (normalization, cropping, etc.)
    train_loader, test_loader, valid_loader = load_split_train_test_valid(data_dir, .2)
    print(train_loader.dataset)

    # Load the model (implementation of the VGG16 model). Though we could build a CNN from scratch specific to birds, it's
    # quicker and requires less bird images to fine tune a pre-existing network.
    model = models.vgg16_bn(pretrained=True)
    print(model)

    # Use GPU CUDA cores for training if available, else default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Given the data similarity between the ImageNet dataset that the VGG16 model was trained on and our bird dataset,
    # we want to preserve the model's original parameters when training; in other words, we need to avoid updating the model when backpropagating
    for parameter in model.parameters():    # Each PyTorch 'Parameter' has an attribute designed to allow granular gradient control
        parameter.requires_grad_(False)

    # VGG-16 has a 4096 dimension output layer, which isn't what we need. Instead, we can modify the fully-connected
    # layer to fine-tune the network for our purposes. In this case, we want 200 dimensional output (for each category of bird)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 512),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(512, 200))    # 200 output layers
    model.to(device)
    # Note: The pre-trained VGG16_bn model lacks a SoftMax layer (as do all pre-trained models from torchvision), and we didn't add one above. 
    # However, this isn't an issue as these models are trained using CrossEntropyLoss, which for this purpose are equivalent to the combination of 
    # LogSoftMax and NLLLoss (so no need to add it to the model itself)

    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    # Decrease learning rate if the global minimum training loss has not decreased in 5 epochs (Adam is already an adaptive learning rate optimizer, but further adaptability will help bring minor improvements)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)   

    # Training the model
    try:
        num_epochs = int(input("How many epochs would you like to spend training the model? "))
        if num_epochs <= 0:
            raise ValueError
    except TypeError:
        print("Must be an integer. Using default number of epochs instead (20)")
        num_epochs = 20
    except ValueError:
        print("Must be a positive integer. Using default number of epochs instead (20)")
        num_epochs = 20

    # Store training loss over time
    loss_stats = {
        'train': [],
        "val": []
    }

    for epoch in range(1, num_epochs):
        # Training:
        train_epoch_loss = 0  # Tracks loss over the epoch
        epoch_losses = []     # Used to store current epoch losses for learning rate scheduler calculation
        for batch_inx, (data, targets) in enumerate(train_loader):
            data = data.to(device)   # Utilize CUDA if available
            targets = targets.to(device)

            # Forward pass (calculates predicted output)
            output = model(data)

            # Calculate batch loss
            train_loss = criterion(output, targets)   

            # Backward pass (calculates gradient with respect to model parameters)
            train_loss.backward()         

            # Iterate over all parameters to update their values based on gradient. 
            # Recall that the parameters of all non-output-layer layers have been "frozen" earlier, so their values won't update
            optimizer.step()    

            # Explicitly set gradients to zero before next backwards pass (no need to keep the gradient accumulated)
            optimizer.zero_grad()

            # Combine current training loss to epoch losses
            epoch_losses.append(train_loss.item())
            train_epoch_loss += train_loss.item()

        # Calculations for the learning rate scheduler
        mean_loss = sum(epoch_losses)/len(epoch_losses)
        scheduler.step(mean_loss)   # See if the learning rate needs to adapt
        print(f"Training loss at epoch {epoch} is {mean_loss}")

        # Validation
        with torch.no_grad():
            val_epoch_loss = 0
            model.eval()

            for batch_inx, (data, targets) in enumerate(valid_loader):
                data = data.to(device)
                targets = targets.to(device)

                y_val_pred = model(data)
                val_loss = criterion(y_val_pred, targets)
                val_epoch_loss += val_loss.item()

        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(valid_loader))

        # Display current epoch, training loss, and validation loss
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(valid_loader):.5f}')
  
    # Visualize Loss and Accuracy
    train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    plt.figure(figsize=(15,8))
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable").set_title('Train-Val Loss/Epoch')

    # Save model and scheduler to file
    save_model_path = "../model/vgg16_bird.pth"
    torch.save(model.state_dict(), save_model_path)
    save_schedular_path = "../model/vgg16_bird_scheduler"
    torch.save(scheduler.state_dict(), save_schedular_path)


if __name__== "__main__" :
    # Set up useful constants/parameters prior to training the model
    np.random.seed(572)
    pd.options.display.max_colwidth = 100   # increase column size (to_string() for image_name is dependent on a large value to not truncate)
    main()
