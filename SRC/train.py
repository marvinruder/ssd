import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD600, MultiBoxLoss
from datasets import GTSDBDataset
from utils import *
import os

# Data parameters
data_folder = '..\\RESOURCES\\'  # folder with data files

# Model parameters
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None # '..\\RESOURCES\\checkpoint.pth.tar'  # path to model checkpoint, None if none
batch_size = 4  # batch size
epochs = 250  # number of epochs to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 0  # print training status every __ batches, 0: print only after every epoch
lr = 3e-4  # learning rate
decay_lr = {150: 1e-4, 200: 3e-5}  # decay learning rate after these many epochs
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        
        # delete the old log file
        if os.path.exists("..\\RESOURCES\\train.log"):
            os.remove("..\\RESOURCES\\train.log")
        
        start_epoch = 0
        model = SSD600(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = GTSDBDataset(data_folder,
                                 split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr.keys():
            adjust_learning_rate(optimizer, decay_lr[epoch])

        # One epoch's training
        loss = train(train_loader=train_loader,
                     model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     epoch=epoch)
        
        # Print status
        if print_freq == 0:
            print('Epoch: [{0}]\tAvg Loss {loss.avg:.4f}'.format(epoch, loss=loss))
            train_log = open('..\\RESOURCES\\train.log', 'a')
            train_log.write('Epoch: [{0}]\tAvg Loss {loss.avg:.4f}\n'.format(epoch, loss=loss))
            train_log.close()


        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()  # loss


    # Batches
    for i, (images, boxes, labels) in enumerate(train_loader):

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 600, 600)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored
    return losses

if __name__ == '__main__':
    main()
