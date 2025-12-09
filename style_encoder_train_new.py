import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
from os.path import isfile
from skimage import io
from torchvision.utils import save_image
from skimage.transform import resize
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from utils.unhd_dataset import UNHDDataset  # Import the UNHD dataset
from utils.auxilary_functions import affine_transformation, image_resize_PIL, centered_PIL
from feature_extractor import ImageEncoder
import timm
import cv2
import time
import json
import random

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class Mixed_Encoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='mobilenetv2_100', num_classes=302, pretrained=True, trainable=True
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool=""
        )
        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Create the classifier
        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            # Fallback, can be adjusted based on the specific model
            num_features = 2048

        self.classifier = nn.Linear(num_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        # Extract features
        features = self.model(x)

        # Pool the features to make them of fixed size
        pooled_features = self.global_pool(features).flatten(1)

        # Classify
        logits = self.classifier(pooled_features)
        # print('logits', logits.shape)
        # print('pooled_features', pooled_features.shape)
        return logits, pooled_features  

#================ Performance and Loss Function ========================
def performance(pred, label):
    
    loss = nn.CrossEntropyLoss()
   
    loss = loss(pred, label)
    return loss 

#===================== Training ==========================================

def train_class_epoch(model, training_data, optimizer, args):
    '''Epoch operation in training phase'''
    
    model.train()
    total_loss = 0
    n_corrects = 0 
    total = 0
    pbar = tqdm(training_data)
    for i, data in enumerate(pbar):
    
        image = data[0].to(args.device)
        labels = torch.tensor([int(l) for l in data[3]]).to(args.device)
        
        optimizer.zero_grad()

        output, _ = model(image)  # Use logits
        
        loss = performance(output, labels)
        _, preds = torch.max(output.data, 1)
 
        loss.backward()
        optimizer.step()
        total_loss += loss.item() 
        total += labels.size(0)
        n_corrects += (preds == labels).sum().item()
        pbar.set_postfix(Loss=loss.item())
        
    loss = total_loss/total
    accuracy = n_corrects/total
    
    return loss, accuracy

def eval_class_epoch(model, validation_data, args):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    total = 0
    n_corrects = 0
    prediction_list = []
    results = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(validation_data)):

            image = data[0].to(args.device)   
            image_paths = data[8]
            labels = torch.tensor([int(l) for l in data[3]]).to(args.device)

            output, _ = model(image)  # Use logits
            
            loss = performance(output, labels)  #performance
            _, preds = torch.max(output.data, 1)
            
            total_loss += loss.item()
            n_corrects += (preds == labels.data).sum().item()
            total += labels.size(0)
            #prediction_list.append(preds)
            #write into a file the img_path and the prediction
            # with open('predictions.txt', 'a') as f:
            #     for i, p in enumerate(preds):
            #         f.write(f'{image_paths[i]},{p}\n')
            
    loss = total_loss/total
    accuracy = n_corrects/total

    return loss, accuracy




########################################################################              
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
    
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img.to(device)

        _, anchor_features = model(anchor)  # Use features
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        loss = criterion(anchor_features, positive_features, negative_features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += img.size(0)
    
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_triplet(val_loader, model, criterion, optimizer, device, args):
    
    running_loss = 0
    total = 0
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]

        positive = data[4].to(device)
        negative = data[5].to(device)
       
        anchor = img.to(device)
    
        _, anchor_features = model(anchor)  # Use features
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        loss = criterion(anchor_features, positive_features, negative_features)
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        pbar.set_postfix(triplet_loss=loss.item())
        total += img.size(0)
    
    print('total', total)
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    return running_loss/total #np.mean(running_loss)/total



############################ MIXED TRAINING ############################################              
def train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        
        img = data[0]
        wids = torch.tensor([int(w) for w in data[3]]).to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img.to(device)
        # Get logits and features from the model
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wids).sum().item()
    
        classification_loss = performance(anchor_logits, wids)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        #pbar.set_postfix(triplet_loss=loss.item())
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg, classification_loss=loss_meter_class.avg, triplet_loss=loss_meter_triplet.avg)
        total += img.size(0)
    
    accuracy = n_corrects/total
    print('total', total)
    print("Training Loss: {:.4f}".format(running_loss/len(train_loader)))
    print("Training Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total

def val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    for i, data in enumerate(pbar):
        
        img = data[0].to(device)
        wids = torch.tensor([int(w) for w in data[3]]).to(device)
        positive = data[4].to(device)
        negative = data[5].to(device)
        
        anchor = img
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wids).sum().item()
    
        classification_loss = performance(anchor_logits, wids)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        #running_loss.append(loss.cpu().detach().numpy())
        running_loss += loss.item()
        count = img.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(mixed_loss=loss_meter.avg)
        total += wids.size(0)
    
    print('total', total)
    accuracy = n_corrects/total
    print("Validation Loss: {:.4f}".format(running_loss/len(val_loader)))
    print("Validation Accuracy: {:.4f}".format(accuracy*100))
    return running_loss/total #np.mean(running_loss)/total






#TRAINING CALLS

def train_mixed(model, train_loader, val_loader, criterion_triplet, criterion_classification, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        
def train_classification(model, training_data, validation_data, optimizer, scheduler, device, args): #scheduler # after optimizer
    ''' Start training '''

    valid_accus = []
    num_of_no_improvement = 0
    best_acc = 0
    
    for epoch_i in range(args.epochs):
        print('[Epoch', epoch_i, ']')

        start = time.time()
        #wandb.log({'lr': scheduler.get_last_lr()})
        #print('Epoch:', epoch_i,'LR:', scheduler.get_last_lr())

        train_loss, train_acc = train_class_epoch(model, training_data, optimizer, args)
        print('Training: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  loss=train_loss, accu=100*train_acc,
                  elapse=(time.time()-start)/60))
        
        start = time.time()
        model_state_dict = model.state_dict()
        checkpoint = {'model': model_state_dict, 'settings': args, 'epoch': epoch_i}

        if validation_data is not None:
            val_loss, val_acc = eval_class_epoch(model, validation_data, args)
            print('Validation: {loss: 8.5f} , accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                        loss=val_loss, accu=100*val_acc,
                    elapse=(time.time()-start)/60))
            
            if val_acc > best_acc:
                
                print('- [Info] The checkpoint file has been updated.')
                best_acc = val_acc
                torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")
                num_of_no_improvement = 0
            else:
                num_of_no_improvement +=1
            
        
            if num_of_no_improvement >= 10:
                        
                print("Early stopping criteria met, stopping...")
                break
        else:
            torch.save(model.state_dict(), f"{args.save_path}/{args.dataset}_classification_{args.model}.pth")

        scheduler.step()
        #wandb.log({'epoch': epoch_i, 'train loss': train_loss, 'val loss': val_loss})
        #wandb.log({'epoch': epoch_i, 'train acc': 100*train_acc, 'val acc': 100*val_acc})
        

def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        model.train()
        train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
        print("Epoch: {}/{}".format(epoch_i+1, args.epochs))
        
        model.eval()
        with torch.no_grad():
            val_loss = val_epoch_triplet(val_loader, model, criterion, optimizer, device, args)
        
        if val_loss < best_loss:
            best_loss =val_loss
            torch.save(model.state_dict(), f'{args.save_path}/triplet_{args.dataset}_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)
        
        

def main():
    '''Main function'''
    parser = argparse.ArgumentParser(description='Train Style Encoder')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', help='type of cnn to use (resnet, densenet, etc.)')
    parser.add_argument('--dataset', type=str, default='unhd', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=320, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, required=False, help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, help='use of feature extractor or not')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--save_path', type=str, default='./style_models', help='path to save models')
    parser.add_argument('--mode', type=str, default='mixed', help='mixed for DiffusionPen, triplet for DiffusionPen-triplet, or classification for DiffusionPen-triplet')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    #========= Data augmentation and normalization for training =====#
    if os.path.exists(args.save_path) == False:
        os.makedirs(args.save_path)

    if args.dataset == 'unhd':
        myDataset = UNHDDataset
        dataset_folder = './unhd_data/UNHD-Complete-Data'
        aug_transforms = [lambda x: affine_transformation(x, s=.1)]
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = myDataset(dataset_folder, 'train', 'line', fixed_size=(64, 256), transforms=train_transform)
        validation_size = int(0.2 * len(train_data))
        train_size = len(train_data) - validation_size
        train_data, val_data = random_split(train_data, [train_size, validation_size], generator=torch.Generator().manual_seed(42))
        print('len train data', len(train_data))
        print('len val data', len(val_data))
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
        with open('./writers_dict.json', 'r') as f:
            wr_dict = json.load(f)
        style_classes = len(wr_dict)

    # Model setup
    if args.model == 'mobilenetv2_100':
        print('Using mobilenetv2_100')
        model = Mixed_Encoder(model_name='mobilenetv2_100', num_classes=style_classes, pretrained=True, trainable=True)
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained == True:
        PATH = args.style_path  # Define PATH if needed
        state_dict = torch.load(PATH, map_location=device)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Pretrained model loaded')

    model = model.to(device)
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)

    # THIS IS THE CONDITION FOR DIFFUSIONPEN
    if args.mode == 'mixed':
        criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        print('Using both classification and metric learning training')
        train_mixed(model, train_loader, val_loader, criterion_triplet, None, optimizer_ft, scheduler, device, args)
        print('finished training')
    if args.mode == 'triplet':
        train_triplet(model, train_loader, val_loader, criterion, optimizer_ft, lr_scheduler, device, args)
        print('finished training')
    elif args.mode == 'classification':
        train_classification(model, train_loader, val_loader, optimizer_ft, scheduler, device, args)
        print('finished training')

if __name__ == '__main__':
    main()