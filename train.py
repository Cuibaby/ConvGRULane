from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
import torchvision.transforms as transforms
import numpy as np
import config
import time
from config import args_setting
import torchvision
from matplotlib.pylab import plt
from model import UNet,get_model
from torch.optim import lr_scheduler 
from dataset import RoadDataset,RoadDatasetList
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from loss import SoftmaxFocalLoss,IouLoss,DiceLoss

def train(args, epoch, model, train_loader, device, optimizer, criterion,loss_1,weight):
   
    model.train() 
    beta1 = torch.tensor(0.5).to(device)
    for batch_idx, sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(
            device) 
        optimizer.zero_grad()

        output = model(data)
        
        out = torch.argmax(output, 1)
        
        loss1 = criterion(output,target)  
        loss2 = beta1 * loss_1(output, target)
        if isinstance(model, torch.nn.DataParallel):
            loss1 = loss1.sum()
            loss2 = loss2.sum()
         
        loss = loss1 + loss2 
     
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    local = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    print('Train Epoch: {} complete in {}'.format(epoch,local))

def evaluate_model(model, test_loader, device, criterion,loss_1,weight):
    model.eval()
    i = 0
    precision = 0.0
    recall = 0.0
    test_loss = 0.0
    correct = 0.0
    error = 0
    F1_measure = 0.0
    beta1 = torch.tensor(0.5).to(device)
    with torch.no_grad():
        for sample_batched in test_loader:
            i += 1
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output = model(data)

            pred = torch.argmax(output, 1)
            out = torch.argmax(output, 1)
            loss1 = criterion(output, target)  
            loss2 = beta1 * loss_1(output, target)
            if isinstance(model, torch.nn.DataParallel):
                loss1 = loss1.sum()
                loss2 = loss2.sum()
   
            test_loss += (loss1 + loss2).item()
          
            img = torch.squeeze(pred).cpu().numpy() * 255
            lab = torch.squeeze(target).cpu().numpy() * 255
            img = img.astype(np.uint8)
            lab = lab.astype(np.uint8)
            kernel = np.uint8(np.ones((3, 3)))
            print(test_loss)
            correct += pred.eq(target.view_as(pred)).sum().item()  
            label_precision = cv2.dilate(lab, kernel)  
            pred_recall = cv2.dilate(img, kernel)
            img = img.astype(np.int32)
            lab = lab.astype(np.int32)
            label_precision = label_precision.astype(np.int32)
            pred_recall = pred_recall.astype(np.int32)
            a = len(np.nonzero(img * label_precision)[1])  
            b = len(np.nonzero(img)[1]) 
            if b == 0:
                error = error + 1
                continue
            else:
                precision += float(a / b)
            c = len(np.nonzero(pred_recall * lab)[1])
            d = len(np.nonzero(lab)[1])
            if d == 0:
                error = error + 1
                continue
            else:
                recall += float(c / d) 

            F1_measure = (2 * precision * recall) / (precision + recall)
            
    test_loss /= i
    test_acc = 100. * int(correct) / (len(test_loader.dataset) * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)'.format(
        test_loss, int(correct), len(test_loader.dataset), test_acc))
    precision /= i
    recall /= i
    F1_measure /= i
    local = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    print('Val Time: {}'.format(local))
    print('Precision: {:.5f}, Recall: {:.5f}, F1_measure: {:.5f}\n'.format(precision, recall, F1_measure))
    return test_acc,F1_measure

if __name__ == '__main__':
    args = args_setting() 
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
   
    transform = transforms.Compose([
        transforms.ToTensor()
      #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    model = get_model(args)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    if args.model == 'UNet':
         train_loader = torch.utils.data.DataLoader(
            dataset=RoadDataset(file_path=config.train_path, transforms=transform),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
         val_loader = torch.utils.data.DataLoader(
        dataset=RoadDataset(file_path=config.val_path, transforms=transform),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset=RoadDatasetList(file_path=config.train_path, transforms=transform,x = 5,y = 5),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(
            dataset=RoadDatasetList(file_path=config.val_path, transforms=transform,x = 5,y = 5),
            batch_size=args.batch_size, shuffle=True, num_workers=1)
    print(len(train_loader.dataset))
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5)
    weight = torch.Tensor(config.class_weight).to(device)
    criterion = SoftmaxFocalLoss(gamma=2.4).to(device)
    loss_1 = DiceLoss(n_classes=2,weights=weight,sotfmax=True).to(device)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    print(model)
    F1 = 0.905
    print(args.epochs)
    if args.resume:
        pretrained_dict = torch.load(config.pretrained_path)
        model_dict = model.state_dict()
        pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
        model_dict.update(pretrained_dict_1)
        model.load_state_dict(model_dict)
       # model.load_state_dict(pretrained_dict)
      
    else:
        print("Resume from checkpoint...")
        checkpoint = torch.load('./checkpoint/unet.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_schedule'])

    for epoch in range(args.epochs + 1):

        train(args, epoch, model, train_loader, device, optimizer, criterion,loss_1,weight)
        
        acc,f1 = evaluate_model(model, val_loader, device, criterion,loss_1,weight)
        scheduler.step() 
        if F1 < f1:
            F1 = f1
          
            if args.resume:
                torch.save(model.state_dict(), 'weights/%s.pth' % F1) 
                pretrained_dict = torch.load('weights/%s.pth' % F1)
                model_dict = model.state_dict()
                pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
                model_dict.update(pretrained_dict_1)
                model.load_state_dict(model_dict)

            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_schedule': scheduler.state_dict(),
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('./weights/checkpoint')
                torch.save(checkpoint, 'checkpoint/%s.pth' % acc)

