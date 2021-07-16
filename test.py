import torch
import config
from config import args_setting
from torchvision import transforms
import torchvision
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import cv2
from model import UNet,get_model
import matplotlib.pylab as plt
from dataset import RoadDataset, GetData,RoadDatasetList
import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from loss import SoftmaxFocalLoss,IouLoss,DiceLoss


index = 0


def save(data1, data2, k):
    global index
    size = data1.shape[0]
    for i in range(size):
        pred, data = data1[i], data2[i]
        img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
        img = Image.fromarray(img.astype(np.uint8))
        data = torch.squeeze(data).cpu().numpy()
        data = np.transpose(data, [1, 2, 0]) * 255  
        data = Image.fromarray(data.astype(np.uint8))
        # print(img.shape,data.shape)
        rows = img.size[0]  # 128
        cols = img.size[1]  # 256
        for i in range(0, rows):
            for j in range(0, cols):
                img2 = (img.getpixel((i, j))) 
                if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200): 
                    data.putpixel((i, j), (234, 53, 57, 255))  

        data = data.convert("RGB")
        index += 1
        data.save(config.save_path + "%s_data.jpg" % index)  # red line on the original image
        img.save(config.save_path + "%s_pred.jpg" % index)  # prediction result



def output_result(model, test_loader, device):
    model.eval()
    k = -1
    with torch.no_grad():
        for sample_batched in test_loader:
            k += 1
            print(k)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output = model(data)
            pred = torch.argmax(output, 1)  
            save(pred, data[:,-1,::], k)


def evaluate_model(model, test_loader, device, criterion,loss_1):
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
            loss1 =  criterion(output, target)  
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
            # accuracy
           # test_loss += criterion(output, target).item()  # sum up batch loss
            print(F1_measure)
            correct += pred.eq(target.view_as(pred)).sum().item()  

            # precision,recall,f1
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
    


if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    model = get_model(args)
    model = torch.nn.DataParallel(model,device_ids=[0,1])
    test_loader = torch.utils.data.DataLoader(
            dataset=RoadDatasetList(file_path=config.test_path, transforms=transform,x = 5,y = 5), 
            batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    weight = torch.Tensor(config.class_weight)
    loss1 = SoftmaxFocalLoss(gamma=2).to(device)
    loss2 = DiceLoss(n_classes=2,weights=weight,sotfmax=True).to(device)
   
    print(len(test_loader.dataset))
   
    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)
    # model.load_state_dict(pretrained_dict)
      
    # output the result pictures
  #  output_result(model, test_loader, device)
    # calculate the values of accuracy, precision, recall, f1_measure
    evaluate_model(model, test_loader, device, loss1,loss2)
