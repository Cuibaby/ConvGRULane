import argparse

# dataset setting
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
train_path = "path/new_train_index.txt" 
val_path = "path/val_index.txt"     
test_path = "path/demo.txt"
save_path = "results/tusimple/"
pretrained_path= 'weights/9035.pth'

# weight
class_weight = [0.02, 2.10]

def args_setting():
    
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvGRU')
    parser.add_argument('--model',type=str, default='UNet-ConvGRU',help='( UNet-ConvGRU | UNet | ')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=0.008, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--resume', type=bool, default=True,
                        help='whether use last state')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args
