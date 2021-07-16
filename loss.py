import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss()

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss
class IouLoss(nn.Module):
    def __init__(self):
        super(IouLoss, self).__init__()
    def forward(self,pre,target):
        n,h,w = target.shape
        return (1.0-pre.eq(target).sum().type(torch.float)/(2.0*n*h*w - pre.eq(target).sum().type(torch.float)))

class DiceLoss(nn.Module):
    def __init__(self, n_classes,weights,sotfmax):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.weight = weights
        self.softmax = sotfmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target):
        if self.softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if self.weight is None:
            self.weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
       # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
          #  class_wise_dice.append(1.0 - dice.item())
            loss += dice * self.weight[i]
        return loss / self.n_classes
