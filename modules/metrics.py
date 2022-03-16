#head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

###########Air Face##############
class AirMarginProduct(nn.Module):
    """fine tune Arcface https://arxiv.org/pdf/1907.12256.pdf :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(pi-(theta + m))
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.4):
        super(AirMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) #init weight
        self.eps = 1e-7

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps)
        t = torch.arccos(cos_t)
        phi = math.pi - 2 * t - 2 * self.m
        not_phi = math.pi - 2 * t
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_t)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot == 1, phi, not_phi)
        output *= (self.s / math.pi)

        return output


###########ArcFace##############
class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight) #init weight
        self.eps = 1e-7
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m) #threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))
        cos_t = cos_t.clamp(-1. + self.eps, 1. - self.eps)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))
        cos_phi = cos_t * self.cos_m - sin_t * self.sin_m
        cos_phi = torch.where(cos_t > self.th, cos_phi, cos_t - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_phi)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot == 1, cos_phi, cos_t)
        output *= self.s

        return output


###########CosFace##############
class CosMarginProduct(nn.Module):
    """Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta) - m
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1))
        phi = cos_t - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = F.one_hot(label, num_classes=self.out_features)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot==1, phi, cos_t)
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

###########MagFace##############
class MagMarginProduct(nn.Module):
    pass
