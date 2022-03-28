#head
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

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
        cos_phi = cos_t - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = F.one_hot(label, num_classes=self.out_features)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = torch.where(one_hot==1, cos_phi, cos_t)
        output *= self.s
        # print(output)

        return output


############SphereFaceLoss################
class SphereMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, m=4, base=1000.0, gamma=0.0001, power=2, lambda_min=5.0, iter=0):
        super(SphereMarginProduct, self).__init__()
        assert m in [1, 2, 3, 4], 'margin should be 1, 2, 3 or 4'
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.margin_formula = [
            lambda x : x ** 0,
            lambda x : x ** 1,
            lambda x : 2 * x ** 2 - 1,
            lambda x : 4 * x ** 3 - 3 * x,
            lambda x : 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x : 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        self.iter += 1
        self.cur_lambda = max(self.lambda_min, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1,1)

        cos_m_theta = self.margin_formula[self.m](cos_theta)
        theta = torch.acos(cos_theta)
        k = ((self.m * theta) / math.pi).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        phi_theta_ = (self.cur_lambda * cos_theta + phi_theta) / (1 + self.cur_lambda)
        norm_of_feature = torch.norm(input, 2, 1)

        one_hot = torch.zeros_like(cos_theta)

        one_hot.scatter_(1, label.view(-1, 1), 1)

        output = one_hot * phi_theta_ + (1 - one_hot) * cos_theta
        output *= norm_of_feature.view(-1, 1)

        return output


###########MagFace##############
class MagMarginProduct(nn.Module):
    """ implement Magface https://arxiv.org/pdf/2103.06627.pdf"""

    def __init__(self, in_features, out_features, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20):
        super(MagMarginProduct, self).__init__()

        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m
        
        self.lambda_g = lambda_g
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.eps = 1e-7

    def compute_m(self, a):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_m

    def compute_g(self, a):
        return torch.mean( (1 / self.u_a**2) * a + 1 / a)

    def forward(self, input, label):
        
        cos_t = F.linear(F.normalize(input), F.normalize(self.weight, dim=1)).clamp(-1. + self.eps, 1. - self.eps)
        sin_t = torch.sqrt(1.0 - torch.pow(cos_t, 2))

        # compute additive margin
        a = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m = self.compute_m(a)
        cos_m, sin_m = torch.cos(m), torch.sin(m)

        g = self.compute_g(a)

        #threshold when phi > 180 
        threshold = torch.cos(math.pi - cos_m)
        mm = torch.sin(math.pi - m) * m

        # phi = theta + m(a) => cos(phi)
        cos_phi = cos_t * cos_m - sin_t * sin_m
        cos_phi = torch.where(cos_phi > threshold, cos_phi, cos_t - mm)

        # one-hot label
        one_hot = torch.zeros_like(cos_phi)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        # build the output logits
        output = one_hot * cos_phi + (1.0 - one_hot) * cos_t
        # feature re-scaling
        output *= self.s

        return output, self.lambda_g * g