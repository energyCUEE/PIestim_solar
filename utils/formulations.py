import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class qd_objective(nn.Module): # adjust version removing N and delta effect
    def __init__(self, delta_ = 0.1, gamma_ = 0.01, soften_ = 100, datanorm = 'quantile'
                 , smoothfunction = 'sigmoid', returnseparatedloss = False):
        super(qd_objective, self).__init__()
        self.soften = soften_
        self.delta_ = delta_
        self.gamma = gamma_
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
            y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
        
        k_hard_u = torch.max(torch.sign(y_pred_upper - y), torch.zeros_like(y).to(self.device))
        k_hard_l = torch.max(torch.sign(y - y_pred_lower), torch.zeros_like(y).to(self.device))
        k_hard = torch.multiply(k_hard_u, k_hard_l)
        c_hard = torch.sum(k_hard)
        PICP_hard = torch.mean(k_hard)
        
        n_PINAW = c_hard + 0.001
        PINAW_capt_hard = torch.sum((y_pred_upper - y_pred_lower)*k_hard)/(n_PINAW*y_range)

        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")
            
        PICP_soft = torch.mean(k_soft, axis = 0)
        
        negloglike_soft = torch.square(torch.max(torch.zeros(1).to(self.device), (1 - self.delta_) - PICP_soft))
        loss_qd = torch.mean(negloglike_soft + self.gamma * PINAW_capt_hard)
        
        if self.returnseparatedloss:
            return loss_qd, negloglike_soft, PINAW_capt_hard
        else:
            return loss_qd
        
class cwcquan_objective(nn.Module): 
    def __init__(self, delta_ = 0.1, gamma_ = 10, soften_ = 100, datanorm = 'quantile'
                 , smoothfunction = 'sigmoid', returnseparatedloss = False):
        super(cwcquan_objective, self).__init__()
        self.soften = soften_
        self.delta_ = delta_
        self.gamma = gamma_
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
#             y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
            y_range = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
        
        
        PINRW = torch.norm((y_pred_upper - y_pred_lower), p = 2, dim = 0)/(torch.sqrt(N+0.001)*y_range)

        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")
            
        PICP_soft = torch.mean(k_soft, axis = 0)
        
        coverage_loss = 1 + torch.exp(self.gamma*torch.max(torch.zeros(1).to(self.device)
                                                           , (1-self.delta_)- PICP_soft))
        
        cwcquan_loss = torch.mean(PINRW*coverage_loss)
        
        if self.returnseparatedloss:
            return cwcquan_loss, coverage_loss, PINRW
        else:
            return cwcquan_loss


class cwcshri_objective(nn.Module): 
    def __init__(self, delta_ = 0.1, gamma_ = 5, soften_ = 100, datanorm = 'quantile'
                 , smoothfunction = 'sigmoid', returnseparatedloss = False):
        super(cwcshri_objective, self).__init__()
        self.soften = soften_
        self.delta_ = delta_
        self.gamma = gamma_
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
#             y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
            y_range = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
        
        
        PINAW = torch.norm((y_pred_upper - y_pred_lower), p = 1, dim = 0)/((N+0.001)*y_range)

        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")
            
        PICP_soft = torch.mean(k_soft, axis = 0)
        
        coverage_loss = torch.exp(self.gamma*torch.max(torch.zeros(1).to(self.device)
                                                                    , (1-self.delta_)- PICP_soft))
        
        cwcshri_loss = torch.mean(PINAW + coverage_loss)
        
        if self.returnseparatedloss:
            return cwcshri_loss, coverage_loss, PINAW
        else:
            return cwcshri_loss
        
class cwcli_objective(nn.Module): 
    def __init__(self, delta_ = 0.1, gamma_ = 15, alpha_ = 0.1, beta_ = 6, soften_ = 100
                 , datanorm = 'quantile', smoothfunction = 'sigmoid', returnseparatedloss = False):
        super(cwcli_objective, self).__init__()
        self.soften = soften_
        
        self.delta_ = delta_
        self.gamma = gamma_
        self.alpha = alpha_
        self.beta = beta_
        
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
#             y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
            y_range = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
        
        PINAW = torch.norm((y_pred_upper - y_pred_lower), p = 1)/((N+0.001)*y_range)

        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")
            
        PICP_soft = torch.mean(k_soft)

        coverage_loss = (self.alpha + (self.beta/2)*PINAW)*torch.exp(self.gamma*torch.max(torch.zeros(1).to(self.device)
                                                                    , (1-self.delta_)- PICP_soft))
    
        
        cwcli_loss = (self.beta/2)*PINAW + coverage_loss
        
        if self.returnseparatedloss:
            return cwcli_loss, coverage_loss, PINAW
        else:
            return cwcli_loss
        
class dic_objective(nn.Module): # Sum of k largest and lowest formulation
    def __init__(self, gamma_ = 0.1, delta_ = 0.1
                 , soften_ = 100, smoothfunction = 'sigmoid', datanorm = 'quantile', returnseparatedloss = False):
        super(dic_objective, self).__init__()
        self.soften = soften_
        self.delta_ = delta_
        self.gamma_ = torch.tensor(gamma_)
        
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss        

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
#             y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
            y_range = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
            
        ## Width term
        PINAW = torch.norm((y_pred_upper - y_pred_lower), p = 1, dim = 0)/((N+0.001)*y_range)
        pun = torch.sum((y_pred_lower - y)[y < y_pred_lower]) + torch.sum((y - y_pred_upper)[y > y_pred_upper])
        
        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")

        PICP_soft = torch.mean(k_soft, axis = 0)
        
        ## Aggregate the loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        coverage_loss = (PICP_soft < (1-self.delta_))*pun
        width_loss = PINAW
        
        loss_soft = coverage_loss + self.gamma_ * width_loss
        
        loss_soft = torch.mean(loss_soft)
        
        if self.returnseparatedloss:
            return loss_soft, coverage_loss, width_loss
        else:
            return loss_soft
        
class sumk_objective(nn.Module): # Sum of k largest and lowest formulation
    def __init__(self, gamma_ = 0.7, percentlargest_ = 0.1, lambda_ = 0.7, delta_ = 0.1
                 , soften_ = 100, smoothfunction = 'sigmoid', datanorm = 'quantile', returnseparatedloss = False):
        super(sumk_objective, self).__init__()
        self.soften = soften_
        self.delta_ = delta_
        self.gamma_ = torch.tensor(gamma_)
        self.lambda_ = lambda_
        self.percentlargest_ = percentlargest_
        
        self.datanorm = datanorm
        self.smoothfunction = smoothfunction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.returnseparatedloss = returnseparatedloss        

    def forward(self, y, y_pred_lower, y_pred_upper):
        N = torch.tensor(y.shape[0])
        if self.datanorm == 'maxmin':
            y_range = y.max() - y.min()
        elif self.datanorm == 'quantile':
#             y_range = torch.quantile(y, 0.95) - torch.quantile(y, 0.05)
            y_range = np.quantile(y, 0.95) - np.quantile(y, 0.05)
        else:
            raise ValueError("Input must be maxmin or quantile")
            
        ## Width term
        widths = y_pred_upper - y_pred_lower
        num_k_largest = int(np.floor(self.percentlargest_ * widths.shape[0]))
        num_k_lowest = int(widths.shape[0] - num_k_largest)
        # Prevent the denominator to become zero
        n_k_largest = num_k_largest + 0.001
        n_k_lowest = num_k_lowest + 0.001
        
        sum_k_largest_PIwidth = torch.sum(torch.topk(widths, num_k_largest, dim = 0, largest = True)[0].abs(), axis = 0)
        sum_k_lowest_PIwidth = torch.sum(torch.topk(widths, num_k_lowest, dim = 0, largest = False)[0].abs(), axis = 0)
        
        PIwidth = sum_k_largest_PIwidth/(n_k_largest * y_range) + self.lambda_ * sum_k_lowest_PIwidth/(n_k_lowest * y_range)
        
        if self.smoothfunction == 'sigmoid': # s = 100
            k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
            k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
            k_soft = torch.multiply(k_soft_u, k_soft_l)
        elif self.smoothfunction == 'tanh': # s = 50
            k_soft = (1/2)*torch.maximum(torch.zeros(1).to(self.device), torch.tanh(self.soften*(y - y_pred_lower)) + torch.tanh(self.soften*(y_pred_upper - y)))
        elif self.smoothfunction == 'arctan':
            k_soft = (1/torch.pi)*torch.maximum(torch.zeros(1).to(self.device),(torch.arctan(self.soften*(y - y_pred_lower)) - torch.arctan(self.soften*(y - y_pred_upper))))
        else:
            raise ValueError("Input must be sigmoid, arctan, or tanh only")

        PICP_soft = torch.mean(k_soft, axis = 0)
        
        ## Aggregate the loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coverage_loss = torch.max(torch.tensor(0.0, device=device), (1-self.delta_) - PICP_soft)
        width_loss = PIwidth
        loss_soft = coverage_loss + self.gamma_ * width_loss
        
        loss_soft = torch.mean(loss_soft)
        
        if self.returnseparatedloss:
            return loss_soft, coverage_loss, width_loss
        else:
            return loss_soft
        
class qr_objective(nn.Module): # Quantile regression
    def __init__(self, delta_ = 0.1):
        super(qr_objective, self).__init__()
        self.delta_ = delta_
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, y, y_pred_lower, y_pred_upper):
        lower_quantile = self.delta_/2
        upper_quantile = 1 - lower_quantile
        pinball_lower = lower_quantile*torch.maximum(torch.zeros_like(y).to(self.device), y - y_pred_lower) + (1 - lower_quantile)*torch.maximum(torch.zeros_like(y).to(self.device), -(y - y_pred_lower))
        pinball_upper = upper_quantile*torch.maximum(torch.zeros_like(y).to(self.device), y - y_pred_upper) + (1 - upper_quantile)*torch.maximum(torch.zeros_like(y).to(self.device),-(y - y_pred_upper))
        pinball_i = pinball_lower + pinball_upper
        pinball = torch.mean(pinball_i)
        
        return pinball
    
    
class MVE_negloglikeGaussian_objective(nn.Module):
    def __init__(self):
        super(MVE_negloglikeGaussian_objective, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, y, y_pred_mean, y_pred_var):
    
        var = torch.exp(y_pred_var) # To avoid numerical issue
        nll = 0.5 * (((y - y_pred_mean)**2)/var + y_pred_var)
        return torch.mean(nll)
    