import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import init
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

class Comparior(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Pooler(nn.Module):
    def __init__(self, hidden_size,class_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, class_size)
        # self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)
        
        return pooled_output

class DualPooler(nn.Module):
    def __init__(self, hidden_size,attr_size,obj_size,class_size):
        super().__init__()
        
        self.dense_attr = nn.Linear(hidden_size, attr_size)
        self.dense_obj = nn.Linear(hidden_size, obj_size)
        self.dense_pair = nn.Linear(attr_size+obj_size, class_size)
        # self.activation_attr = nn.ReLU()
        # self.activation_obj = nn.ReLU()
        # self.activation_pair = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        second_token_tensor = hidden_states[:, 1]
        pooled_output_attr = self.dense_attr(first_token_tensor)
        pooled_output_obj = self.dense_obj(second_token_tensor)
        pooled_output = torch.cat((F.relu(pooled_output_attr), F.relu(pooled_output_obj)), 1)
        
        pooled_output = self.dense_pair(pooled_output)
        return pooled_output,pooled_output_attr,pooled_output_obj

class DualSimilarityPooler(nn.Module):
    def __init__(self, hidden_size,attr_size,obj_size,class_size,unseen_score,temp=0.05):
        super().__init__()
        
        self.dense_attr = nn.Linear(hidden_size, attr_size)
        self.dense_obj = nn.Linear(hidden_size, obj_size)
        # self.pair_embedding = pair_score
        # self.dense_pair=nn.Parameter(self.pair_embedding)
        self.temp=temp
        self.feasible_mask = unseen_score


    def forward(self, hidden_states,scale=False):
        first_token_tensor = hidden_states[:, 0]
        second_token_tensor = hidden_states[:, 1]
        pooled_output_attr = self.dense_attr(first_token_tensor)
        pooled_output_obj = self.dense_obj(second_token_tensor)


        attr_norm = torch.unsqueeze(F.normalize(pooled_output_attr, dim=-1),-1)
        obj_norm = torch.unsqueeze(F.normalize(pooled_output_obj, dim=-1),-1)

        # img_norm = F.relu(pooled_output_attr)
        # concept_norm = F.relu(pooled_output_obj)
        pooled_output = torch.matmul(attr_norm,obj_norm.transpose(1,2))
        
        if scale:
            pooled_output = pooled_output / self.temp

        pooled_output=torch.flatten(pooled_output, start_dim=1)
        # pooled_output = pooled_output*self.dense_pair
        self.feasible_mask=self.feasible_mask.to(pooled_output.get_device())
        pooled_output = pooled_output*self.feasible_mask
        
        return pooled_output,pooled_output_attr,pooled_output_obj

class DualCombinePooler(nn.Module):
    def __init__(self, hidden_size,attr_size,obj_size,class_size,unseen_score,temp=0.05,neta=0.05):
        super().__init__()
        self.attr_size=attr_size
        self.obj_size=obj_size
        self.dense_attr = nn.Linear(hidden_size, self.attr_size)
        self.dense_obj = nn.Linear(hidden_size, obj_size)
        # self.pair_embedding = pair_score
        self.dense_pair=nn.Linear(hidden_size,attr_size+obj_size)
        # self.activation = nn.Tanh()
        self.temp=temp
        self.neta=neta
        self.feasible_mask = unseen_score
        # attr = torch.ones((attr_size,obj_size), requires_grad=True) 
        # obj = torch.ones((attr_size,obj_size), requires_grad=True)
        self.attr_weights=nn.Parameter(torch.ones((attr_size,obj_size)),requires_grad=True)
        self.obj_weights=nn.Parameter(torch.ones((attr_size,obj_size)),requires_grad=True)
        self.pair_bias=nn.Parameter(torch.ones((attr_size*obj_size)),requires_grad=True)
        # init.kaiming_uniform_(self.attr_weights, a=math.sqrt(5))
        # init.kaiming_uniform_(self.obj_weights, a=math.sqrt(5))
        # fan_in, _ = init._calculate_fan_in_and_fan_out(self.attr_weights)
        # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # init.uniform_(self.pair_bias, -bound, bound)

    def forward(self, hidden_states,scale=False):
        first_token_tensor = hidden_states[:, 0]
        second_token_tensor = hidden_states[:, 1]
        third_token_tensor = hidden_states[:, 2]
        pooled_output_attr = self.dense_attr(first_token_tensor)
        pooled_output_obj = self.dense_obj(second_token_tensor)
        pooled_output_pair = self.dense_pair(third_token_tensor)

        attr_norm = torch.unsqueeze(F.normalize(pooled_output_attr, dim=-1),-1)
        obj_norm = torch.unsqueeze(F.normalize(pooled_output_obj, dim=-1),-1)
        
        # pooled_output_pair=self.activation(pooled_output_pair)

        # img_norm = F.relu(pooled_output_attr)
        # concept_norm = F.relu(pooled_output_obj)
        pooled_output = torch.matmul(attr_norm,obj_norm.transpose(1,2))
        
        if scale:
            pooled_output = pooled_output / self.temp
        pooled_output=torch.flatten(pooled_output, start_dim=1)
        # pooled_output = pooled_output*self.dense_pair
        self.feasible_mask=self.feasible_mask.to(pooled_output.get_device())
        pooled_output = pooled_output*self.feasible_mask
        
        pooled_output_pair=F.relu(pooled_output_pair).unsqueeze(-1)+torch.cat((F.relu(pooled_output_attr.detach()).unsqueeze(-1),F.relu(pooled_output_obj.detach()).unsqueeze(-1)), dim=1)

        attr_projection = torch.flatten(torch.repeat_interleave(pooled_output_pair[:,:self.attr_size],self.obj_size,dim=-1)*self.attr_weights, start_dim=1)

        obj_projection = torch.flatten(pooled_output_pair[:,self.attr_size:].permute(0,2,1).repeat(1,self.attr_size,1)*self.obj_weights,start_dim=1)

        aux_pooled_output=F.relu(attr_projection+obj_projection+self.pair_bias)*self.feasible_mask

        pooled_output_comb=pooled_output+self.neta*aux_pooled_output
        
        return pooled_output_comb,pooled_output,aux_pooled_output,pooled_output_attr,pooled_output_obj
# class SparseLinear(torch.autograd.Function): 
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.save_for_backward(input, weight)
#         output = input.mm(weight.t())
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight = ctx.saved_tensors
#         grad_input = grad_output.mm(weight)
#         grad_weight = grad_output.t().mm(input)
#         return grad_input, grad_weight

# class SparseLinear(nn.Module):
#     def __init__(self,     def __init__(self, attr_size,obj_sizehidden_size):
# ):
#         super().__init__()
#         self.weight_attr = nn.Parameter(2, in_channels // 2)
#         self.weight_obj = nn.Parameter(2, in_channels // 2)
#         self.bias = torch.zeros(in_channels // 2)

#     @staticmethod
#     def forward(self, attr,obj):
#         return torch.matmul(inp.reshape(-1, inp.shape[-1]//2, 2), self.weight) + self.bias
#     @staticmethod
#     def backward(ctx, grad_output):

class MaskingFeasibityMod(nn.Module):
    def __init__(self, attr_size,obj_size,class_size,unseen_score):
        super().__init__()
        # self.w = nn.Parameter(torch.rand(input_size), requires_grad=True)
        self.unseen_score=unseen_score
        self.mask = unseen_score
        # self.mask[:freeze_rows] = 0
    
    # def feasibility(self):
    #     self.feasible_comp=

    def forward(self, x):
        x = x*self.mask#[:,:,None,None,None] 
        # x = self.w * x
        return x

class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x

class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
