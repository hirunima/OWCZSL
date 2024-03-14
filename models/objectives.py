import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange

from .dist_utils import all_gather


def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance



def compute_mlm(mlm_score,infer,text_embeds,config):
    mlm_logits = mlm_score(infer["text_feats"])
    mlm_labels = infer["text_labels"]

    
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    ret = {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        # "mlm_ids": infer["text_ids"],
    }

    return ret

def compute_calss_aux(infer_pair,attr_labels,obj_labels,pair_labels,config):
    predict_labels = infer_pair["cls_feats"]
    predict_main_labels = infer_pair["main_cls_feats"]
    predict_aux_labels = infer_pair["aux_cls_feats"]
    predict_labels_attr =infer_pair["attr_feats"]
    predict_labels_obj =infer_pair["obj_feats"]
    
    class_loss = F.cross_entropy(
        predict_labels_attr,
        attr_labels)+F.cross_entropy(
        predict_labels_obj,
        obj_labels)+F.cross_entropy(
        predict_aux_labels,
        pair_labels)
    ret = {
        "class_loss": class_loss,
        "pred": predict_labels.detach(),
        "pred_obj": predict_labels_obj.detach(),
        "pred_attr": predict_labels_attr.detach()
        }
    

    return ret

def compute_calss(infer_pair,attr_labels,obj_labels,pair_labels,config):
    predict_labels = infer_pair["cls_feats"]
    # predict_main_labels = infer_pair["main_cls_feats"]
    predict_labels_attr =infer_pair["attr_feats"]
    predict_labels_obj =infer_pair["obj_feats"]
    
    class_loss = F.cross_entropy(
        predict_labels_attr,
        attr_labels)+F.cross_entropy(
        predict_labels_obj,
        obj_labels)
    ret = {
        "class_loss": class_loss,
        "pred": predict_labels.detach(),
        "pred_obj": predict_labels_obj.detach(),
        "pred_attr": predict_labels_attr.detach()
        }
    

    return ret

def compute_second(infer,labels,st,config):
    if st=='attr':
        predict_labels =infer["attr_feats"]
    else:
        predict_labels =infer["obj_feats"]
    

    class_loss = F.cross_entropy(
        predict_labels,
        labels)
        # +0.1*F.cross_entropy(
        # predict_labels,
        # pair_labels)
    ret = {
        "class_loss": class_loss,
        }
    

    return ret

def compute_obj(infer_pair,infer_obj,config):
    predict_labels_pair_obj = infer_pair["obj_feats"]
    predict_labels_obj_obj =infer_obj["obj_feats"]

    """
    Cross-entropy between softmax outputs of the teacher and student networks.
    """
    # student_out = student_output / self.student_temp
    # student_out = student_out.chunk(self.ncrops)

    # # teacher centering and sharpening
    # temp = self.teacher_temp_schedule[epoch]
    # teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
    # teacher_out = teacher_out.detach().chunk(2)

    # total_loss = 0
    # n_loss_terms = 0
    # for iq, q in enumerate(predict_labels_pair_obj):
    #     for v in range(len(predict_labels_obj_obj)):
    #         # if v == iq:
    #         #     # we skip cases where student and teacher operate on the same view
    #         #     continue
    #         loss = torch.sum(-F.softmax(predict_labels_pair_obj[iq], dim=-1) * F.log_softmax(predict_labels_obj_obj[v], dim=-1), dim=-1)
    #         total_loss += loss.mean()
    #         n_loss_terms += 1
    
    # total_loss /= n_loss_terms
    # # self.update_center(teacher_output)
    # total_loss=total_loss#+
    total_loss=F.mse_loss(F.sigmoid(predict_labels_pair_obj),F.sigmoid(predict_labels_obj_obj))
    ret = {
        "obj_loss": total_loss,
        # "pred": predict_labels,
        # "pred_obj": predict_labels_obj,
        }
    

    return ret

def compute_attr(infer_pair,infer_attr,config):

    predict_labels_pair_attr = infer_pair["attr_feats"]
    predict_labels_attr_attr =infer_attr["attr_feats"]
    
    class_loss = F.mse_loss(F.sigmoid(predict_labels_pair_attr),F.sigmoid(predict_labels_attr_attr))#F.kl_div(predict_labels_attr_attr.log(), predict_labels_pair_attr, None, None, 'sum')#+

    ret = {
        "attr_loss": class_loss,
        }

    return ret

def compute_topk(infer_pair,attr_labels,obj_labels,config):
    logits_attr = infer_pair["attn_attr"]
    logits_obj = infer_pair["attn_obj"]
    
    # topk_loss = F.binary_cross_entropy_with_logits(logits_attr, attr_labels)+F.binary_cross_entropy_with_logits(logits_obj, obj_labels)
    topk_loss = F.cross_entropy(
        logits_attr,
        attr_labels)+F.cross_entropy(
        logits_obj,
        obj_labels)

    # nn.BCEWithLogitsLoss()
    ret = {
        "topk_loss": topk_loss,
        }

    return ret

def compute_cosine(infer_pair,attr_labels,obj_labels,pair_labels,mask_task,config):
    predict_pair_attr =infer_pair["attr_feats"]
    predict_pair_obj =infer_pair["obj_feats"]

    predict_attr_attr =infer_pair["attr_feats"]
    predict_attr_obj =infer_pair["obj_feats"]

    predict_obj_attr =infer_pair["attr_feats"]
    predict_obj_obj =infer_pair["obj_feats"]

    y_pos=torch.ones(predict_pair_attr.size(0))
    y_neg=-1*torch.ones(predict_pair_attr.size(0))
    criterian=nn.CosineEmbeddingLoss(reduction='none')

    loss=criterian(predict_pair_attr,predict_attr_attr,y_pos)+ criterian(predict_pair_obj,predict_obj_obj,y_pos)+mask_task*criterian(predict_pair_attr,predict_obj_attr,y_neg)+mask_task*criterian(predict_pair_obj,predict_attr_obj,y_neg)


    ret = {
        "cosine_loss": loss.sum(),
        }

    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

