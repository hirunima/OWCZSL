import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from .bert_transformer import BertEmbedder

from . import vision_transformer_multitoken as vit
from . import cross_attention as cross_vit
# from . import vision_transformer_vit as vit_img
from . import heads, objectives#, vilt_utils

class OADIS(nn.Module):
    """Object-Attribute Compositional Learning from Image Pair.
    """
    def __init__(self, dset,unseen_scores, cfg,config):
        super(OADIS, self).__init__()
        
        self.cfg = cfg
        self.dset=dset
        self.unseen_scores=unseen_scores
        self.num_attrs = dset.attrs
        self.num_objs = dset.objs
        self.pair2idx = dset.pair2idx
        self.attr2idx = dset.attr2idx
        self.obj2idx = dset.obj2idx
        self.config=config

        # Set training pairs.
        train_attrs, train_objs = zip(*dset.pairs)
        self.val_pairs=dset.pairs

        train_attrs = [dset.attr2idx[attr] for attr in train_attrs]
        train_objs = [dset.obj2idx[obj] for obj in train_objs]

        #train dataset labels.
        self.train_attrs = torch.LongTensor(train_attrs).cuda()
        self.train_objs = torch.LongTensor(train_objs).cuda()

        self.text_embeddings = BertEmbedder(config=self.config)
        # self.text_embeddings.apply(objectives.init_weights)

        self.token_type_embeddings = nn.Embedding(2, config["hidden_size"])
        self.token_type_embeddings.apply(objectives.init_weights)

        for param in self.text_embeddings.model.parameters():
            param.requires_grad = False

        if self.config["load_path"] == "":
            self.transformer = getattr(vit, self.config["vit"])(
                pretrained=True, config=self.config
            )
        else:
            self.transformer = getattr(vit, self.config["vit"])(
                pretrained=True, config=self.config
            )

        self.transformer_cross_attr = cross_vit.CrossAttentionBlock(dim=self.config["hidden_size"], num_heads=1)
        self.transformer_cross_obj = cross_vit.CrossAttentionBlock(dim=self.config["hidden_size"], num_heads=1)

        self.text_embeds_attr=[]

        for attrs_emb in self.num_attrs:
            _, encoding=self.text_embeddings.get_text(attrs_emb)
            self.text_embeds_attr.append(self.text_embeddings.get_bert_embeddings(encoding["input_ids"],encoding["attention_mask"]))
        self.text_embeds_attr=torch.stack(self.text_embeds_attr).squeeze(1)
        
        self.text_embeds_obj=[]
        for objs_emb in self.num_objs:
            _, encoding=self.text_embeddings.get_text(objs_emb)
            self.text_embeds_obj.append(self.text_embeddings.get_bert_embeddings(encoding["input_ids"],encoding["attention_mask"]))
        self.text_embeds_obj=torch.stack(self.text_embeds_obj).squeeze(1)


        self.pooler = heads.DualCombinePooler(self.config["hidden_size"],len(self.num_attrs),len(self.num_objs),len(dset.pairs),self.unseen_scores,neta=config['neta'])

        self.pooler.apply(objectives.init_weights)
        

    def infer(
        self,
        img,
        mask_text=False,
        mask_image=False,
        text_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
        max_image_len=-1,
            ):


        if image_embeds is None and image_masks is None:
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=max_image_len,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        #seperate obj attr cross attention modules
        text_embeds_attr=self.text_embeds_attr.unsqueeze(0).repeat(image_embeds.size(0),1,1).to(img.get_device())
        text_embeds_obj=self.text_embeds_obj.unsqueeze(0).repeat(image_embeds.size(0),1,1).to(img.get_device())

        #for detaching the image embeds https://discuss.pytorch.org/t/how-to-detach-specific-components-in-the-loss/13983/7
        _,attn_attr=self.transformer_cross_attr(text_embeds_attr,image_embeds.detach())
        # torchvision.utils.save_image(attn_attr[0,0,:,:]*255.,f'attn_maps/attr.png')
        _,attn_obj=self.transformer_cross_obj(text_embeds_obj,image_embeds.detach())
        # torchvision.utils.save_image(attn_obj[0,0,:,:]*255.,f'attn_maps/obj.png')
        
        k=self.config["k"]
        attn_attr=torch.sum(attn_attr.squeeze(1), dim=1)
        attr_k=torch.topk(attn_attr,k,dim=-1)
        attn_obj=torch.sum(attn_obj.squeeze(1), dim=1)
        obj_k=torch.topk(attn_obj,k,dim=-1)

        text_embeds=torch.cat((text_embeds_attr[torch.arange(text_embeds_attr.size(0)).unsqueeze(1), attr_k[1],:],text_embeds_obj[torch.arange(text_embeds_obj.size(0)).unsqueeze(1), obj_k[1],:]),dim=1).to(img.get_device())
        
        text_masks=torch.ones_like(text_embeds[:,:,0]).type(torch.LongTensor).to(img.get_device())
        image_embeds,text_embeds = (
            image_embeds
            + self.token_type_embeddings(torch.zeros_like(image_masks)),
            text_embeds + self.token_type_embeddings(torch.full_like(text_masks, text_token_type_idx))
        )

        co_embeds = torch.cat([ image_embeds,text_embeds], dim=1)
        co_masks = torch.cat([image_masks,text_masks], dim=1)
        
        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        class_feats,main_class_feats,aux_class_feats,attr_feats,obj_feats = self.pooler(x)

        ret = {
            "cls_feats": class_feats,
            "main_cls_feats": main_class_feats,
            "aux_cls_feats":aux_class_feats,
            "obj_feats": obj_feats,
            "attr_feats": attr_feats,
            "attn_attr":attn_attr,
            "attn_obj":attn_obj,
 
        }

        return ret


    def train_forward(self,batch):

        img = batch['img'].to('cuda')
        bs=img.size(0)

        img2_a = batch['img1_a'].to('cuda') # Image that shares the same attribute
        img2_o = batch['img1_o'].to('cuda') # Image that shares the same object
        # Labels of 1st image.
        attr_labels = batch['attr'].to('cuda')
        obj_labels = batch['obj'].to('cuda')
        pair_labels = batch['pair'].to('cuda')
        k,v=batch['kv'][0],batch['kv'][0]

        mask_task = batch['mask_task']
        
        text_labels=batch["text_labels"].type(torch.FloatTensor).to('cuda')

        infer_pair = self.infer(img, mask_text=False, mask_image=False)
        infer_attr = self.infer(img2_a, mask_text=False, mask_image=False)
        infer_obj = self.infer(img2_o, mask_text=False, mask_image=False)
        
        calc_class=objectives.compute_calss_aux(infer_pair,attr_labels,obj_labels,pair_labels,self.config)

        calc_attr=objectives.compute_second(infer_attr,attr_labels,'attr',self.config)
        calc_obj=objectives.compute_second(infer_obj,obj_labels,'obj',self.config)
        calc_topk=objectives.compute_topk(infer_pair,attr_labels,obj_labels,self.config)
 
        correct_pair = torch.eq(torch.max(calc_class['pred'].data, 1).indices,pair_labels)
        correct_attr = torch.eq(torch.max(calc_class['pred_attr'].data, 1).indices,attr_labels)
        correct_obj = torch.eq(torch.max(calc_class['pred_obj'].data, 1).indices,obj_labels)
            

        out = {
            'loss_total': calc_class['class_loss']+calc_attr['class_loss']+calc_obj['class_loss']+0.8*calc_topk['topk_loss'],
            'acc_attr': torch.div(correct_attr.sum(),float(bs)), 
            'acc_obj': torch.div(correct_obj.sum(),float(bs)), 
            'acc_pair': torch.div(correct_pair.sum(),float(bs)),
            }
        return out

    def val_forward(self, batch):
 
        img = batch['img'].to('cuda')
        bs=img.size(0)
        pair_labels = batch['pair'].to('cuda')
        attr_labels = batch['attr'].to('cuda')
        obj_labels = batch['obj'].to('cuda')

        infer = self.infer(img, mask_text=False, mask_image=False)

        calc_class=objectives.compute_calss_aux(infer,attr_labels,obj_labels,pair_labels,self.config)
        
        out = {}
        out['scores'] = calc_class['pred']
        
        correct_pair = torch.eq(torch.max(calc_class['pred'].data, 1).indices,pair_labels)
        correct_attr = torch.eq(torch.max(calc_class['pred_attr'].data, 1).indices,attr_labels)
        correct_obj = torch.eq(torch.max(calc_class['pred_obj'].data, 1).indices,obj_labels)
        
        out['scores'] = {}
        for _, pair in enumerate(self.val_pairs):

            out['scores'][pair] = calc_class['pred'][:,self.pair2idx[pair]].detach()
        
        out['acc_pair']=torch.div(correct_pair.sum(),float(bs))
        out['acc_attr']=torch.div(correct_attr.sum(),float(bs))
        out['acc_obj']=torch.div(correct_obj.sum(),float(bs))

        
        return out


    def forward(self, x):
        if self.training:
            out = self.train_forward(x)
        else:
            with torch.no_grad():
                out = self.val_forward(x)
        return out


class CosineClassifier(nn.Module):
    def __init__(self, temp=1):
        super(CosineClassifier, self).__init__()
        self.temp = temp

    def forward(self, concept1, concept2, scale=True):
        """
        img: (bs, emb_dim)
        concept: (n_class, emb_dim)
        """
        concept1_norm = F.normalize(concept1, dim=-1)
        concept2_norm = F.normalize(concept2, dim=-1)
        pred = torch.matmul(concept1_norm, concept2_norm.transpose(0, 1))
        if scale:
            pred = pred / self.temp
        pred=torch.sum(torch.diagonal(pred))
        return pred