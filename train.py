import argparse
import numpy as np
import os
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from bisect import bisect_right
from models.oadis import OADIS
from dataset import CompositionDataset
import evaluator_ge
from tqdm import tqdm
from utils import utils
from config import cfg
from torch.utils.tensorboard import SummaryWriter
from config.config import ex
import copy
import functools
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
import feasibility
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def freeze(m):
    """Freezes module m.
    """
    
    # for modules in m:
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
        p.grad = None


def decay_learning_rate(parameters,wd, cfg):
    """Decays layerwise learning rate using the decay factor in cfg.
    """
    new_parameters = []
    lr_trans= cfg['lr_transformer']
    lr_mult=0.9
    temp_parameters=[p for (n,p) in parameters if ("norm.weight" in n) or ("norm.bias" in n)]
    print('got temp 1',len(temp_parameters))
    # store params & learning rates
    for idx in range(cfg['num_layers'],-1,-1):      
        for (name,param) in parameters:
            if "blocks."+str(idx)+"." in name:
                temp_parameters.append(param)
        print(f'{idx}: lr = {lr_trans:.6f}, {len(temp_parameters)}')

        lr_trans = lr_trans*lr_mult
        print('got temp 2',len(temp_parameters))
        new_parameters.append({'params': temp_parameters, 'lr':lr_trans,"weight_decay":wd})
        temp_parameters=[]

    temp_parameters=[p for (n,p) in parameters if ("token" in n) or ("embed" in n)]
    print('got temp 3',len(temp_parameters))
    new_parameters.append({'params': temp_parameters, 'lr':lr_trans,"weight_decay":wd})
    return new_parameters
    

def save_checkpoint(model_or_optim, name, cfg):
    """Saves checkpoint.
    """
    state_dict = model_or_optim.state_dict()
    path = os.path.join(
        f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}/{name}.pth')
    torch.save(state_dict, path)


def train(epoch, model, optimizer, scheduler,trainloader, logger, device, cfg,_config):
    model.train()

    list_meters = [
        'loss_total'
    ]
    if cfg.MODEL.use_obj_loss:
        list_meters.append('loss_aux_obj')
        list_meters.append('acc_aux_obj')
    if cfg.MODEL.use_attr_loss:
        list_meters.append('loss_aux_attr')
        list_meters.append('acc_aux_attr')
    if cfg.MODEL.use_emb_pair_loss:
        list_meters.append('emb_loss')

    dict_meters = { 
        k: utils.AverageMeter() for k in list_meters
    }

    acc_attr_meter = utils.AverageMeter()
    acc_obj_meter = utils.AverageMeter()
    acc_pair_meter = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    end_time = time.time()

    start_iter = (epoch - 1) * len(trainloader)

    for idx, batch in enumerate(tqdm(trainloader)):
        it = start_iter + idx + 1
        data_time.update(time.time() - end_time)
        for k in batch:
            if isinstance(batch[k], list): 
                continue
            batch[k] = batch[k].to(device, non_blocking=True)
        out = model(batch)
        loss = out['loss_total'].sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        acc_attr_meter.update(out['acc_attr'].mean().detach().item())
        acc_obj_meter.update(out['acc_obj'].mean().detach().item())
        acc_pair_meter.update(out['acc_pair'].mean().detach().item())
        for k in out:
            if k in dict_meters:
                dict_meters[k].update(out[k].mean().detach().item())
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        
        if (idx + 1) % cfg.TRAIN.disp_interval == 0:
            print(
                f'Epoch: {epoch} Iter: {idx+1}/{len(trainloader)}, '
                f'Loss: {dict_meters["loss_total"].avg:.5f},'
                f'Acc_Pair: {acc_pair_meter.avg*100:.2f},'
                f'Batch_time: {batch_time.avg:.3f}, Data_time: {data_time.avg:.3f}',
                flush=True)
            
            for k in out:
                if k in dict_meters:
                    logger.add_scalar('train/%s' % k, dict_meters[k].avg, it)

            logger.add_scalar('train/acc_pair', acc_pair_meter.avg, it)
            last_log_loss=dict_meters["loss_total"].avg  
            batch_time.reset()
            data_time.reset()
            acc_pair_meter.reset()
            acc_attr_meter.reset()
            acc_obj_meter.reset()
            for k in out:
                if k in dict_meters:
                    dict_meters[k].reset()
    del loss
    del out
    
def validate_ge(epoch, model, testloader, evaluator, device,ontesing_set=False, topk=1):
    model.eval()
    dset = testloader.dataset
    val_attrs, val_objs = zip(*dset.pairs)
    val_attrs = [dset.get_text(attr)[1] for attr in val_attrs]
    val_objs = [dset.get_text(obj)[1] for obj in val_objs]
    model.val_pairs = dset.pairs
    acc_val_meter = utils.AverageMeter()
    attr_val_meter = utils.AverageMeter()
    obj_val_meter = utils.AverageMeter()
    _, _, all_attr_gt, all_obj_gt, all_pair_gt, all_pred = [], [], [], [], [], []
    for  data in tqdm((testloader), total=len(testloader), desc='Testing'):
        for k in data:
            if isinstance(data[k], list): 
                continue
            data[k] = data[k].to(device, non_blocking=True)
        with torch.no_grad():
            out = model.module(data)
            

            predictions = out['scores']
            
            for i in list(predictions.keys()):
                    predictions[i]=predictions[i].detach().cpu()

            acc_val_meter.update(out['acc_pair'].item())
            attr_val_meter.update(out['acc_attr'].item())
            obj_val_meter.update(out['acc_obj'].item())
            
            attr_truth, obj_truth, pair_truth = data['attr'].clone().detach(), data['obj'].clone().detach(), data['pair'].clone().detach()

            all_pred.append(predictions)
            all_attr_gt.append(attr_truth)
            all_obj_gt.append(obj_truth)
            all_pair_gt.append(pair_truth)
            del out

    all_attr_gt = torch.cat(all_attr_gt).to('cpu')
    all_obj_gt = torch.cat(all_obj_gt).to('cpu') 
    all_pair_gt = torch.cat(all_pair_gt).to('cpu')
    all_pred_dict = {}
    # # # Gather values as dict of (attr, obj) as key and list of predictions as values
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k].to('cpu') for i in range(len(all_pred))])
    del all_pred
    # # Calculate best unseen accuracy
    results = evaluator.score_model(all_pred_dict, all_obj_gt, bias=1e3, topk=topk)
    stats = evaluator.evaluate_predictions(results, all_attr_gt, all_obj_gt, all_pair_gt, all_pred_dict, topk=topk)
    stats['a_epoch'] = epoch
    stats['pair_accuracy'] = acc_val_meter.avg
    stats['attribute_accuracy'] = attr_val_meter.avg
    stats['object_accuracy'] = obj_val_meter.avg

    acc_val_meter.reset()
    attr_val_meter.reset()
    obj_val_meter.reset()
    result = ''
    # # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    print(f'Val Epoch: {epoch}')
    print(result)

    del val_attrs
    del val_objs
    del model.val_pairs
    del data
    del predictions
    return stats['AUC'], stats['best_hm']

def show_batch(dl, nmax=8):
    for data in dl:
        images=data['img']

        print(torch.max(images),torch.min(images),torch.mean(images),torch.median(images))
        torchvision.utils.save_image(images,f'batch.png')
        break

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main_worker(gpu, cfg,_config):
    """Main training code.
    """
    seed = cfg.TRAIN.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f'Use GPU {gpu} for training', flush=True)
    torch.cuda.set_device(gpu)
    device = f'cuda:{gpu}'

    # Log directory for tensorboard.
    log_dir = f'{cfg.TRAIN.log_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
    logger = SummaryWriter(log_dir=log_dir)

    # Directory to save checkpoints.
    ckpt_dir = f'{cfg.TRAIN.checkpoint_dir}/{cfg.config_name}_{cfg.TRAIN.seed}'
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=False)
    os.makedirs(ckpt_dir)

    cfg.TRAIN.batch_size = _config['per_gpu_batchsize']
    print('Batch size on each gpu: %d' % cfg.TRAIN.batch_size)
    
    print('Prepare dataset with',cfg.TRAIN.num_workers)

    trainset = CompositionDataset(
        phase='train', split=cfg.DATASET.splitname, cfg=cfg)


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.TRAIN.batch_size, shuffle=True,
        num_workers=cfg.TRAIN.num_workers,
        pin_memory=True, drop_last=False, worker_init_fn=seed_worker)

    show_batch(trainloader)  
         
    valset = CompositionDataset(
        phase='val', split=cfg.DATASET.splitname, cfg=cfg)


    valloader = torch.utils.data.DataLoader(
        valset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
        num_workers=cfg.TRAIN.num_workers)

    
    testset = CompositionDataset(
        phase='test', split=cfg.DATASET.splitname, cfg=cfg)


    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.TRAIN.test_batch_size, shuffle=False,
        num_workers=cfg.TRAIN.num_workers)

    #set threshold
    feasibility_threshold,unseen_scores=feasibility.thresholding(valset.val_seen_mask,trainset.seen_mask,cfg.DATASET.name,_config['offset_val'])
    unseen_scores=unseen_scores.to(device)
    cfg.DATASET.feasibility_threshold=feasibility_threshold.item()
    model = OADIS(trainset, unseen_scores, cfg,_config)

    for i in range(_config['num_freeze_layers']):
        freeze(model.transformer.blocks[i])
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    print('Feasibility thereshold:',feasibility_threshold.item())

    
    total_params = utils.count_parameters(model)

    evaluator_val_ge = evaluator_ge.Evaluator(valset, model)
    evaluator_test_ge = evaluator_ge.Evaluator(testset, model)
    
    torch.backends.cudnn.benchmark = True

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    params_word_embedding = []
    params_name_transformer_nd = []
    params_name_transformer_d = []
    params_cross_nd = []
    params_cross_d = []
    params_nd = []
    params_d = []
    opt_parameters = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if cfg.TRAIN.lr_word_embedding > 0:      
            params_word_embedding.append(p)
            print('params_word_embedding: %s' % name)

        if (name.startswith('module.transformer_cross_attr') or name.startswith('module.transformer_cross_obj')):
            if any(nd in name for nd in no_decay):
                params_cross_nd.append(p)
            else:
                params_cross_d.append(p)
            print('params_cross: %s' % name)

        elif name.startswith('module.transformer') or name.startswith('module.token_type_embeddings'): 
            if any(nd in name for nd in no_decay):
                params_name_transformer_nd.append((name,p))
            else:
                params_name_transformer_d.append((name,p))
            print('params_transformer: %s' % name)

        else:
            if any(nd in name for nd in no_decay):
                params_nd.append(p)
            else:
                params_d.append(p)
            print('params_main: %s' % name)

    #layerwise lr for transformer

    opt_parameters.append({"params": params_cross_nd, "lr": _config['lr_cross'], "weight_decay": 0.0001})
    opt_parameters.append({"params": params_cross_d, "lr": _config['lr_cross'], "weight_decay": cfg.TRAIN.wd})
    opt_parameters.extend(decay_learning_rate(params_name_transformer_nd,0.0001,_config))
    opt_parameters.extend(decay_learning_rate(params_name_transformer_d,cfg.TRAIN.wd,_config))
    opt_parameters.append({"params": params_nd, "lr": _config['lr'], "weight_decay": 0.0001})
    opt_parameters.append({"params": params_d, "lr": _config['lr'], "weight_decay": cfg.TRAIN.wd})
    
 
    optimizer = optim.AdamW(opt_parameters, lr=_config['lr_cross'], weight_decay=cfg.TRAIN.wd)
    group_lrs = [_config['lr_transformer'], _config['lr'], _config['lr_cross']]

    for i in range(len(optimizer.param_groups)):
        print(i,len(optimizer.param_groups[i]['params']),optimizer.param_groups[i]['lr'])

    warmup_steps = cfg.TRAIN.warmup_steps
    if isinstance(cfg.TRAIN.warmup_steps, float):
        warmup_steps = int(len(trainloader) * cfg.TRAIN.max_epoch * cfg.TRAIN.warmup_steps)
    print('number of warmup steps:', warmup_steps,len(trainloader))

    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(trainloader) * cfg.TRAIN.max_epoch,
            num_cycles=0.5

        )

    start_epoch = cfg.TRAIN.start_epoch
    epoch = start_epoch

    best_records = {
        'val/best_auc': 0.0,
        'val/best_hm': 0.0,
        'test/auc_at_best_val': 0.0,
        'test/hm_at_best_val': 0.0,
    }

    best_auc = -1

    while epoch <= cfg.TRAIN.max_epoch:

        train(epoch, model, optimizer,scheduler, trainloader, logger, device, cfg,_config)
        print('current lr for the epoch',scheduler._last_lr)
        wandb.log({'lr0':scheduler._last_lr[0],'lr1':scheduler._last_lr[1]})

        max_gpu_usage_mb = torch.cuda.max_memory_allocated(device=device) / 1048576.0
        print(f'Max GPU usage in MB till now: {max_gpu_usage_mb}')

        if epoch < cfg.TRAIN.start_epoch_validate:
            epoch += 1
            continue
        if epoch % cfg.TRAIN.eval_every_epoch == 0:
            
            # Validate.
            print('Validation set ===>')
            auc, best_hm = validate_ge(epoch, model, valloader, evaluator_val_ge, device,ontesing_set=False, topk=cfg.EVAL.topk)
            logger.add_scalar('val/auc', auc, epoch * len(trainloader))
            logger.add_scalar('val/best_hm', best_hm, epoch * len(trainloader))

            if (auc > best_auc or auc / best_auc >= 0.99) and epoch == cfg.TRAIN.max_epoch and epoch+1 < cfg.TRAIN.final_max_epoch:
                cfg.TRAIN.max_epoch += 1

            if auc > best_records['val/best_auc']:
                best_records['val/best_auc'] = auc
                best_records['val/best_hm'] = best_hm
                print('Beat best Val AUC, now evaluate on test set')
                
                # Test.
                print('Testing set ===>')
                auc, best_hm = validate_ge(epoch, model, testloader, evaluator_test_ge,device,ontesing_set=True,topk=cfg.EVAL.topk)
                
                logger.add_scalar('test/auc', auc, epoch * len(trainloader))
                logger.add_scalar('test/best_hm', best_hm, epoch * len(trainloader))
                best_records['test/auc_at_best_val'] = auc
                best_records['test/hm_at_best_val'] = best_hm
                save_checkpoint(model, f'model_epoch{epoch}', cfg)
        print(f'Ending epoch {epoch}')
        
        epoch += 1

    logger.close()
    
    print('Done: %s' % cfg.config_name)
    print('New Best AUC:',best_records['val/best_auc'])
    print('New Best HM:',best_records['val/best_hm'])
                
@ex.automain
def main(_config):

    # get number of GPUs available
    print(torch.cuda.device_count()) 

    # get the name of the device
    print(torch.cuda.get_device_name(0))

    _config = copy.deepcopy(_config)

    cfg.merge_from_file(_config['cfg'])

    print(cfg)
    print(_config)

    seed = cfg.TRAIN.seed
    if seed == -1:
        seed = np.random.randint(1, 10000)
    print('Random seed:', seed)
    cfg.TRAIN.seed = seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    
    main_worker(0, cfg,_config)

