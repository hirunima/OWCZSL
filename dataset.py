import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as tdata
import torchvision.models as tmodels
import torchvision.transforms as transforms
import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizer,
)
from PIL import Image
import random
from itertools import product
BICUBIC = transforms.InterpolationMode.BICUBIC
n_px = 224

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = f'{self.img_dir}/{img}'
        try:
            img = Image.open(file).convert('RGB')

        except Exception as e:
            if "_" in file.split("/")[-2]:
                file=self.img_dir+'/'+file.split("/")[-2].replace("_"," ")+"/"+file.split("/")[-1]
        #     elif  "-" in file.split("/")[-1]:
        #         file=self.img_dir+'/'+'_'.join(file.split("/")[-1].split(".")[-2].split("-")[-2:])+"/"+file.split("/")[-1]
                
            img = Image.open(file).convert('RGB')

        # img = Image.open(file).convert('RGB')
        return img


def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if phase == 'train':
        transform = transforms.Compose(
            [
                transforms.transforms.RandomResizedCrop(n_px),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean,
                    std,
                ),
            ]
        )

    elif phase == "test" or phase == "val":
        transform = transforms.Compose(
            [
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform


def imagenet_transform_zappos(phase, cfg):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase == 'train':
        transform = transforms.Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                transforms.RandomHorizontalFlip(),
                transforms.RandomPerspective(),
                transforms.RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    elif phase == 'test' or phase == 'val':
        transform = transforms.Compose(
            [
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform


class CompositionDataset(tdata.Dataset):
    def __init__(
        self,
        phase,
        split='compositional-split',
        open_world=True,
        max_text_len=8,
        cfg=None
    ):
        self.phase = phase
        self.cfg = cfg
        self.split = split
        self.open_world = open_world

        if 'ut-zap50k' in cfg.DATASET.name:
            # self.transform = imagenet_transform_zappos(phase, cfg)
            self.transform = imagenet_transform(phase)
        else:
            self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(f'{cfg.DATASET.root_dir}/images')
        
        self.attrs, self.objs, self.close_pairs, \
            self.train_pairs, self.val_pairs, \
            self.test_pairs = self.parse_split()

        self.pairs = sorted(list(product(self.attrs, self.objs)))
        #self.pairs = sorted(sorted(list(product(self.attrs, self.objs))), key=lambda n: n.split()[1])  #uncomment this if you want to sort by object name

        if not(self.open_world):
            self.close_world_mask = [1 if any_pair in self.close_pairs else 0 for any_pair in self.pairs]

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        tokenizer = "bert-base-uncased"
        self.tokenizer = self.get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size
        self.max_text_len=max_text_len


        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}
        self.train_pair2idx = {pair: idx for idx, pair in enumerate(self.train_pairs)}
        # import pdb; pdb.set_trace()

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d | # open world pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs), len(self.pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))


        if cfg.TRAIN.use_precomputed_features:
            feat_file = f'{cfg.DATASET.root_dir}/features.t7'
            feat_avgpool = True
            if not os.path.exists(feat_file):
                with torch.no_grad():
                    self.generate_features(feat_file, feat_avgpool)

            activation_data = torch.load(feat_file)
            self.activations = dict(
                zip(activation_data['files'], activation_data['features']))
            self.feat_dim = activation_data['features'].size(1)

            print('%d activations loaded' % (len(self.activations)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask)

            val_mask = [1 if pair in set(self.val_pairs) else 0 for pair in self.pairs]
            # self.val_seen_mask = torch.BoolTensor(val_mask) * 1.
            self.val_seen_mask = torch.logical_or(torch.BoolTensor(val_mask) ,self.seen_mask)
            
            test_mask = [1 if pair in set(self.test_pairs) else 0 for pair in self.pairs]
            # self.test_seen_mask = torch.logical_or(torch.BoolTensor(test_mask) ,self.val_seen_mask) * 1.
            self.test_seen_mask = torch.BoolTensor(test_mask)  * 1.

            self.seen_mask =self.seen_mask * 1.
            self.val_seen_mask =self.val_seen_mask * 1.


            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)


        # Affordance.
        self.attr_affordance = {} # -> contains objects compatible with an attribute.
        for _attr in self.attrs:
            candidates = [
                obj
                for (_, attr, obj) in self.train_data
                if attr == _attr
            ]
            self.attr_affordance[_attr] = sorted(list(set(candidates)))
            if len(self.attr_affordance[_attr]) <= 1:
                print(f'{_attr} is associated with <= 1 object: {self.attr_affordance[_attr]}')
        
        self.obj_affordance = {} # -> contains attributess compatible with an object.
        for _obj in self.objs:
            candidates = [
                attr
                for (_, attr, obj) in self.train_data
                if obj == _obj
            ]
            self.obj_affordance[_obj] = sorted(list(set(candidates)))
            if len(self.obj_affordance[_obj]) <= 1:
                print(f'{_obj} is associated with <= 1 object: {self.obj_affordance[_obj]}')

        # Images that contain an object.
        self.image_with_obj = {}
        for i, instance in enumerate(self.train_data):
            obj = instance[2]
            if obj not in self.image_with_obj:
                self.image_with_obj[obj] = []
            self.image_with_obj[obj].append(i)
        
        # Images that contain an attribute.
        self.image_with_attr = {}
        for i, instance in enumerate(self.train_data):
            attr = instance[1]
            if attr not in self.image_with_attr:
                self.image_with_attr[attr] = []
            self.image_with_attr[attr].append(i)

        # Images that contain a pair.
        self.image_with_pair = {}
        for i, instance in enumerate(self.train_data):
            attr, obj = instance[1], instance[2]
            if (attr, obj) not in self.image_with_pair:
                self.image_with_pair[(attr, obj)] = []
            self.image_with_pair[(attr, obj)].append(i)
        
        if cfg.MODEL.use_composed_pair_loss:
            # with open('unseen_pairs/'+cfg.DATASET.name+'_unseen_pairs.txt', 'r') as f:
            #     self.unseen_pairs = [tuple(l.strip().split()) for l in f.readlines()]
            unseen_pairs = set()
            for pair in self.val_pairs + self.test_pairs:
                if pair not in self.train_pair2idx:
                    unseen_pairs.add(pair)
            self.unseen_pairs = list(unseen_pairs)
            self.unseen_pair2idx = {pair: idx for idx, pair in enumerate(self.unseen_pairs)}

    def get_text(self, text):
        #index, caption_index = self.index_mapper[raw_index]

        # text = self.all_texts[raw_index]#[caption_index]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )

        return (text, encoding)
        
            
    def get_split_info(self):
        data = torch.load(f'{self.cfg.DATASET.root_dir}/metadata_{self.split}.t7')
        train_data, val_data, test_data = [], [], []

        for instance in data:
            
            image, attr, obj, settype = \
                instance['image'], instance['attr'], instance['obj'], instance['set']
            if attr == 'NA' or (attr, obj) not in self.pairs or settype == 'NA':
                continue
            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

   #from vilt module
    def get_pretrained_tokenizer(self,from_pretrained):
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                BertTokenizer.from_pretrained(
                    from_pretrained, do_lower_case="uncased" in from_pretrained
                )
            torch.distributed.barrier()
        return BertTokenizer.from_pretrained(
            from_pretrained, do_lower_case="uncased" in from_pretrained
        )
    
    

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                if self.cfg.DATASET.name == 'vaw-czsl':
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            
            return attrs, objs, pairs
        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/train_pairs.txt')
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/val_pairs.txt')
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            f'{self.cfg.DATASET.root_dir}/{self.split}/test_pairs.txt')

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def __getitem__(self, index):
        
        image, attr, obj = self.data[index]
        if self.cfg.TRAIN.use_precomputed_features:
            # img = self.activations[image]
            try:
                img = self.activations[image]
            except Exception as e:
                if "_" in image.split("/")[0]:
                    image=image.split("/")[0].replace("_"," ")+"/"+image.split("/")[1]
                    img = self.activations[image]
        else:
            
            img = self.loader(image)
            img = self.transform(img)
        
        k = random.randint(0, 1)
        # v = random.randint(0, 1)
        # if 'vaw-czsl' in self.cfg.DATASET.name:
        #     objattpair=image.split('/')[0].split('+')
        # else:
        #     objattpair=image.split('/')[0].split('_')
        objattpair=[attr, obj]
        objattpair=' '.join(objattpair)

        if self.phase == 'train':

            data = {
                'img': img,
                'text_pair' : objattpair,
                'kv':(k,0),
                'img_name': self.data[index][0],
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
            }
            text_labels=torch.zeros(len(self.attrs)+len(self.objs))
            text_labels[data['attr']]=1
            text_labels[len(self.attrs)+data['obj']]=1
            data['text_labels']=text_labels

            data['mask_task'] = 1 # Attribute task
            i2 = self.sample_same_attribute(attr, obj, with_different_obj=True)
            if i2 == -1:
                data['mask_task'] = 0
                i2=index
            img1, attr1, obj1_a = self.data[i2]

            if self.cfg.TRAIN.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)
            data['img1_a'] = img1

            data['text_pair_a'] = attr1+' '+obj1_a
            data['idx1_a'] = i2
            data['img1_name_a'] = self.data[i2][0]

            # Object task.
            i2 = self.sample_same_object(attr, obj, with_different_attr=True)
            if i2 == -1:
                data['mask_task'] = 0
                i2=index
            img1, attr1_o, obj1 = self.data[i2]

            if self.cfg.TRAIN.use_precomputed_features:
                img1 = self.activations[img1]
            else:
                img1 = self.loader(img1)
                img1 = self.transform(img1)

            data['img1_o'] = img1
            data['text_pair1_o'] = attr1_o+' '+obj1
            data['idx1_o'] = i2
            data['img1_name_o'] = self.data[i2][0]

            if self.cfg.MODEL.use_composed_pair_loss:
                if (attr1_o, obj1_a) in self.unseen_pair2idx:
                    data['composed_unseen_pair'] = self.unseen_pair2idx[(attr1_o, obj1_a)]
                    data['composed_seen_pair'] = 2000
                elif (attr1_o, obj1_a) in self.train_pair2idx:
                    data['composed_seen_pair'] = self.train_pair2idx[(attr1_o, obj1_a)]
                    data['composed_unseen_pair'] = 2000
                else:
                    data['composed_unseen_pair'] = 2000
                    data['composed_seen_pair'] = 2000

        else:
            # Testing mode.
            data = {
                'img': img,
                'text_pair' : attr+' '+obj,
                'attr': self.attr2idx[attr],
                'obj': self.obj2idx[obj],
                'pair': self.pair2idx[(attr, obj)],
            }
        return data

    def __len__(self):
        return len(self.data)

    def sample_same_attribute(self, attr, obj, with_different_obj=True):
        if with_different_obj:
            if len(self.attr_affordance[attr]) == 1:
                return -1
            i2 = np.random.choice(self.image_with_attr[attr])
            img1, attr1, obj1 = self.data[i2]
            while obj1 == obj:
                i2 = np.random.choice(self.image_with_attr[attr])
                img1, attr1, obj1 = self.data[i2]
            assert obj1 != obj
        else:
            i2 = np.random.choice(self.image_with_attr[attr])
        return i2

    def sample_same_object(self, attr, obj, with_different_attr=True):        
        if with_different_attr:
            if len(self.obj_affordance[obj]) == 1:
                return -1
            i2 = np.random.choice(self.image_with_obj[obj])
            img1, attr1, obj1 = self.data[i2]
            while attr1 == attr:
                i2 = np.random.choice(self.image_with_obj[obj])
                img1, attr1, obj1 = self.data[i2]
        return i2

    def generate_features(self, out_file, feat_avgpool=True):
        data = self.train_data + self.val_data + self.test_data
        transform = imagenet_transform('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(
                chunks(data, 512), total=len(data) // 512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            imgs = torch.stack(imgs, 0).cuda()
            if feat_avgpool:
                feats = feat_extractor(imgs)
            else:
                feats = feat_extractor.conv1(imgs)
                feats = feat_extractor.bn1(feats)
                feats = feat_extractor.relu(feats)
                feats = feat_extractor.maxpool(feats)
                feats = feat_extractor.layer1(feats)
                feats = feat_extractor.layer2(feats)
                feats = feat_extractor.layer3(feats)
                feats = feat_extractor.layer4(feats)
                assert feats.shape[-3:] == (512, 7, 7), feats.shape
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print('features for %d images generated' % (len(image_files)))
        torch.save({'features': image_feats, 'files': image_files}, out_file)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]