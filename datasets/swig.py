from pathlib import Path
from PIL import Image
import json
import pickle as pkl
from collections import defaultdict
import numpy as np
import random
import csv
import torch
import torch.utils.data
import torchvision
from timm.data.transforms_factory import create_transform
from timm.data.constants import (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

import datasets.transforms as T

class GSRDetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, noun_list, verb_list, role_list, verb_info_file, transforms, pad_box):
        self.img_set = img_set
        self.img_folder = img_folder
        self._transforms = transforms

        with open(verb_info_file) as f:
            self.verb_to_role_order = {}
            self.verb_to_abs = {}
            self.noun_to_word = {}
            all = json.load(f)
            for verb, verb_info in all['verbs'].items():
                self.verb_to_role_order[verb] = verb_info['order']
                self.verb_to_abs[verb] = verb_info['abstract']#.lower()
            for noun, word in all['nouns'].items():
                self.noun_to_word[noun] = word['gloss'][0]

        with open(noun_list, 'r') as file:
            self.noun_to_idx, self.idx_to_noun = self.load_classes(file)

        self.verb_to_idx = {}
        self.idx_to_verb = []
        with open(verb_list) as f:
            for k,line in enumerate(f):
                verb = line.split('\n')[0]
                self.idx_to_verb.append(verb)
                self.verb_to_idx[verb] = k

        self.role_to_idx = {}
        self.idx_to_role = []
        with open(role_list) as f:
            for k,line in enumerate(f):
                role = line.split('\n')[0]
                self.idx_to_role.append(role)
                self.role_to_idx[role] = k

        with open(anno_file) as file:
            self.annotations = json.load(file)
            self.img_names = list(self.annotations.keys())
            self.img_names = self.img_names

        self.ignore_index = -1
        self.pad_box = pad_box


    def load_classes(self, reader):
        result = {}
        idx_to_result = []
        lines = reader.readlines()
        for line, row in enumerate(lines):
            row = row.strip()
            class_name = row
            class_id = line
            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            idx_to_result.append(class_name)

        return result, idx_to_result

    def get_num_verbs(self):
        return len(self.idx_to_verb)

    def get_num_nouns(self):
        return len(self.idx_to_noun)

    def get_num_roles(self):
        return len(self.idx_to_role)

    def get_noun_idx(self, noun):
        if noun == '':
            noun = 'blank'
        if noun not in self.idx_to_noun:
            noun = 'oov'
        return self.noun_to_idx[noun]

    def get_verb_roles(self):
        verb_roles = []
        verb_roles_dict = {}
        for verb in self.idx_to_verb:
            role_order = self.verb_to_role_order[verb]
            roles = [self.role_to_idx[r] for r in role_order]
            roles_key = '/'.join(role_order)
            if roles_key not in verb_roles_dict:
                verb_roles_dict[roles_key] = []
            verb_roles_dict[roles_key].append(verb)
            roles = roles + [self.get_num_roles()] * (6 - len(roles))
            verb_roles.append(roles)
        verb_roles = torch.tensor(verb_roles).long()

        return verb_roles

    def get_verb_role_order(self):
        verb_role_order = []
        for verb in self.idx_to_verb:
            role_order = self.verb_to_role_order[verb]
            role_order = [r.upper() for r in role_order]
            verb_role_order.append(role_order)
        return verb_role_order

    def get_verb_abstract(self):
        verb_abstract = []
        for verb in self.idx_to_verb:
            abs = self.verb_to_abs[verb]
            verb_abstract.append(abs)
        return verb_abstract

    def get_noun_words(self):
        noun_words = []
        for noun in self.idx_to_noun:
            if noun == 'blank':
                noun_words.append('')
            elif noun == 'oov':
                noun_words.append('[UNK]')
            else:
                noun_words.append(self.noun_to_word[noun])
        return noun_words

    def get_role_mask(self, pad_value=float('-inf')):
        role_mask = [[0] * self.get_num_roles() for _ in range(self.get_num_verbs())]
        for i,verb in enumerate(self.idx_to_verb):
            role_order = self.verb_to_role_order[verb]
            role_mask[i][len(role_order):] = [pad_value] * (6 - len(role_order))
        role_mask = torch.tensor(role_mask).float()
        return role_mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_anno = self.annotations[img_name]

        img = Image.open(self.img_folder / img_name).convert('RGB')
        w,h = img.size
        verb_name = img_anno['verb']
        role_order = self.verb_to_role_order[verb_name]
        box_roles = torch.tensor([(i, self.role_to_idx[r]) for i, r in enumerate(role_order)], dtype=torch.int64)
        roles = [self.role_to_idx[r] for r in role_order]

        boxes = [img_anno['bb'][r] for r in role_order]
        if self.img_set != 'train':
            boxes = boxes + [[-1, -1, -1, -1]] * (6 - len(boxes))
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        frames = img_anno['frames']
        noun1 = torch.ones((6)) * self.ignore_index
        noun2 = torch.ones((6)) * self.ignore_index
        noun3 = torch.ones((6)) * self.ignore_index
        for i, r_id in enumerate(roles):
            noun1[i] = self.get_noun_idx(frames[0][self.idx_to_role[r_id]])
            noun2[i] = self.get_noun_idx(frames[1][self.idx_to_role[r_id]])
            noun3[i] = self.get_noun_idx(frames[2][self.idx_to_role[r_id]])

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])

        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            box_roles = box_roles[keep]
            target['boxes'] = boxes
            target['labels'] = box_roles
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                try:
                    img, target = self._transforms(img, target)
                except:
                    img = self._transforms(img)

            target['labels'] = target['labels'][:, 0]

            if self.pad_box:
                t_boxes = torch.ones((6, 4)) * self.ignore_index
                for i,b_ind in enumerate(target['labels']):
                    t_boxes[b_ind] = target['boxes'][i]
                target['boxes'] = t_boxes

        else:
            img,_ = self._transforms(img, target={})
            target['boxes'] = boxes

        roles = roles + [self.ignore_index] * (6 - len(roles))
        roles = torch.tensor(roles, dtype=torch.int64)
        target['roles'] = roles
        target['verb'] = torch.as_tensor(self.verb_to_idx[verb_name]).long()
        noun1 = noun1.long()
        noun2 = noun2.long()
        noun3 = noun3.long()
        target['noun1'] = noun1
        target['noun2'] = noun2
        target['noun3'] = noun3
        target['img_name'] = img_name

        return img, target

def make_gsr_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                # T.RandomResize(scales, max_size=1333),
                T.RandomResize(scales, max_size=640),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=640),
                    # T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val' or image_set == 'dev':
        return T.Compose([
            # T.RandomResize([800], max_size=1333),
            T.RandomResize([600], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_sr_transforms(image_set):
    if image_set == 'train':
        return create_transform(
                640,#224,
                is_training=True,
                use_prefetcher=False,
                no_aug=False,
                scale=None,
                ratio=None,
                hflip=0.5,
                vflip=0.,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bilinear',
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                crop_pct=None,
                tf_preprocessing=False,
                re_prob=0.,
                re_mode='pixel',
                re_count=1,
                re_num_splits=0,
                separate=False,
            )

    if image_set == 'val':
        return create_transform(
                640, #224,
                is_training=False,
                use_prefetcher=False,
                no_aug=False,
                scale=None,
                ratio=None,
                hflip=0.5,
                vflip=0.,
                color_jitter=0.4,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bilinear',
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                crop_pct=None,
                tf_preprocessing=False,
                re_prob=0.,
                re_mode='pixel',
                re_count=1,
                re_num_splits=0,
                separate=False,
                 )

def build(image_set, args):
    root = Path(args.gsr_path)
    assert root.exists(), f'provided GSR path {root} does not exist'
    PATHS = {
        'train': (root / 'images_512', root / 'SWiG_jsons' / 'train.json'),
        'val': (root / 'images_512' , root / 'SWiG_jsons' / 'test.json'),
        'dev': (root / 'images_512', root / 'SWiG_jsons' / 'dev.json')
    }
    noun_list = root / 'global_utils' / 'noun_indices.txt'
    verb_list = root/ 'global_utils' / 'verb_indices.txt'
    role_list = root/ 'global_utils' / 'role_indices.txt'
    verb_info_file = root/ 'SWiG_jsons' / 'imsitu_space.json'

    img_folder, anno_file = PATHS[image_set]

    if args.img_aug:
        transforms = make_sr_transforms(image_set)
    else:
        transforms = make_gsr_transforms(image_set)

    dataset = GSRDetection(image_set, img_folder, anno_file, noun_list,verb_list,
                           role_list, verb_info_file, transforms=transforms, pad_box=args.pad_box)
    return dataset



















