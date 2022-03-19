# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import os
import argparse
from datetime import datetime
import random
import time
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from models.backbone import build_backbone
from models.transformer import build_transformer
from models.gsr import GSR
from timm.utils import *
from timm.loss import LabelSmoothingCrossEntropy
from collections import OrderedDict
from util.metrics import NounACC
from util.misc import is_dist_avail_and_initialized, get_world_size
from util import box_ops
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='training config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser = argparse.ArgumentParser(description='Verb Training and Evaluating')
parser.add_argument('--model', default='noun_model')
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--lr_drop', default=10, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')

# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")

# * Transformer
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=512, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--pre_norm', action='store_true')
# * Segmentation
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")

# GSR
parser.add_argument('--gsr', action='store_true',
                    help="Train for GSR if the flag is provided")
parser.add_argument('--extract', action='store_true',
                    help="extract feature")
parser.add_argument('--num_nouns', type=int, default=9928,
                    help="Number of object classes")
parser.add_argument('--num_verbs', type=int, default=504,
                    help="Number of verb classes")
parser.add_argument('--num_roles', type=int, default=190,
                    help="Number of verb classes")
parser.add_argument('--img_aug', action='store_true')
parser.add_argument('--pad_box', default=False, type=bool)

parser.add_argument('--loss_weight_enc_verb', default=1.0, type=float)
parser.add_argument('--loss_weight_dec_verb', default=1.0, type=float)
parser.add_argument('--loss_weight_dec_noun', default=1.0, type=float)

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
# * Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
parser.add_argument('--set_cost_obj_class', default=1, type=float,
                    help="Object class coefficient in the matching cost")
parser.add_argument('--set_cost_verb_class', default=1, type=float,
                    help="Verb class coefficient in the matching cost")

# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=3, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--obj_loss_coef', default=1, type=float)
parser.add_argument('--verb_loss_coef', default=1, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")

# dataset parameters
parser.add_argument('--dataset_file', default='swig')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--coco_panoptic_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')
parser.add_argument('--gsr_path', default='/storage/mwei/data/swig'
                    ,type=str)
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--output', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--port', default='2333')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    args, args_text = _parse_args()
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    args.rank = utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)
    backbone = build_backbone(args)

    transformer = build_transformer(args)
    model = GSR(
        backbone=backbone,
        transformer=transformer,
        num_nouns=dataset_train.get_num_nouns(),
        num_verbs=dataset_train.get_num_verbs(),
        num_roles=dataset_train.get_num_roles(),
        verb_roles=dataset_train.get_verb_roles(),
        role_mask=dataset_train.get_role_mask(pad_value=1),
        aux_loss=args.aux_loss,
        use_verb=True
    )

    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    if args.rank == 0:
        for name, param in model_without_ddp.named_parameters():
            print(name, param.size())

    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10*591,20*591], gamma=1 / args.lr_drop)


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    saver = None
    output_dir = None
    if args.rank == 0:
        output_base = args.output if args.output else './output'
        exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = get_outdir(output_base, 'train', exp_name)
        print(f'output dir is {output_dir}')
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args,
            checkpoint_dir=output_dir, recovery_dir=output_dir)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)


    print("Start training")
    train_loss_noun = nn.CrossEntropyLoss(ignore_index=-1)
    train_loss_verb = LabelSmoothingCrossEntropy(smoothing=0.2)
    best_metric = None
    best_epoch = None
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_metrics = train_epoch(epoch,
                                    model,
                                    data_loader_train,
                                    optimizer,
                                    train_loss_noun,
                                    train_loss_verb,
                                    args,
                                    lr_scheduler,
                                    saver,
                                    device,
                                    output_dir)
        eval_metrics = validate(model,
                                data_loader_val,
                                dataset_train.get_verb_roles(),
                                args,
                                device)
        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics['top1'])
    if best_metric is not None:
        print('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def loss_role_boxes(outputs, targets, num_boxes):
    assert 'pred_boxes' in outputs
    src_boxes = outputs['pred_boxes']
    target_roles = torch.cat([t['labels'] for t in targets])
    batch_idx = torch.cat([torch.full_like(t['labels'], i) for i, t in enumerate(targets)])
    src_boxes = src_boxes[batch_idx, target_roles].cuda()
    target_boxes = torch.cat([t['boxes'] for t in targets]).cuda()

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    loss_bbox = loss_bbox.sum() / num_boxes
    loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                       box_cxcywh_to_xyxy(target_boxes)))
    loss_giou = loss_giou.sum() / num_boxes

    return loss_bbox, loss_giou

def train_epoch(epoch,
                model,
                loader,
                optimizer,
                loss_fn_noun,
                loss_fn_verb,
                args,
                lr_scheduler=None,
                saver=None,
                device=None,
                output_dir=''):
    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()

    model.train()
    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)

    for batch_idx, (samples, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        samples = samples.to(device)
        gt_verb = torch.stack([t['verb'] for t in target]).cuda()

        gt_n1 = torch.stack([t['noun1'] for t in target]).cuda()
        gt_n2 = torch.stack([t['noun2'] for t in target]).cuda()
        gt_n3 = torch.stack([t['noun3'] for t in target]).cuda()

        out = model(samples, gt_verb)

        dec_noun = out['pred_obj_logits']

        num_boxes = sum(len(t["boxes"]) for t in target)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(out.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        loss_dec_noun = loss_fn_noun(dec_noun, gt_n1) + loss_fn_noun(dec_noun, gt_n2) + loss_fn_noun(dec_noun,
                                                                                                     gt_n3)

        loss_bbox, loss_giou = loss_role_boxes(out, target, num_boxes)

        if 'pred_verb_logits' in out:
            pred_verb = out['pred_verb_logits']
            loss_verb = loss_fn_verb(pred_verb, gt_verb)

            loss = args.loss_weight_dec_noun * loss_dec_noun + args.bbox_loss_coef * loss_bbox + args.giou_loss_coef * loss_giou + args.loss_weight_enc_verb * loss_verb
        else:
            loss = args.loss_weight_dec_noun * loss_dec_noun + args.bbox_loss_coef * loss_bbox + args.giou_loss_coef * loss_giou

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        num_updates += 1

        batch_time_m.update(time.time() - end)

        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
            else:
                reduced_loss = loss
            losses_m.update(reduced_loss.item(), dec_noun.size(0))

            if args.rank == 0:
                print(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '    
                    'Loss dec noun: {:>9.6f}  ' 
                    'Loss bbox: {:>9.6f}  ' 
                    'Loss giou: {:>9.6f}  ' 
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss_dec_noun.item(),
                        loss_bbox.item(),
                        loss_giou.item(),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=dec_noun.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=dec_noun.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step()

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(model,
             loader,
             verb_roles,
             args,
             device=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()

    noun_top1 = NounACC(1, args.distributed)
    noun_top5 = NounACC(5, args.distributed)

    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (samples, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            samples = samples.to(device)
            gt_verb = torch.stack([t['verb'] for t in target]).cuda()
            gt_roles = verb_roles[gt_verb]

            out = model(samples, gt_verb)

            dec_noun = out['pred_obj_logits']

            gt_n1 = torch.stack([t['noun1'] for t in target]).cuda()
            gt_n2 = torch.stack([t['noun2'] for t in target]).cuda()
            gt_n3 = torch.stack([t['noun3'] for t in target]).cuda()

            pred_boxes = out['pred_boxes']
            boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)
            target_sizes = torch.stack([t["orig_size"] for t in target], dim=0)
            img_h, img_w = target_sizes.unbind(1)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
            boxes = boxes * scale_fct[:, None, :].cuda()
            gt_boxes = torch.stack([t['boxes'] for t in target]).cuda()

            noun_top1.update(gt_verb,
                             gt_n1,
                             gt_n2,
                             gt_n3,
                             None,
                             dec_noun,
                             boxes,
                             gt_boxes)

            noun_top5.update(gt_verb,
                             gt_n1,
                             gt_n2,
                             gt_n3,
                             None,
                             dec_noun,
                             boxes,
                             gt_boxes)


    noun_top1_res = noun_top1.get_value()
    noun_top5_res = noun_top5.get_value()
    if args.rank == 0:
        print('[VAL-noun-top1] verb: %.7f value: %.7f value*: %.7f value_all: %.7f value_all*: %.7f '
              'grnd-value: %.7f grnd-value*: %.7f grnd-value_all: %.7f grnd-value_all*: %.7f' % (
                                                                        noun_top1_res['verb'],
                                                                        noun_top1_res['value'],
                                                                        noun_top1_res['value*'],
                                                                        noun_top1_res['value-all'],
                                                                        noun_top1_res['value-all*'],
                                                                        noun_top1_res['grnd-value'],
                                                                        noun_top1_res['grnd-value*'],
                                                                        noun_top1_res['grnd-value-all'],
                                                                        noun_top1_res['grnd-value-all*']))
        print('[VAL-noun-top5] verb: %.7f value: %.7f value*: %.7f value_all: %.7f value_all*: %.7f '
              'grnd-value: %.7f grnd-value*: %.7f grnd-value_all: %.7f grnd-value_all*: %.7f' % (
                  noun_top5_res['verb'],
                  noun_top5_res['value'],
                  noun_top5_res['value*'],
                  noun_top5_res['value-all'],
                  noun_top5_res['value-all*'],
                  noun_top5_res['grnd-value'],
                  noun_top5_res['grnd-value*'],
                  noun_top5_res['grnd-value-all'],
                  noun_top5_res['grnd-value-all*']))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', noun_top1_res['value*']), ('top5', noun_top5_res['value*'])])
    return metrics

if __name__ == '__main__':
    main()
