import torch
import torch.nn as nn

import os
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

# from .base_model import BaseModel
from .modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, conv1x1, conv3x3, BasicBlock2D, BasicBlockP3D, BasicBlock3D

from einops import rearrange

# from .loss_aggregator import DirectionLossAgrregator
from utils import get_valid_args, get_attr_from
from utils import is_list, is_dict, np2var, ts2np, list2var
# from utils import ddp_all_gather
from utils import Odict, NoOp
from utils import get_msg_mgr, mkdir
from evaluation import evaluator as eval_functions
from .loss_aggregator import LossAggregator


class DirectionEstimation(nn.Module):

    def __init__(self,cfgs,training):
        super(DirectionEstimation, self).__init__()
        self.msg_mgr = get_msg_mgr()
        self.cfgs = cfgs
        self.iteration = 0
        self.engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
        if self.engine_cfg is None:
            raise Exception("Initialize a model without -Engine-Cfgs-")

        if training and self.engine_cfg['enable_float16']:
            self.Scaler = GradScaler()
        self.save_path = osp.join('output/', cfgs['data_cfg']['dataset_name'],
                                  cfgs['model_cfg']['model'], self.engine_cfg['save_name'])
        
        self.build_network(cfgs['model_cfg'])
        self.init_parameters()
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
          
        if training:
            self.direction_loss_aggregator = LossAggregator(cfgs['direction_loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'],self.optimizer)
            
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer, optimizer_cfg, ['solver'])
        optimizer = optimizer(
            filter(lambda p: p.requires_grad, self.parameters()), **valid_arg)
        return optimizer
    
    
    def get_scheduler(self, scheduler_cfg,optimizer):
        self.msg_mgr.log_info(scheduler_cfg)
        Scheduler = get_attr_from(
            [optim.lr_scheduler], scheduler_cfg['scheduler'])
        valid_arg = get_valid_args(Scheduler, scheduler_cfg, ['scheduler'])
        scheduler = Scheduler(optimizer, **valid_arg)
        return scheduler
    
    
    def build_network(self, model_cfg):
        self.inference_use_emb2 = model_cfg['use_emb2'] if 'use_emb2' in model_cfg else False  
        channels    = model_cfg['Backbone']['channels']
        
        self.FCs = SeparateFCs(16, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(16, channels[2], class_num=11)

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=[16])

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def save_ckpt(self, iteration):
        mkdir(osp.join(self.save_path, "direction_checkpoints/"))
        save_name = self.engine_cfg['save_name']
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': iteration}
        torch.save(checkpoint,
                   osp.join(self.save_path, 'direction_checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))
        
    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'direction_checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")

        self._load_ckpt(save_name)
        
    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']
        
        map_location = f"{self.device}" if torch.cuda.is_available() else "cpu"

        # checkpoint = torch.load(save_name, map_location=torch.device(
        #     "cuda", self.device))
        checkpoint = torch.load(save_name, map_location=map_location)
        model_state_dict = checkpoint['model']

        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(model_state_dict, strict=load_ckpt_strict)
        if self.training:
            if not self.engine_cfg["optimizer_reset"] and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Optimizer from %s !!!" % save_name)
            if not self.engine_cfg["scheduler_reset"] and 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler'])
            else:
                self.msg_mgr.log_warning(
                    "Restore NO Scheduler from %s !!!" % save_name)
        self.msg_mgr.log_info("Restore Parameters from %s !!!" % save_name)
        
    
    def inference(self, rank):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        if rank == 0:
            pbar = tqdm(total=total_size, desc='Transforming')
        else:
            pbar = NoOp()
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts)
                inference_feat = retval['inference_feat']
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size
            if rest_size >= 0:
                update_size = batch_size
            else:
                update_size = total_size % batch_size
            pbar.update(update_size)
        pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v
        return info_dict


    def forward(self,features,vies,seqL):
        # Temporal Pooling, TP
        outs = self.TP(features, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]

        if self.inference_use_emb2:
                embed = embed_2
        else:
                embed = embed_1

        retval = {
            'training_feat': {
                # 'triplet': {'embeddings': embed_1, 'labels': vies}, # We don't use triplet loss for direction estimation task
                'softmax': {'logits': logits, 'labels': vies}
            },

            'inference_feat': {
                'embeddings': embed
            }
        }

        return retval
    
    
    