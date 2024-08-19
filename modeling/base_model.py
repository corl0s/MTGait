import torch
import numpy as np
import os.path as osp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata
import torch.nn.functional as F

from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from abc import ABCMeta
from abc import abstractmethod

from data.dataset import DataSet
import data.sampler as Samplers
from data.collate_fn import CollateFn
from data.transform import get_transform

from .loss_aggregator import LossAggregator
from utils import get_valid_args, get_attr_from
from utils import is_list, is_dict, np2var, ts2np, list2var
from utils import Odict, NoOp, merge_odicts
from utils import get_msg_mgr, mkdir
from evaluation import evaluator as eval_functions
from .direction_estimation import DirectionEstimation
from .gait_estimation import GaitEstimation


torch.autograd.set_detect_anomaly(True)


class BaseModel(nn.Module):
    def __init__(self,cfgs,training):
        """Initialize the Training Paradigms.

        Complete the model initialization, including the data loader, the network, the optimizer, the scheduler, the loss.

        Args:
        cfgs:
            All of the configs.
        training:
            Whether the model is in training mode.
        """
        super(BaseModel, self).__init__()
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
        self.trainer_trfs = get_transform(cfgs['trainer_cfg']['transform'])
        
        self.direction_estimation = DirectionEstimation(cfgs,training)
        self.gait_estimation = GaitEstimation(cfgs,training)
        
        
        if training:
            self.train_loader = self.get_loader(
                cfgs['data_cfg'], train=True)
            
        if not training or self.engine_cfg['with_test']:
            self.test_loader = self.get_loader(
                cfgs['data_cfg'], train=False)
            self.evaluator_trfs = get_transform(
                cfgs['evaluator_cfg']['transform'])
            
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
        if training:
            self.loss_aggregator = LossAggregator(cfgs['loss_cfg'])
            self.optimizer = self.get_optimizer(self.cfgs['optimizer_cfg'])
            self.scheduler = self.get_scheduler(cfgs['scheduler_cfg'],self.optimizer)
        self.train(training)
        restore_hint = self.engine_cfg['restore_hint']
        if restore_hint != 0:
            self.resume_ckpt(restore_hint)

            
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

    def build_network(self, model_cfg):
        if 'backbone_cfg' in model_cfg.keys():
            self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
            
    def get_loader(self,data_cfg, train = True):
        sampler_cfg = self.cfgs['trainer_cfg']['sampler'] if train else self.cfgs['evaluator_cfg']['sampler']
        dataset = DataSet(data_cfg, train)
        
        Sampler = get_attr_from([Samplers],sampler_cfg['type'])
        valid_args = get_valid_args(Sampler, sampler_cfg, free_keys=['sampler_type','type'])
        sampler = Sampler(dataset, **valid_args)
        
        loader = tordata.DataLoader(
            dataset = dataset,
            batch_sampler = sampler,
            collate_fn = CollateFn(dataset.label_set,sampler_cfg),
            num_workers=data_cfg['num_workers']
        )
        
        return loader
    
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
    
    def save_ckpt(self, iteration):
        mkdir(osp.join(self.save_path, "checkpoints/"))
        save_name = self.engine_cfg['save_name']
        checkpoint = {
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iteration': iteration}
        torch.save(checkpoint,
                   osp.join(self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, iteration)))

    def _load_ckpt(self, save_name):
        load_ckpt_strict = self.engine_cfg['restore_ckpt_strict']
        
        map_location = f"{self.device}" if torch.cuda.is_available() else "cpu"

        # checkpoint = torch.load(save_name, map_location=torch.device(
        #     "cuda", self.device))
        checkpoint = torch.load(save_name, map_location=map_location)
        model_state_dict = checkpoint['model']

        
        filtered_state_dict = {}
        for key, value in model_state_dict.items():
            if not key.startswith('FCs.fc_bin') and not key.startswith('BNNecks.bn1d') and not key.startswith('BNNecks.fc_bin'):
                filtered_state_dict[key] = value
        
        if not load_ckpt_strict:
            self.msg_mgr.log_info("-------- Restored Params List --------")
            self.msg_mgr.log_info(sorted(set(model_state_dict.keys()).intersection(
                set(self.state_dict().keys()))))

        self.load_state_dict(filtered_state_dict, strict=load_ckpt_strict)
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
    
    
    def resume_ckpt(self, restore_hint):
        if isinstance(restore_hint, int):
            save_name = self.engine_cfg['save_name']
            save_name = osp.join(
                self.save_path, 'checkpoints/{}-{:0>5}.pt'.format(save_name, restore_hint))
            self.iteration = restore_hint
        elif isinstance(restore_hint, str):
            save_name = restore_hint
            self.iteration = 0
        else:
            raise ValueError(
                "Error type for -Restore_Hint-, supported: int or string.")
            
        self._load_ckpt(save_name)
        
    
    def fix_BN(self):
        for module in self.modules():
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                module.eval()
        
    # ---------------------------------------------------------------------------------------------------------
    # From here Training Procedure
    
    def inputs_pretreament(self, inputs):
        """Conduct transforms on input data.

        Args:
            inputs: the input data.
        Returns:
            tuple: training data including inputs, labels, and some meta data.
        """
        seqs_batch, labs_batch, typs_batch, vies_batch, seqL_batch = inputs
        seq_trfs = self.trainer_trfs if self.training else self.evaluator_trfs
        if len(seqs_batch) != len(seq_trfs):
            raise ValueError(
                "The number of types of input data and transform should be same. But got {} and {}".format(len(seqs_batch), len(seq_trfs)))
        requires_grad = bool(self.training)
        seqs = [np2var(np.asarray([trf(fra) for fra in seq]), requires_grad=requires_grad).float()
                for trf, seq in zip(seq_trfs, seqs_batch)]

        typs = typs_batch
        vies = [int(x) for x in vies_batch]
        vies = list2var(vies).float()

        labs = list2var(labs_batch).long()

        if seqL_batch is not None:
            seqL_batch = np2var(seqL_batch).int()
        seqL = seqL_batch

        if seqL is not None:
            seqL_sum = int(seqL.sum().data.cpu().numpy())
            ipts = [_[:, :seqL_sum] for _ in seqs]
        else:
            ipts = seqs
        del seqs
        return ipts, labs, typs, vies, seqL
    

    def inference(self, rank, type):
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
                labels = ipts[1]
                views = ipts[3]
                seqL = ipts[4]
                if type == 'recognition':
                    gait_retval = self.gait_estimation.forward(retval['extracted_features'],labels,seqL)
                    inference_feat = gait_retval['inference_feat']
                    del gait_retval
                if type == 'direction':
                    direction_retval = self.direction_estimation.forward(retval['extracted_features'],views, seqL)
                    inference_feat = direction_retval['inference_feat']
                    del direction_retval
                
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

    @ staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for inputs in model.train_loader:
            # print(len(inputs))
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                retval = model(ipts)
                visual_summary,features = retval['visual_summary'], retval ['extracted_features']
                labels = ipts[1]
                views = ipts[3]
                seqL = ipts[4]
                
                direction_retval = model.direction_estimation(features,views,seqL)
                direction_training_feat = direction_retval['training_feat']
                
                gait_retval = model.gait_estimation(features,labels,seqL)
                gait_training_feat = gait_retval['training_feat']
            
                del direction_retval
                del gait_retval
                del retval
            
            direction_loss, direction_loss_info = model.direction_estimation.direction_loss_aggregator(direction_training_feat)
            gait_loss, gait_info = model.gait_estimation.gait_loss_aggregator(gait_training_feat)
            
            total_loss = (1.0)*gait_loss + 1*(direction_loss)
            
            # Zero gradients for each optimizer
            model.direction_estimation.optimizer.zero_grad()
            model.gait_estimation.optimizer.zero_grad()
            model.optimizer.zero_grad()

            
            if model.engine_cfg['enable_float16']:
                model.Scaler.scale(total_loss).backward()

                model.Scaler.step(model.direction_estimation.optimizer)
                model.Scaler.step(model.gait_estimation.optimizer)
                model.Scaler.step(model.optimizer)

                scale = model.Scaler.get_scale()
                model.Scaler.update()

                #    Check if the optimizer step was skipped due to NaN gradients
                if scale != model.Scaler.get_scale():
                    model.msg_mgr.log_debug("Training step skipped. Expected the former scale to equal the present, got {} and {}".format(
                        scale, model.Scaler.get_scale()))
                    continue
            else:
                total_loss.backward()

                model.direction_estimation.optimizer.step()
                model.gait_estimation.optimizer.step()
                model.optimizer.step()
                
            model.iteration += 1
            model.direction_estimation.iteration += 1
            model.gait_estimation.iteration += 1
            model.scheduler.step()
            model.direction_estimation.scheduler.step()
            model.gait_estimation.scheduler.step()
            
        
            visual_summary.update(gait_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']
            visual_summary.update(direction_loss_info)
            if model.iteration % 100 == 0:
                print(f"Direction Loss: {direction_loss}, Gait Loss sum: {gait_loss}, Total Loss: {total_loss}")


            total_info = merge_odicts(gait_info,direction_loss_info)
            model.msg_mgr.train_step(total_info, visual_summary)
            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)
                model.direction_estimation.save_ckpt(model.iteration)
                model.gait_estimation.save_ckpt(model.iteration)
                # run test if with_test = true
                if model.engine_cfg['with_test']:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = BaseModel.run_test(model)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    if result_dict:
                        model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break
    
    @ staticmethod
    def run_test(model,type):
        """Accept the instance object(model) here, and then run the test loop."""
        evaluator_cfg = model.cfgs['evaluator_cfg']
        rank = 0
        with torch.no_grad():
            info_dict = model.inference(rank=0, type = type)
        if rank == 0:
            loader = model.test_loader
            label_list = loader.dataset.label_list
            types_list = loader.dataset.types_list
            views_list = loader.dataset.views_list
            class_labels = ['000', '018', '036', '054', '072', '090', '108', '126', '144', '162', '180']
            views_list = [class_labels.index(label) for label in views_list]
            # vies = [int(x) for x in views_list]
            # views_list = list2var(vies).long()

            info_dict.update({
                'labels': label_list, 'types': types_list, 'views': views_list})

            if 'eval_func' in evaluator_cfg.keys():
                eval_func = evaluator_cfg["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, evaluator_cfg, ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']
            return eval_func(info_dict, dataset_name, type,**valid_args)
        
  