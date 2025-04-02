#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
from re import M
import yaml
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict
from pytorch_lightning.core.lightning import LightningModule

from models.loss import MOSLoss,SemanticLoss,AutomaticWeightedLoss
from models.metrics import ClassificationMetrics
from models.backbones_2d.mean_vfe import MeanVFE

from models.backbones_3d.instance_aware_backbone import Instance_Aware_Backbone
from models.backbones_3d.voxel_generate import VoxelGenerate

from spconv.pytorch.utils import gather_features_by_pc_voxel_id

class SegNet4D(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.poses = (
            self.hparams["DATA"]["POSES"].split(".")[0]
            if self.hparams["DATA"]["TRANSFORM"]
            else "no_poses"
        )
        self.cfg = hparams
        self.id = self.hparams["EXPERIMENT"]["ID"]
        self.dt_prediction = self.hparams["MODEL"]["DELTA_T_PREDICTION"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.n_past_steps = hparams["MODEL"]["N_PAST_STEPS"]
        self.batch_size = hparams["TRAIN"]["BATCH_SIZE"]

        self.point_cloud_range = np.array(self.cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        in_cahnnel = len(self.cfg["MODEL"]["POINT_FEATURE_ENCODING"]["src_feature_list"]) + self.n_past_steps - 1
        self.voxel_size = self.cfg["DATA"]['VOXEL_SIZE']  
        self.voxel_generate = VoxelGenerate(self.voxel_size,self.point_cloud_range,120000,5,in_cahnnel) 

        self.semantic_mos_config = yaml.safe_load(open(hparams["DATA"]["SEMANTIC_MOS_CONFIG_FILE"]))
        self.semantic_config = yaml.safe_load(open(hparams["DATA"]["SEMANTIC_CONFIG_FILE"]))
        self.semantic_config_all =yaml.safe_load(open(hparams["DATA"]["SEMANTIC_CONFIG_FILE_ALL"]))
        
        self.n_mos_classes = len(self.semantic_mos_config["learning_map_inv"])
        self.n_semantic_class = len(self.semantic_config["learning_map_inv"]) # not need unlabeled
        self.n_semantic_class_all = len(self.semantic_config_all["learning_map_inv"]) # not need unlabeled
        self.ignore_index = [
            key for key, ignore in self.semantic_mos_config["learning_ignore"].items() if ignore
        ]

        self.ignore_semantic_index = [
            key for key, ignore in self.semantic_config["learning_ignore"].items() if ignore
        ]
        self.ignore_semantic_index_all = [
            key for key, ignore in self.semantic_config_all["learning_ignore"].items() if ignore
        ]
        self.seg_num_per_class = hparams["TRAIN"]["SEG_NUM_PER_SEMANTIC_CLASS"]
        self.seg_num_per_class_all = hparams["TRAIN"]["SEG_NUM_PER_SEMANTIC_CLASS_ALL"]


        loss_param = 4
        self.multi_task_loss = AutomaticWeightedLoss(loss_param) 
        self.model = SegNet4D_Model(hparams, self.n_mos_classes,self.n_semantic_class,self.n_semantic_class_all,in_cahnnel)
        
        self.MOSLoss = MOSLoss(self.n_mos_classes, self.ignore_index)
        self.SemanticLoss  = SemanticLoss(self.n_semantic_class,self.seg_num_per_class,self.ignore_semantic_index)

        self.SemanticLoss_all  = SemanticLoss(self.n_semantic_class_all,self.seg_num_per_class_all,self.ignore_semantic_index_all)

        self.MOS_ClassificationMetrics = ClassificationMetrics(self.n_mos_classes, self.ignore_index)
        self.Semantic_ClassificationMetrics = ClassificationMetrics(self.n_semantic_class, self.ignore_semantic_index)
        self.Semantic_ClassificationMetrics_All = ClassificationMetrics(self.n_semantic_class_all, self.ignore_semantic_index_all)
        self.num_epoch_end = 0


    def forward(self, batch_data,Model_mode):
        out = self.model(batch_data,Model_mode)
        return out

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        Model_mode = 'train'
        
        batch = self.voxel_generate(batch)

        loss_rpn,train_loss_dict,batch_mos_feature,batch_semantic_feature,batch_semantic_feature_all = self.forward(batch,Model_mode)
        self.log("train_loss", loss_rpn.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("cls_loss", train_loss_dict["rpn_loss_cls"], on_step=True,batch_size=self.batch_size)
        self.log("box_loss", train_loss_dict["rpn_loss_loc"], on_step=True,batch_size=self.batch_size)

        loss =  torch.tensor([0.], device = batch["mos_labels"][0].device)
        loss_mos =  torch.tensor([0.], device = batch["mos_labels"][0].device)
        loss_semantic = torch.tensor([0.], device = batch["mos_labels"][0].device)
        loss_semantic_all = torch.tensor([0.], device = batch["mos_labels"][0].device)
  
        preb_mos_list = []
        preb_semantic_list = []
        preb_semantic_list_all = []
        for i in range(batch["batch_size"]):
            batch_mask = (batch['voxel_coords'][:,0]==i)
            mos_feature = gather_features_by_pc_voxel_id(batch_mos_feature[batch_mask], batch['pc_voxel_id'][i])
            semantic_feature = gather_features_by_pc_voxel_id(batch_semantic_feature[batch_mask],batch['pc_voxel_id'][i])
            semantic_feature_all = gather_features_by_pc_voxel_id(batch_semantic_feature_all[batch_mask],batch['pc_voxel_id'][i])

            loss_mos = loss_mos + self.MOSLoss.compute_loss(mos_feature, batch["mos_labels"][i])
            loss_semantic =  loss_semantic + self.SemanticLoss.compute_loss(semantic_feature, batch["semantic_labels"][i])
            loss_semantic_all = loss_semantic_all + self.SemanticLoss_all.compute_loss(semantic_feature_all,batch["semantic_labels_all"][i])

            preb_mos_list.append(mos_feature.detach().clone())
            preb_semantic_list.append(semantic_feature.detach().clone())
            preb_semantic_list_all.append(semantic_feature_all.detach().clone())


        loss = self.multi_task_loss(loss_rpn,loss_mos,loss_semantic,loss_semantic_all)

        gt_mos_label = torch.cat(batch["mos_labels"],dim=0)
        gt_semantic_label = torch.cat(batch["semantic_labels"],dim=0)
        gt_semantic_label_all = torch.cat(batch["semantic_labels_all"],dim=0)
        preb_mos = torch.cat(preb_mos_list,dim=0)
        preb_semantic = torch.cat(preb_semantic_list,dim=0)
        preb_semantic_all = torch.cat(preb_semantic_list_all,dim=0)
        
        self.log("loss", loss.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("mos_loss", loss_mos.item(), on_step=True,batch_size=self.batch_size)
        self.log("semantic_loss", loss_semantic.item(), on_step=True,batch_size=self.batch_size)
        self.log("semantic_loss_all", loss_semantic_all.item(), on_step=True,batch_size=self.batch_size)

        confusion_matrix = (
                self.get_step_confusion_matrix(preb_mos, gt_mos_label).detach().cpu()
            )

        confusion_semantic_matrix = (
            self.get_step_semantic_confusion_matrix(preb_semantic, gt_semantic_label).detach().cpu()
        )
        confusion_semantic_matrix_all = (
            self.get_step_semantic_confusion_matrix_all(preb_semantic_all, gt_semantic_label_all).detach().cpu()
        )
        torch.cuda.empty_cache()
        return {"loss": loss,"confusion_matrix": confusion_matrix,"confusion_semantic_matrix":confusion_semantic_matrix,"confusion_semantic_matrix_all":confusion_semantic_matrix_all}


    def training_epoch_end(self, training_step_outputs):
        list_dict_confusion_matrix = [
            output["confusion_matrix"] for output in training_step_outputs
        ]

        list_dict_confusion_semantic_matrix =[
            output["confusion_semantic_matrix"] for output in training_step_outputs
        ]

        list_dict_confusion_semantic_matrix_all =[
            output["confusion_semantic_matrix_all"] for output in training_step_outputs
        ]

        add_confusion_matrix = torch.zeros(self.n_mos_classes, self.n_mos_classes)
        for dict_confusion_matrix in list_dict_confusion_matrix:
            add_confusion_matrix = add_confusion_matrix.add(dict_confusion_matrix)
        iou = self.MOS_ClassificationMetrics.getIoU(add_confusion_matrix)

        add_confusion_semantic_matrix = torch.zeros(self.n_semantic_class, self.n_semantic_class)
        for dict_confusion_semantic_matrix in list_dict_confusion_semantic_matrix:
            add_confusion_semantic_matrix = add_confusion_semantic_matrix.add(dict_confusion_semantic_matrix)
        iou_semantic = self.Semantic_ClassificationMetrics.getIoU(add_confusion_semantic_matrix)

        iou_semantic_mean = torch.mean(iou_semantic[1:]) 

        add_confusion_semantic_all_matrix = torch.zeros(self.n_semantic_class_all, self.n_semantic_class_all)
        for dict_confusion_semantic_all_matrix in list_dict_confusion_semantic_matrix_all:
            add_confusion_semantic_all_matrix = add_confusion_semantic_all_matrix.add(dict_confusion_semantic_all_matrix)
        iou_semantic_all = self.Semantic_ClassificationMetrics_All.getIoU(add_confusion_semantic_all_matrix)

        iou_semantic_all_mean = torch.mean(iou_semantic_all[1:]) 

        self.log("train_mos_iou", iou[2].item(),batch_size=self.batch_size)
        self.log("train_semantic_iou", iou_semantic_mean.item(),batch_size=self.batch_size)
        self.log("train_semantic_all_iou", iou_semantic_all_mean.item(),batch_size=self.batch_size)
        torch.cuda.empty_cache()

    def validation_step(self, batch: tuple, batch_idx):
        Model_mode = 'eval'

        batch = self.voxel_generate(batch)

        batch_mos_feature, batch_semantic_feature,batch_semantic_feature_all, preb_dict_list,recall_dict = self.forward(batch,Model_mode)

        loss_mos = 0
        loss_semantic = 0
        loss_semantic_all = 0

        preb_mos_list = []
        preb_semantic_list = []
        preb_semantic_all_list = []
        for i in range(batch["batch_size"]):
            batch_mask = (batch['voxel_coords'][:,0]==i)
            mos_feature = gather_features_by_pc_voxel_id(batch_mos_feature[batch_mask], batch['pc_voxel_id'][i])
            semantic_feature = gather_features_by_pc_voxel_id(batch_semantic_feature[batch_mask],batch['pc_voxel_id'][i])
            semantic_feature_all = gather_features_by_pc_voxel_id(batch_semantic_feature_all[batch_mask],batch['pc_voxel_id'][i])

            loss_mos = loss_mos + self.MOSLoss.compute_loss(mos_feature, batch["mos_labels"][i])
            loss_semantic =  loss_semantic + self.SemanticLoss.compute_loss(semantic_feature, batch["semantic_labels"][i])
            loss_semantic_all =  loss_semantic_all + self.SemanticLoss_all.compute_loss(semantic_feature_all, batch["semantic_labels_all"][i])

            preb_mos_list.append(mos_feature.detach().clone())
            preb_semantic_list.append(semantic_feature.detach().clone())
            preb_semantic_all_list.append(semantic_feature_all.detach().clone())


        gt_mos_label = torch.cat(batch["mos_labels"],dim=0)
        gt_semantic_label = torch.cat(batch["semantic_labels"],dim=0)
        gt_semantic_all_label = torch.cat(batch["semantic_labels_all"],dim=0)
        preb_mos = torch.cat(preb_mos_list,dim=0)
        preb_semantic = torch.cat(preb_semantic_list,dim=0)
        preb_semantic_all = torch.cat(preb_semantic_all_list,dim=0)

        metric  = {
            'batch_gt_num': 0,
        }


        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            metric['batch_recall_roi_%s' % str(cur_thresh)] = 0
            metric['batch_recall_rcnn_%s' % str(cur_thresh)] = 0

        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            metric['batch_recall_roi_%s' % str(cur_thresh)] += recall_dict.get('roi_%s' % str(cur_thresh), 0)
            metric['batch_recall_rcnn_%s' % str(cur_thresh)] += recall_dict.get('rcnn_%s' % str(cur_thresh), 0)
        metric['batch_gt_num'] += recall_dict.get('gt',0)

        confusion_matrix = (
                self.get_step_confusion_matrix(preb_mos, gt_mos_label).detach().cpu()
            )
        confusion_semantic_matrix = (
            self.get_step_semantic_confusion_matrix(preb_semantic, gt_semantic_label).detach().cpu()
        )
        confusion_semantic_all_matrix = (
            self.get_step_semantic_confusion_matrix_all(preb_semantic_all, gt_semantic_all_label).detach().cpu()
        )
        metric["confusion_matrix"] = confusion_matrix
        metric["confusion_semantic_matrix"] = confusion_semantic_matrix
        metric["confusion_semantic_all_matrix"] = confusion_semantic_all_matrix

        self.log("val_mos_loss", loss_mos.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("val_semantic_loss", loss_semantic.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)
        self.log("val_semantic_all_loss", loss_semantic_all.item(), on_step=True,on_epoch=True,batch_size=self.batch_size)

        torch.cuda.empty_cache()
        return metric

    def validation_epoch_end(self, validation_step_outputs):
        val_metric ={
            'gt_num': 0,
        }
        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            val_metric['recall_roi_%s' % str(cur_thresh)] = 0
            val_metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        add_confusion_matrix = torch.zeros(self.n_mos_classes, self.n_mos_classes)
        add_confusion_semantic_matrix = torch.zeros(self.n_semantic_class, self.n_semantic_class)
        add_confusion_semantic_all_matrix = torch.zeros(self.n_semantic_class_all, self.n_semantic_class_all)
        for batch_metric in validation_step_outputs:
            for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
                val_metric['recall_roi_%s' % str(cur_thresh)] += batch_metric.get('batch_recall_roi_%s' % str(cur_thresh), 0)
                val_metric['recall_rcnn_%s' % str(cur_thresh)] += batch_metric.get('batch_recall_rcnn_%s' % str(cur_thresh), 0)
            val_metric['gt_num'] += batch_metric.get('batch_gt_num', 0)

            add_confusion_matrix = add_confusion_matrix.add(batch_metric["confusion_matrix"])
            add_confusion_semantic_matrix = add_confusion_semantic_matrix.add(batch_metric["confusion_semantic_matrix"])
            add_confusion_semantic_all_matrix = add_confusion_semantic_all_matrix.add(batch_metric["confusion_semantic_all_matrix"])

        iou = self.MOS_ClassificationMetrics.getIoU(add_confusion_matrix)
        iou_semantic = self.Semantic_ClassificationMetrics.getIoU(add_confusion_semantic_matrix)
        iou_semantic_all = self.Semantic_ClassificationMetrics_All.getIoU(add_confusion_semantic_all_matrix)
        
        iou_semantic_mean = torch.mean(iou_semantic[1:]) 
        iou_semantic_all_mean = torch.mean(iou_semantic_all[1:]) 
        self.log("val_mos_iou", iou[2].item(),batch_size=self.batch_size)
        self.log("val_semantic_mean_iou", iou_semantic_mean.item(),batch_size=self.batch_size)
        self.log("val_semantic_all_mean_iou", iou_semantic_all_mean.item(),batch_size=self.batch_size)
        

        gt_num_cnt = val_metric['gt_num']
        for cur_thresh in self.hparams["MODEL"]["POST_PROCESSING"]["RECALL_THRESH_LIST"]:
            cur_roi_recall = val_metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            cur_rcnn_recall = val_metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
            self.log('recall_roi_%s' % str(int(cur_thresh*10)), cur_roi_recall,batch_size=self.batch_size)
            self.log('recall_rcnn_%s' % str(int(cur_thresh*10)), cur_rcnn_recall,batch_size=self.batch_size)

        torch.cuda.empty_cache()

    def get_step_confusion_matrix(self, out, past_labels):
        confusion_matrix = self.MOS_ClassificationMetrics.compute_confusion_matrix(
            out, past_labels
        )
        return confusion_matrix

    def get_step_semantic_confusion_matrix(self, out, past_labels):
        confusion_matrix = self.Semantic_ClassificationMetrics.compute_confusion_matrix(
            out, past_labels
        )
        return confusion_matrix

    def get_step_semantic_confusion_matrix_all(self, out, past_labels):
        confusion_matrix = self.Semantic_ClassificationMetrics_All.compute_confusion_matrix(
            out, past_labels
        )
        return confusion_matrix

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False , self.parameters()), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay  #主要调这个控制学习率衰减
        )
        return [optimizer], [scheduler]
    

#######################################
# Modules
#######################################


class SegNet4D_Model(nn.Module):
    def __init__(self, cfg: dict, n_mos_classes: int, n_semantic_class: int, n_semantic_class_all: int, channel_dim):
        super().__init__()

        self.dt_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.post_process = cfg["MODEL"]["POST_PROCESSING"]
        self.num_class = cfg["MODEL"]["DENSE_HEAD"]["NUM_CLASS"]

        self.point_cloud_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"]) 
        self.voxel_size = cfg["DATA"]['VOXEL_SIZE']  
        self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size) 
        self.grid_size=np.round(self.grid_size).astype(np.int64) 

        self.n_past_step = cfg["MODEL"]["N_PAST_STEPS"]
        self.mos_class = n_mos_classes
        self.semantic_class = n_semantic_class
        self.multi_semantic_class = n_semantic_class_all

        self.vfe = MeanVFE(cfg["MODEL"]["VFE"],channel_dim)
        self.unet = Instance_Aware_Backbone(cfg,channel_dim,self.grid_size,self.voxel_size,self.point_cloud_range,self.mos_class,self.semantic_class,self.multi_semantic_class)
    
        
    def forward(self, batch_dict,Model_mode):

        batch_dict = self.vfe(batch_dict)
        if Model_mode =='train':
            (loss_rpn, tb_dict),point_seg_feature, semantic_seg_feature,semantic_seg_feature_all = self.unet(batch_dict,Model_mode)
            return loss_rpn, tb_dict, point_seg_feature, semantic_seg_feature,semantic_seg_feature_all
        else:
            point_seg_feature,semantic_seg_feature,semantic_seg_feature_all, pred_dicts, recall_dicts= self.unet(batch_dict,Model_mode)
            return point_seg_feature,semantic_seg_feature,semantic_seg_feature_all, pred_dicts, recall_dicts
