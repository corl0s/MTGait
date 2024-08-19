import os
from time import strftime, localtime
import numpy as np
from utils import get_msg_mgr, mkdir
import torch
import torch.nn.functional as F

from .metric import cuda_dist
def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py

def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5 
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(view, probe_view) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(view, [gallery_view]) # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict



def single_id_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],}
    msg_mgr = get_msg_mgr()
    acc = {}
    # view_list = sorted(np.unique(view))
    label_list = sorted(np.unique(label))
    num_rank = 1
    # if dataset == 'CASIA-E':
    #     view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5 
    label_num = len(label_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((label_num, label_num, num_rank)) - 1.
        for (l1, probe_label) in enumerate(label_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                label, probe_label)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(label, probe_label) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            # probe_y = label[pseq_mask]

            probe_y = view[pseq_mask]
            # print(probe_y)
            for (l2, gallery_label) in enumerate(label_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    label, [gallery_label])   
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(label, [gallery_label]) # For SUSTech1K only
                # gallery_y = label[gseq_mask]
                gallery_y = view[gseq_mask]
                # print("Gallery_y", len(gallery_y))
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][l1, l2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            # sub_acc = acc[type_][:,:,rank]
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict



def evaluate_indoor_dataset(data, dataset, type = 'recognition', metric='euc'):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    # print(sorted(np.unique(view)))
    # print(len(label))

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if type == 'recognition':
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    if type == 'direction':
        return single_id_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)


