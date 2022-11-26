import os
import glob
from argparse import ArgumentParser
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from model import SPnet,SpatialTransformer
import datagenerators
import scipy.io as sio
        
def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric
    The default is to *not* return a measure for the background layer (label = 0)
    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.
    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)
    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
        
Frontal = [1000 + i for i in[3, 12, 14, 17, 18, 19, 20, 24, 27, 28]] + [2000+ j for j in[3, 12, 14, 17, 18, 19, 20, 24, 27, 28]]
Parietal = [1000 + i for i in [8, 22, 25, 29, 31]] + [2000 + j for j in [8, 22, 25, 29, 31]]
Occipital = [1000 + i for i in [5, 11, 13, 21]] + [2000 + j for j in [5, 11, 13, 21]]
Temporal = [1000 + i for i in [6, 7, 9, 15, 16, 30, 34]] + [2000 + i for i in [6, 7, 9, 15, 16, 30, 34]]
Cingulate = [1000 + i for i in [2, 10, 23, 26, 35]] + [2000 + i for i in [2, 10, 23, 26, 35]]

label_list = [1002, 1003, 1005, 1006, 1007, 1008, 1009, 1011,
       1012, 1013, 1014, 1015, 1016, 1017, 1018, 1021, 1022,
       1024, 1025, 1028, 1029, 1030, 1031, 1034, 1035, 2002,
       2003, 2005, 2006, 2007, 2008, 2009, 2011, 2012, 2013,
       2014, 2015, 2016, 2017, 2018, 2021, 2022, 2024, 2025,
       2028, 2029, 2030, 2031, 2034, 2035]

print(len(label_list))

new_fro = [label_list.index(x) for x in label_list if x in Frontal]
new_par = [label_list.index(x) for x in label_list if x in Parietal]
new_occ = [label_list.index(x) for x in label_list if x in Occipital]
new_tem = [label_list.index(x) for x in label_list if x in Temporal]
new_cin = [label_list.index(x) for x in label_list if x in Cingulate]



def test(gpu, test_path, init_model_file):
    """
    model training function
    :param gpu: integer specifying the gpu to use
    :param model: either vm1 or vm2 (based on CVPR 2018 paper)
    :param init_model_file: the model directory to load from
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = "cuda"
    vol_size = [160,192,160]

    brain_item = glob.glob(os.path.join(test_path,'image','*.npy'))
    brain_pair = [(x,y) for x in brain_item for y in brain_item if x!=y]
    model = SPnet(vol_size)
    model.to(device)
    model.load_state_dict(torch.load(init_model_file, map_location=lambda storage, loc: storage))

    trf = SpatialTransformer(vol_size, mode='nearest')
    trf.to(device)
    flow_trf = SpatialTransformer(vol_size)
    flow_trf.to(device)
    res = []
    res_jab = []
    res_diff = []

    for k in range(0, len(brain_pair)):

        print(k)
        moving_name, fixed_name = brain_pair[k]
        moving_seg, fixed_seg = os.path.join(test_path,'label',os.path.basename(moving_name).split('.')[0]+'_label.npy'),os.path.join(test_path,'label',os.path.basename(fixed_name).split('.')[0]+'_label.npy')
        moving_vol, moving_vol_seg = datagenerators.load_example_by_name(moving_name, moving_seg)
        fixed_vol, fixed_vol_seg = datagenerators.load_example_by_name(fixed_name, fixed_seg)

        input_moving  = torch.from_numpy(moving_vol).to(device).float()
        input_moving  = input_moving.permute(0, 4, 1, 2, 3)
        input_fixed  = torch.from_numpy(fixed_vol).to(device).float()
        input_fixed  = input_fixed.permute(0, 4, 1, 2, 3)

        with torch.no_grad():
            ([wraped,wraped1,wraped2,wraped3],[fixed,fixed1,fixed2,fixed3],
            [sym_wraped,sym_wraped1,sym_wraped2,sym_wraped3],[moving,moving1,moving2,moving3],
            [flow0,flow1,flow2,flow3],[sym_flow0,sym_flow1,sym_flow2,sym_flow3],
            [vec0,vec1,vec2,vec3]) = model(input_fixed,input_moving)

        # Warp segment using flow
        moving_seg = torch.from_numpy(moving_vol_seg).to(device).float()
        moving_seg = moving_seg.permute(0, 4, 1, 2, 3)
        fixed_seg = torch.from_numpy(fixed_vol_seg).to(device).float()
        fixed_seg = fixed_seg.permute(0, 4, 1, 2, 3)
        fixed_vol_seg = fixed_vol_seg.transpose(0,4,1,2,3)
        moving_vol_seg = moving_vol_seg.transpose(0,4,1,2,3)
        flow_diff = flow0 + flow_trf(sym_flow0,flow0)
        diff = flow_diff.abs().mean()
        res_diff.append(diff.item())
        warp_seg   = trf(moving_seg, flow0)
        vals, labels = dice(warp_seg, fixed_vol_seg,label_list,nargout=2)
        res.append(vals)


    print('final result')
    res = np.array(res)
    ave = res.mean(0)
    print(ave[new_fro].mean(),ave[new_par].mean(),ave[new_occ].mean(),ave[new_tem].mean(),ave[new_cin].mean())
    print(res_jab.mean(),res_diff.mean())

        #return
 

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--gpu",
                        type=str,
                        default='0',
                        help="gpu id")

    parser.add_argument("--test_path",
                        type=str,
                        dest="atlas_file",
                        default='./data/test',
                        help="gpu id number")

    parser.add_argument("--init_model_file", 
                        type=str,
                        default="./model/final_model.ckpt",
                        dest="init_model_file", 
                        help="model weight file")


    test(**vars(parser.parse_args()))

