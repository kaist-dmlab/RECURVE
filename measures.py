import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import PowerTransformer




def smooth_cos_dist(test_repr_long, dv_window_size, device):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    test_repr_long_torch = torch.Tensor(test_repr_long).to(device)
    test_repr_long_torch_pad = F.pad(test_repr_long_torch.transpose(0,1), (dv_window_size-1,dv_window_size), mode="replicate")
    test_repr_long_torch_pad = test_repr_long_torch_pad.transpose(0,1)

    cos_sim = cos(test_repr_long_torch[1:], test_repr_long_torch[:-1])
    cos_sim = torch.cat((cos_sim,cos_sim[-1:]),dim=0)
    cos_sim = cos_sim.cpu().numpy()

    cos_sim = (cos_sim - np.min(cos_sim))/(np.max(cos_sim)-np.min(cos_sim))

    cos_sim_pad = cos(test_repr_long_torch_pad[1:], test_repr_long_torch_pad[:-1])
    cos_sim_pad = torch.cat((cos_sim_pad,cos_sim_pad[-1:]),dim=0)
    cos_sim_pad = cos_sim_pad[None,None,:]
    
    mov_avg = F.conv1d(cos_sim_pad, torch.ones(1, 1, 2*dv_window_size).to(device), padding="valid", groups=1)
    mov_avg /= 2*dv_window_size 
    mov_avg = mov_avg.squeeze().cpu().numpy()

    mov_avg_scale = (mov_avg - np.min(mov_avg))/(np.max(mov_avg)-np.min(mov_avg))
    diff = cos_sim.squeeze() - mov_avg
    diff = np.abs(diff)
    diff = (diff - np.min(diff))/(np.max(diff)-np.min(diff))
    return 1-cos_sim, (1-mov_avg_scale)[:len(test_repr_long)], diff[:len(test_repr_long)]


def curvature_estimation(embs, q, device, w=10):
    # curvature = angle between two change vectors (CV(t-q,t) and CV(t,t+q))/ sum of CV norms (|CV(t-q,t)| + |CV(t,t+q)|)
    # CV(i,j) = a vector starting from z_i to z_j = z_j-z_i
    assert(w%2==0)
    embs = torch.tensor(embs, device=device)
    embs_pad_left = F.pad(embs.transpose(0,1), (q-1,0), mode="replicate")[None,:,:]
    embs_pad_right = F.pad(embs.transpose(0,1), (0,q-1), mode="replicate")[None,:,:]

    # Difference kernel that substract first timestamp vector from last timestamp vector
    kernel = torch.ones(embs.shape[1], 1, q)
    kernel[:,:,1:-1] = 0
    kernel[:,:,0] = -1

    cv_left= F.conv1d(
        embs_pad_left, 
        kernel.to(device), 
        padding="valid", 
        groups=embs.shape[1]
        ).transpose(1,2).squeeze()
    cv_right= F.conv1d(
        embs_pad_right, 
        kernel.to(device), 
        padding="valid", 
        groups=embs.shape[1]
        ).transpose(1,2).squeeze()
    cv_left_norm = torch.sqrt(torch.sum(cv_left*cv_left, dim=1))
    cv_right_norm = torch.sqrt(torch.sum(cv_right*cv_right, dim=1))

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(cv_left, cv_right)[:,None]
    cos_sim = torch.clamp(cos_sim, min=-1+1e-7,max=1-1e-7) # prevent arccos result nan
    angle = torch.acos(cos_sim).squeeze()
    
    
    
    curvature = (angle/(cv_left_norm+cv_right_norm+1e-6))[None,:]
    
    movavg = F.pad(curvature, (w-1,w), mode="replicate")[None,:,:]
    movavg = F.conv1d(movavg, torch.ones(1, 1, 2*w).to(device), padding="valid", groups=1)
    movavg /= 2*w
    movavg = torch.squeeze(movavg).cpu().numpy()
    movavg = np.max(movavg)-movavg # make CP higher than In-Segment
    movavg = (movavg - np.min(movavg))/(np.max(movavg)-np.min(movavg))

    
    curv = torch.squeeze(curvature).cpu().numpy()
    curv = np.max(curv)-curv # make CP higher than In-Segment
    curv = (curv - np.min(curv))/(np.max(curv)-np.min(curv))

    curv_reciprocal = ((cv_left_norm+cv_right_norm)/(angle+1e-6)).cpu().numpy()
    curv_reciprocal = (curv_reciprocal - np.min(curv_reciprocal))/(np.max(curv_reciprocal)-np.min(curv_reciprocal))


    return curv, curv_reciprocal, movavg
