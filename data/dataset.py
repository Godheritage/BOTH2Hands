import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
from tqdm import tqdm
import pytorch3d.transforms as transforms 
from human_body_prior.body_model.body_model import BodyModel
import joblib
from scipy.spatial.transform import Rotation as R
import clip
import torch.nn as nn

SMPLH_PATH = "body_model/smpl_models/smplh"

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def qbetween_np(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v0 = torch.from_numpy(v0).float()
    v1 = torch.from_numpy(v1).float()
    return qbetween(v0, v1).numpy()
def qbetween(v0, v1):
    '''
    find the quaternion used to rotate v0 to v1
    '''
    assert v0.shape[-1] == 3, 'v0 must be of the shape (*, 3)'
    assert v1.shape[-1] == 3, 'v1 must be of the shape (*, 3)'

    v = torch.cross(v0, v1)
    w = torch.sqrt((v0 ** 2).sum(dim=-1, keepdim=True) * (v1 ** 2).sum(dim=-1, keepdim=True)) + (v0 * v1).sum(dim=-1,
                                                                                                              keepdim=True)
    return qnormalize(torch.cat([w, v], dim=-1))

def qnormalize(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    return q / torch.norm(q, dim=-1, keepdim=True)


def rt_normalizer_cut(ori_root_orient,ori_root_trans):
    ori_root_r = R.from_rotvec(ori_root_orient)
    target_root_r = R.from_rotvec(ori_root_orient[0])

    xz_rot_vec = target_root_r.as_rotvec() * np.array([0, 1, 0])

    inv_xz_rot_vec = -xz_rot_vec

    yz_inv_rot_vec = np.array([0, inv_xz_rot_vec[1], 0])
    yz_inv_root_r = R.from_rotvec(yz_inv_rot_vec)

    processed_root_r = target_root_r * yz_inv_root_r

    target_root_r = processed_root_r
    transform_r = (ori_root_r[0].inv()*target_root_r) # T X 3 X 3 
    curr_root_r = (transform_r*ori_root_r)
    curr_root_rotvec = curr_root_r.as_rotvec().astype(np.float32)
    velocities = ori_root_trans[1:] - ori_root_trans[:-1]  # calculates velocity T-1 X 3 
    velocities_xz = velocities.copy()
    velocities_xz[:,1] = 0
    # compute position in target orientation
    curr_root_trans = np.zeros_like(ori_root_trans)
    for i in range(1, len(ori_root_trans)):  
        rotated_forward = transform_r.apply(velocities_xz[i-1])  
        curr_root_trans[i] = curr_root_trans[i-1] + rotated_forward
    curr_root_trans[:,1] = ori_root_trans[:,1]
    
    return curr_root_rotvec,curr_root_trans

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat

def run_smpl_model(root_trans, aa_rot_rep, betas, gender, bm_dict):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female']
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 
    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 

def get_smpl_parents():
    bm_path = pjoin(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    # parents = ori_kintree_table[0, :22] # 22 
    parents = ori_kintree_table[0] # 52 
    parents[0] = -1 # Assign -1 for the root joint's parent idx.

    return parents

def prepare_smplh_model(device):
    surface_model_male_fname = pjoin(SMPLH_PATH, "male", 'model.npz')
    surface_model_female_fname = pjoin(SMPLH_PATH, "female", 'model.npz')
    
    dmpl_fname = None
    num_dmpls = None 
    num_expressions = None
    num_betas = 16
    
    male_bm = BodyModel(bm_fname=surface_model_male_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)
    female_bm = BodyModel(bm_fname=surface_model_female_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname)
    
    for p in male_bm.parameters():
        p.requires_grad = False
    for p in female_bm.parameters():
        p.requires_grad = False 

    male_bm = male_bm.cuda(device)
    female_bm = female_bm.cuda(device)
    
    bm_dict = {'male' : male_bm, 'female' : female_bm}
    return bm_dict

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 
    # try:
    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])
    return global_pose # T X J X 3 X 3 

def quat_fk_torch(lrot_mat, lpos):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res
    

class BodyHandDataset(data.Dataset):
    def __init__(
        self,
        opt,
        datasets_to_process=[],
    ):
        
        self.opt = opt 
        self.frame_rate = opt.frame_rate
        self.window = opt.window
        self.datasets_to_process = datasets_to_process
        self.havecleanhand=True
        if not opt.have_hand:
            self.havecleanhand=False
        self.output_mode = opt.output_mode
        self.data_list = []
        self.max_length = 0
        self.data_root_folder = opt.data_root_folder
        
        
        print("Preparing text encoder ")
        self.init_text_encoder()
        
        
        print("Initalizing SMPLH ")
        self.bm_dict = prepare_smplh_model(self.opt.device)
        self.rest_human_offsets = self.get_rest_pose_joints()
        
        
        
        print("Loading data ")
        self.load_or_process_dataset()
        
        
    def init_text_encoder(self):
        self.clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        set_requires_grad(self.clip_model, False)
        self.clip_training = "text_"
        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=4)
        self.text_ln = nn.LayerNorm(768)
        
    def load_or_process_dataset(self):
        proceed_data_str = "bodyhand_processed_dataset_"
        meanstd_str = "bodyhand_processed_dataset_meanstd_"
        
        for dataset in self.datasets_to_process:
            raw_data_path = pjoin(self.data_root_folder,dataset, "bodyhand_raw_dataset_"+str(dataset)+".p")
            proceed_data_str +=(str(dataset)+"_")
            meanstd_str +=(str(dataset)+"_")
            
            normalize_data_path = pjoin(self.data_root_folder,proceed_data_str +str(self.window)+".p")
            meanstd_data_path = pjoin(self.data_root_folder, meanstd_str+str(self.window)+".p")
            
            if(not os.path.exists(normalize_data_path)):
                print("Loading raw "+dataset)
                self.data_list.extend(joblib.load(raw_data_path))
                self.max_length = max(motion["seq_len"] for motion in self.data_list)
                
                print("Calculating normalized data ")
                if self.havecleanhand:
                    self.cal_normalize_data_input_withhand() 
                else:
                    self.cal_normalize_data_input_nohand()
                joblib.dump(self.window_data_dict, normalize_data_path)   
            else:
                print("Loading normalized data ")
                self.window_data_dict = joblib.load(normalize_data_path)
    
            if(not os.path.exists(meanstd_data_path)):    
                print("Calculating mean std ")
                self.min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(self.min_max_mean_std_jpos_data, meanstd_data_path)
            else:
                print("Loading mean std ")
                self.min_max_mean_std_jpos_data = joblib.load(meanstd_data_path)
        
        self.global_jpos_min = torch.from_numpy(self.min_max_mean_std_jpos_data['global_jpos_min']).float().reshape(52, 3)[None]
        self.global_jpos_max = torch.from_numpy(self.min_max_mean_std_jpos_data['global_jpos_max']).float().reshape(52, 3)[None]
        self.global_jvel_min = torch.from_numpy(self.min_max_mean_std_jpos_data['global_jvel_min']).float().reshape(52, 3)[None]
        self.global_jvel_max = torch.from_numpy(self.min_max_mean_std_jpos_data['global_jvel_max']).float().reshape(52, 3)[None]
        
        if(self.output_mode=="Body"):
            self.num_joints = 22
        elif(self.output_mode=="Hand"):
            self.num_joints = 30
        else: self.num_joints = 52
    
    def encode_text(self,raw_text,device):
        ##clip
        text = clip.tokenize(raw_text, truncate=True).to(device)
        x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)

        x = self.textTransEncoder(x)
        x = self.text_ln(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].unsqueeze(1)
        
        return x
    
    
    def cal_normalize_data_input_nohand(self):
        
        self.window_data_dict = {}
        s_idx = 0 
        print("Precessing dataset without hand")
        for motion_dict in tqdm(self.data_list):
            
            seq_root_trans = motion_dict['trans'].reshape(-1, 3) # T X 3 
            seq_root_orient = motion_dict['root_orient'].reshape(-1, 3) # T X 3 
            seq_pose_body = motion_dict['body_pose'].reshape(-1, 21, 3) # T X 21 X 3
            seq_pose_hand = motion_dict['hand_pose'].reshape(-1, 30, 3) # T X 30 X 3
            seq_pose_all = np.concatenate((seq_pose_body,seq_pose_hand),axis=1)

            num_steps = seq_root_trans.shape[0]
            
            for start_t_idx in range(0, num_steps, self.window):
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps 

                # Skip the segment that has a length < window 
                if end_t_idx - start_t_idx +1 < self.window:
                    continue 

                query = self.process_window_data(seq_root_trans, seq_root_orient, seq_pose_all, start_t_idx, end_t_idx,need_fix_height=True,need_fix_initface=self.opt.need_fix_initface)
                    
                self.window_data_dict[s_idx] = {}
                curr_global_jpos = query['global_jpos']
                curr_global_jvel = query['global_jvel']
                curr_global_rot_6d = query['global_rot_6d']

                self.window_data_dict[s_idx]['start_t_idx'] = start_t_idx
                self.window_data_dict[s_idx]['end_t_idx'] = end_t_idx 

                self.window_data_dict[s_idx]['global_jpos'] = curr_global_jpos.reshape(-1, 156).detach().cpu().numpy()
                self.window_data_dict[s_idx]['global_jvel'] = curr_global_jvel.reshape(-1, 156).detach().cpu().numpy()
                self.window_data_dict[s_idx]['global_rot_6d'] = curr_global_rot_6d.reshape(-1, 52*6).detach().cpu().numpy()
                
                self.window_data_dict[s_idx]['body_cond'] = motion_dict['body_cond']
                self.window_data_dict[s_idx]['hand_cond'] = None
                self.window_data_dict[s_idx]['data_path'] = motion_dict['data_path']
                
                s_idx += 1 
   
    def cal_normalize_data_input_withhand(self):
        self.window_data_dict = {}
        s_idx = 0 
        print("Processing dataset with hand")
        
        for motion_dict in tqdm(self.data_list):
            seq_root_trans = motion_dict['trans'].reshape(-1, 3) # T X 3 
            seq_root_orient = motion_dict['root_orient'].reshape(-1, 3) # T X 3 
            seq_pose_body = motion_dict['body_pose'].reshape(-1, 21, 3) # T X 21 X 3
            seq_pose_hand = motion_dict['hand_pose'].reshape(-1, 30, 3) # T X 30 X 3
            seq_pose_all = np.concatenate((seq_pose_body,seq_pose_hand),axis=1)

            motion_end_index = seq_root_trans.shape[0]
            
            if motion_end_index < self.window:
                continue
            motion_end_index = motion_end_index-1
            
            query = self.process_window_data(seq_root_trans, seq_root_orient, seq_pose_all, 0, motion_end_index,need_local_hand=True,need_fix_height=True,need_fix_initface=self.opt.need_fix_initface)
            
            
            self.window_data_dict[s_idx] = {}
                
            curr_global_jpos = query['global_jpos']
            curr_global_jvel = query['global_jvel']
            curr_global_rot_6d = query['global_rot_6d']

            self.window_data_dict[s_idx]['start_t_idx'] = 0
            self.window_data_dict[s_idx]['end_t_idx'] = motion_end_index 

            self.window_data_dict[s_idx]['global_jpos'] = curr_global_jpos.reshape(-1, 156).detach().cpu().numpy()
            self.window_data_dict[s_idx]['global_jvel'] = curr_global_jvel.reshape(-1, 156).detach().cpu().numpy()
            self.window_data_dict[s_idx]['global_rot_6d'] = curr_global_rot_6d.reshape(-1, 52*6).detach().cpu().numpy()
            
            self.window_data_dict[s_idx]['body_cond'] = self.encode_text(motion_dict['body_cond'],'cpu') 
            if motion_dict['hand_cond'] !=[]:
                self.window_data_dict[s_idx]['raw_text'] = motion_dict['hand_cond']
                self.window_data_dict[s_idx]['hand_cond'] = self.encode_text(motion_dict['hand_cond'],'cpu')
            else:
                self.window_data_dict[s_idx]['raw_text'] = motion_dict['body_cond']
                self.window_data_dict[s_idx]['hand_cond'] = None
            self.window_data_dict[s_idx]['data_path'] = motion_dict['data_path']
            
            
            self.window_data_dict[s_idx]['clean_wist_hand_jpos'] = query['clean_wist_hand_jpos'].reshape(-1, 90).detach().cpu().numpy()
            self.window_data_dict[s_idx]['clean_wist_hand_rot_6d'] = query['clean_wist_hand_rot_6d'].reshape(-1, 180).detach().cpu().numpy()
            
            
            s_idx += 1   
         
    def process_window_data(self, seq_root_trans, seq_root_orient, seq_pose_all, random_t_idx, end_t_idx,need_fix_height=False,need_fix_initface = False,need_local_hand=False):
        
        window_pose_all  = torch.from_numpy(seq_pose_all[random_t_idx:end_t_idx+1]).float().cuda(self.opt.device)
        
        if need_fix_initface:
            window_root_orient,window_root_trans = rt_normalizer_cut(seq_root_orient[random_t_idx:end_t_idx+1],seq_root_trans[random_t_idx:end_t_idx+1])
            window_root_trans = torch.from_numpy(window_root_trans).float().cuda(self.opt.device)
            window_root_orient = torch.from_numpy(window_root_orient).float().cuda(self.opt.device)
        else:
            window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).float().cuda(self.opt.device)
            window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda(self.opt.device)
        
        
        curr_seq_pose_all_aa = torch.cat((window_root_orient[:, None, :], window_pose_all), dim=1) # T' X 52 X 3 
        curr_seq_local_jpos_all = self.rest_human_offsets.repeat(curr_seq_pose_all_aa.shape[0], 1, 1) # T' X 52 X 3 
        
        curr_seq_pose_rot_mat_all = transforms.axis_angle_to_matrix(curr_seq_pose_all_aa)
        _, human_jnts = quat_fk_torch(curr_seq_pose_rot_mat_all, curr_seq_local_jpos_all)
        
        
        if need_fix_height:
            for i in range(0,window_root_trans.shape[0]):
                if window_root_trans[i,1] > 0:
                    window_root_trans[i,1] = 0
        
        
        global_jpos = human_jnts + window_root_trans[:, None, :] # T' X 52 X 3  
        global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 52 X 3 
        
        local_joint_rot_mat_all = transforms.axis_angle_to_matrix(curr_seq_pose_all_aa) # T' X 52 X 3 X 3 
        global_joint_rot_mat_all = local2global_pose(local_joint_rot_mat_all) # T' X 52 X 3 X 3 

        global_rot_6d_all = transforms.matrix_to_rotation_6d(global_joint_rot_mat_all)
        
        query = {}
        
        query['global_jpos'] = global_jpos # T' X 52 X 3 
        query['global_jvel'] = torch.cat((global_jvel, \
                torch.zeros(1, 52, 3).to(global_jvel.device)), dim=0) # T' X 52 X 3
        query['global_rot_6d'] = global_rot_6d_all # T' X 52 X 6 
        
        if need_local_hand:
            pose_all_aa_forlocalhand = curr_seq_pose_all_aa.clone()
            pose_all_aa_forlocalhand[:,0:22]=0
            local_jpos_all_forlocalhand = self.rest_human_offsets.repeat(pose_all_aa_forlocalhand.shape[0], 1, 1)
            pose_rot_mat_all_forlocalhand = transforms.axis_angle_to_matrix(pose_all_aa_forlocalhand)
            _, human_jnts_forlocalhand = quat_fk_torch(pose_rot_mat_all_forlocalhand, local_jpos_all_forlocalhand)
            left_hand_jnts = (human_jnts_forlocalhand[:,22:37]-human_jnts_forlocalhand[:,20:21])
            right_hand_jnts = (human_jnts_forlocalhand[:,37:]-human_jnts_forlocalhand[:,21:22])
            hand_jnts = torch.cat((left_hand_jnts,right_hand_jnts),dim=1)
            query['clean_wist_hand_jpos'] = hand_jnts
            
            local_hand_rot_mat_all = transforms.axis_angle_to_matrix(pose_all_aa_forlocalhand) # T' X 52 X 3 X 3 
            global_hand_rot_mat_all = local2global_pose(local_hand_rot_mat_all)[:,22:] # T' X 22 X 3 X 3 
            global_hand_rot_6d_all = transforms.matrix_to_rotation_6d(global_hand_rot_mat_all)
            query['clean_wist_hand_rot_6d'] = global_hand_rot_6d_all
        else:
            query['clean_wist_hand_jpos'] = torch.zeros((150,30,3))
            query['clean_wist_hand_rot_6d'] = torch.zeros((150,30,6))
        
        
        return query 

    def get_rest_pose_joints(self):
        zero_root_trans = torch.zeros(1, 1, 3).cuda(self.opt.device).float()
        zero_rot_aa_rep = torch.zeros(1, 1, 52, 3).cuda(self.opt.device).float()
        bs = 1 
        betas = torch.zeros(1, 16).cuda(self.opt.device).float()
        gender = ["male"] * bs 

        rest_human_jnts, _, _ = \
        run_smpl_model(zero_root_trans, zero_rot_aa_rep, betas, gender, self.bm_dict)
        # 1 X 1 X J X 3 

        parents = get_smpl_parents()
        parents[0] = 0 # Make root joint's parent itself so that after deduction, the root offsets are 0
        rest_human_offsets = rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]

        return rest_human_offsets # 1 X J X 3 
    
    def fk_smpl(self, root_trans, lrot_aa):
        # root_trans: N X 3 
        # lrot_aa: N X J X 3 

        # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
        # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
        
        parents = get_smpl_parents() 

        lrot_mat = transforms.axis_angle_to_matrix(lrot_aa) # N X J X 3 X 3 

        lrot = transforms.matrix_to_quaternion(lrot_mat)

        # Generate global joint position 
        lpos = self.rest_human_offsets.repeat(lrot_mat.shape[0], 1, 1) # T' X 22 X 3 

        gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
        for i in range(1, len(parents)):
            gp.append(
                transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
            )
            gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

        global_rot = torch.cat(gr, dim=-2) # T X 22 X 4 
        global_jpos = torch.cat(gp, dim=-2) # T X 22 X 3 

        global_jpos += root_trans[:, None, :] # T X 22 X 3

        return global_rot, global_jpos 
    
    def de_normalize_jpos_min_max(self, normalized_jpos):
        normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

        min_val = self.global_jpos_min.to(normalized_jpos.device)
        max_val = self.global_jpos_max.to(normalized_jpos.device)

        diff = max_val - min_val
        diff[diff == 0] = 1e-8  # set very small value to prevent division by zero

        de_jpos = normalized_jpos * diff + min_val
        return de_jpos # T X 22 X 3
    
    def normalize_jpos_min_max(self, ori_jpos):
        
        # ori_jpos: T X 22 X 3 
        min_val = self.global_jpos_min.to(ori_jpos.device)
        max_val = self.global_jpos_max.to(ori_jpos.device)
        
        diff = max_val - min_val
        # Handle the case when min == max
        diff[diff == 0] = 1e-8  # set very small value to prevent division by zero
        
        normalized_jpos = (ori_jpos - min_val) / diff
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 
        return normalized_jpos # T X 22 X 3
    
    def normalize_jpos_min_max_hand(self, ori_jpos):
        
        # ori_jpos: T X 22 X 3 
        min_val = self.global_jpos_min.to(ori_jpos.device)
        max_val = self.global_jpos_max.to(ori_jpos.device)
        
        diff = max_val - min_val
        # Handle the case when min == max
        diff[diff == 0] = 1e-8  # set very small value to prevent division by zero
        
        min_val = min_val[:,22:]
        diff = diff[:,22:]
        
        normalized_jpos = (ori_jpos - min_val) / diff
        normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 
        return normalized_jpos # T X 22 X 3
    
    def extract_min_max_mean_std_from_data(self):
        min_jpos = np.inf*np.ones(156) 
        max_jpos = -np.inf*np.ones(156) 
        min_jvel = np.inf*np.ones(156)
        max_jvel = -np.inf*np.ones(156) 

        for s_idx in tqdm(self.window_data_dict): 
            current_jpos = self.window_data_dict[s_idx]['global_jpos'].reshape(-1, 156)
            current_jvel = self.window_data_dict[s_idx]['global_jvel'].reshape(-1, 156)

            min_jpos = np.minimum(min_jpos, current_jpos.min(axis=0))
            max_jpos = np.maximum(max_jpos, current_jpos.max(axis=0))
            min_jvel = np.minimum(min_jvel, current_jvel.min(axis=0))
            max_jvel = np.maximum(max_jvel, current_jvel.max(axis=0))

        stats_dict = {}
        stats_dict['global_jpos_min'] = min_jpos 
        stats_dict['global_jpos_max'] = max_jpos 
        stats_dict['global_jvel_min'] = min_jvel 
        stats_dict['global_jvel_max'] = max_jvel  

        return stats_dict
    
    def __len__(self):
        return len(self.window_data_dict)
    
    def __getitem__(self, index):
        globla_jpos = torch.from_numpy(self.window_data_dict[index]['global_jpos']).float().to(self.opt.device)# T X 52 X 3 
        global_rot6d = torch.from_numpy(self.window_data_dict[index]['global_rot_6d']).float().to(self.opt.device)# T X 52 X 6


        data_input = torch.cat((globla_jpos, global_rot6d), dim=-1) 
        normalized_jpos = self.normalize_jpos_min_max(data_input[:, :52*3].reshape(-1, 52, 3)) # T X 52 X 3 

        
        if(self.output_mode=="Body"):
            new_data_input = torch.cat((normalized_jpos.reshape(-1, 156)[:,:22*3], global_rot6d[:,:22*6]), dim=1)
        elif(self.output_mode=="Hand"):
            new_data_input = torch.cat((normalized_jpos.reshape(-1, 156)[:,22*3:], global_rot6d[:,22*6:]), dim=1)
        else: new_data_input = torch.cat((normalized_jpos.reshape(-1, 156), global_rot6d), dim=1) #T*(52*3+52*6)

        actual_seq_len = new_data_input.shape[0]

        if actual_seq_len < self.window:
            # Add padding
            padded_data_input = torch.zeros(self.window-actual_seq_len, new_data_input.shape[1]).to(self.opt.device) 
            new_data_input = torch.cat((new_data_input, padded_data_input), dim=0)
        elif actual_seq_len > self.window:
            new_data_input = new_data_input[0: 0 + self.window]
            actual_seq_len = self.window

        data_input_dict = {}
        data_input_dict['motion'] = new_data_input # T X (num_joints*3+num_joints*6) range [-1, 1]  #global jpos global rot 6d
        data_input_dict['seq_len'] = actual_seq_len 
        data_path = self.window_data_dict[index]['data_path']
        
        body_annot_list = self.window_data_dict[index]['body_cond']
        if body_annot_list is not None:
            hand_annot_list = self.window_data_dict[index]['hand_cond']
            if hand_annot_list is not None:
                annot_index = np.random.choice(len(hand_annot_list))
                data_input_dict['cond'] = hand_annot_list[annot_index] ###text    150 768 
                try:
                    raw_text = self.window_data_dict[index]['raw_text'][annot_index]
                except:
                    raw_text ="000"
            else:
                annot_index = np.random.choice(len(body_annot_list))
                data_input_dict['cond'] = body_annot_list[annot_index] ###text    150 768     
                try:
                    raw_text = self.window_data_dict[index]['raw_text'][annot_index]
                except:
                    raw_text ="000"
        else:
            data_input_dict['cond'] = np.zeros((150,768))

        data_input_dict['data_path'] = data_path

        if self.havecleanhand:
            hand_jpos = torch.from_numpy(self.window_data_dict[index]['clean_wist_hand_jpos']).float().to(self.opt.device)# T X 30 X 3 
            hand_rot6d = torch.from_numpy(self.window_data_dict[index]['clean_wist_hand_rot_6d']).float().reshape(-1, 30*6).to(self.opt.device)# T X 30 X 6
            hand_normalized_jpos = self.normalize_jpos_min_max_hand(hand_jpos.reshape(-1, 30, 3)).reshape(-1, 30*3) # T X 30 X 3 
            
            clean_wist_handmotion = torch.cat((hand_normalized_jpos,hand_rot6d),dim=-1).reshape(-1,270)
            hand_actual_seq_len = clean_wist_handmotion.shape[0]

            if hand_actual_seq_len < self.window:
                # Add padding
                padded_hand_data_input = torch.zeros(self.window-hand_actual_seq_len, clean_wist_handmotion.shape[1]) 
                clean_wist_handmotion = torch.cat((clean_wist_handmotion, padded_hand_data_input), dim=0)
            elif hand_actual_seq_len > self.window:
                clean_wist_handmotion = clean_wist_handmotion[0: 0 + self.window]
        else:
            clean_wist_handmotion = torch.zeros((self.window,270)).to(self.opt.device)
        
        return new_data_input, actual_seq_len,data_path,data_input_dict['cond'],clean_wist_handmotion,raw_text