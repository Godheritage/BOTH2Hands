import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pytorch3d.transforms as transforms 
from tools.cmd_loader import OptParse
from tools.utils import print_current_loss
from model.transformer_cond_diffusion_model import CondGaussianDiffusion
from ema_pytorch.ema_pytorch import EMA
import torch
import time
from torch.optim import Adam
from data.dataset import BodyHandDataset,quat_ik_torch,quat_fk_torch,local2global_pose
import numpy as np
import random
from einops import reduce
import clip
import torch.nn as nn
from torch.optim import AdamW
from eval_encoder.eval_warper import evaulation

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

class LitBodyhandMDM(pl.LightningModule):
    def __init__(self,
                 opt,
                 diffusion_model,
                 val_dl,
                 train_ds
                 ):

        super().__init__()
        self.save_hyperparameters(ignore=['diffusion_model','val_dl','train_ds'])
        
        self.val_epoch_counter = 0   
        
        self.it = 0
        self.opt = opt
        
        self.model = diffusion_model
        self.val_dl = val_dl
        self.train_ds = train_ds
        self.bm_dict = self.train_ds.bm_dict
        self.val_step_counter = 0
        
        random.seed(time.time())
        
        self.ema = EMA(diffusion_model, beta=0.995, update_every=10)
        self.step_start_ema = 2000
        
        ##trainer基础参数
        self.batch_size = self.opt.batch_size
        self.window = opt.window 
        self.avg_loss_logs = {}
        self.print_every = opt.print_every 
        self.log_every = opt.log_every 
        self.save_and_sample_every = opt.vis_every
        
        self.joints_num = 52*(opt.output_mode=="BodyHand") + 30*(opt.output_mode=="Hand") + 22*(opt.output_mode=="Body")
        
        ##结果保存
        self.results_folder = self.opt.save_root
        self.vis_folder = self.opt.vis_dir
        self.evl_folder = self.opt.evl_dir
        
        ##准备text encoder
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
        
        
        
    
    def prep_padding_mask(self, val_data, seq_len,for_encoder = False):
        # Generate padding mask 
        actual_seq_len = seq_len + 1 # BS, + 1 since we need additional timestep for noise level 
        
        if for_encoder:
            tmp_mask = torch.arange(self.window,device=val_data.device).expand(val_data.shape[0], \
            self.window) < actual_seq_len[:, None].repeat(1, self.window)
        else:
            tmp_mask = torch.arange(self.window+1,device=val_data.device).expand(val_data.shape[0], \
            self.window+1) < actual_seq_len[:, None].repeat(1, self.window+1)
        # BS X max_timesteps
        padding_mask = tmp_mask[:, None, :].to(val_data.device)

        return padding_mask
    
    
    def prep_body_condition_mask(self, data,joint_idxs=None): #joint_idxs = [0,1,...]
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)
        
        if(joint_idxs is not None):
            for joint_idx in joint_idxs:
                cond_pos_dim_idx = joint_idx * 3 
                cond_rot_dim_idx = self.joints_num * 3 + joint_idx * 6
                mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
                mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)

        return mask
    
    def prep_body_mask(self, data,data_path,joint_idxs=None): #joint_idxs = [0,1,...]
        # data: BS X T X D 
        mask = torch.ones_like(data).to(data.device)
        for i in range(len(data_path)):
            if(joint_idxs is not None):
                for joint_idx in joint_idxs:
                    cond_pos_dim_idx = joint_idx * 3 
                    cond_rot_dim_idx = self.joints_num * 3 + joint_idx * 6
                    mask[i:i+1, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(1, data.shape[1], 3).to(data.device)
                    mask[i:i+1, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(1, data.shape[1], 6).to(data.device)

        return mask
    
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
    
    def configure_optimizers(self):
        optimizer=Adam(self.model.parameters(), lr=self.opt.lr)
        return {"optimizer": optimizer,}
    
    def on_train_start(self):
        self.start_time = time.time()
    
    
    def clean_hand(self,fullmotion):
        
        B = fullmotion.shape[0] 
        
        body_global_rot_6d = fullmotion[:, :, self.joints_num*3:].reshape(B, -1, self.joints_num, 6)
        body_global_rotmat = transforms.rotation_6d_to_matrix(body_global_rot_6d.reshape(B,self.window,52,6))
        body_local_rotmat = quat_ik_torch(body_global_rotmat)
        
        body_local_rotaa = transforms.matrix_to_axis_angle(body_local_rotmat)
        
        return body_local_rotaa[:,:,22:].reshape(-1,30*3)
        
    def fullbodyforhand(self,handmotion,fullmotion):
        for joint_idx in list(range(22,52)):
                cond_pos_dim_idx = joint_idx * 3 
                cond_rot_dim_idx = 52 * 3 + joint_idx * 6
                
                hand_list_pos_idx = (joint_idx-22)*3
                hand_list_rot_idx = 30 * 3 +(joint_idx-22)*6

                fullmotion[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = handmotion[:, :, hand_list_pos_idx:hand_list_pos_idx+3].to(handmotion.device)
                fullmotion[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = handmotion[:, :, hand_list_rot_idx:hand_list_rot_idx+6].to(handmotion.device)
        return fullmotion
    
    def set_texthand_tobody(self,text_hand,body_motion):
        
        B = text_hand.shape[0] 
        local_tpose = self.train_ds.rest_human_offsets.repeat(B,self.opt.window, 1, 1) #b t j 3
        
        local_tpose_forhand = local_tpose.clone()
        local_tpose_forhand[:,:,0:22]=0
        local_tpose_mat = transforms.axis_angle_to_matrix(local_tpose_forhand)
        global_tpose = local2global_pose(local_tpose_mat.reshape(B*self.opt.window,-1,3,3)).reshape(B,self.opt.window,-1,3,3)
        
        text_clean_global_rot_6d = text_hand[:, :, 30*3:].reshape(B, -1, 30, 6)
        text_clean_global_rotmat = transforms.rotation_6d_to_matrix(text_clean_global_rot_6d) # N X T X self.joints_num X 3 X 3 
        
        global_tpose[:,:,22:] = text_clean_global_rotmat[:,:,:]
        
        local_tpose_withcleanhand_mat= quat_ik_torch(global_tpose)
        
        body_global_rot_6d = body_motion[:, :, self.joints_num*3:].reshape(B, -1, self.joints_num, 6)
        body_global_rotmat = transforms.rotation_6d_to_matrix(body_global_rot_6d.reshape(B,self.window,52,6))
        body_local_rotmat= quat_ik_torch(body_global_rotmat)
            
        text_clean_curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(local_tpose_withcleanhand_mat) # T X self.joints_num X 3
        body_local_rot_aa = transforms.matrix_to_axis_angle(body_local_rotmat)
        
        text_local_rotaa_fullbody = body_local_rot_aa.clone()
        # text_local_rotaa_fullbody = text_clean_curr_local_rot_aa_rep.clone()
        text_local_rotaa_fullbody[:,:,22:] = text_clean_curr_local_rot_aa_rep[:,:,22:] 
        text_local_rotmat_fullbody = transforms.axis_angle_to_matrix(text_local_rotaa_fullbody)
        
        _, text_jointlocalpos = quat_fk_torch(text_local_rotmat_fullbody, local_tpose)
        text_jointlocalpos = self.train_ds.normalize_jpos_min_max(text_jointlocalpos)
        
        text_left_hand_jnts = (text_jointlocalpos[:,:,22:37])
        text_right_hand_jnts = (text_jointlocalpos[:,:,37:])
        text_hand_jnts = torch.cat((text_left_hand_jnts,text_right_hand_jnts),dim=2)
        
        single_bs_text_global_rotmat = local2global_pose(text_local_rotmat_fullbody.reshape(B*self.opt.window,-1,3,3)).reshape(B,self.opt.window,-1,3,3) # T' X 52 X 3 X 3 
        text_hand_global_rot6d = transforms.matrix_to_rotation_6d(single_bs_text_global_rotmat)[:,:,22:]
            
        
        text_onbodyjpos = text_hand_jnts.reshape(B,self.window,30*3)
        text_pose = text_hand_global_rot6d.reshape(B,self.window,30*6)
        
        return torch.cat((text_onbodyjpos,text_pose),dim=-1).to(self.opt.device)
    
    def recover_smpl_from_data(self,text_hand,body_hand,style_hand):
        motion_Loss_type = torch.nn.L1Loss(reduction='none')
        text_onbodyjpos = text_hand[:,:,:30*3].reshape(-1,self.window,30,3)
        text_pose = text_hand[:,:,30*3:].reshape(-1,self.window,30,6)
        body_jpos = body_hand[:,:,:30*3].reshape(-1,self.window,30,3)
        body_pose = body_hand[:,:,30*3:].reshape(-1,self.window,30,6)
        style_jpos = style_hand[:,:,:30*3].reshape(-1,self.window,30,3)
        style_pose = style_hand[:,:,30*3:].reshape(-1,self.window,30,6)
            
        ##学习style motion
        body_pose_loss = motion_Loss_type(style_pose, body_pose)
        body_local_jpos_loss = motion_Loss_type(style_jpos, body_jpos)
        
        body_pose_loss = reduce(body_pose_loss, 'b ... -> b (...)', 'mean')
        body_local_jpos_loss = reduce(body_local_jpos_loss, 'b ... -> b (...)', 'mean')
        
        text_pose_loss = motion_Loss_type(style_pose, text_pose)
        text_local_jpos_loss = motion_Loss_type(style_jpos, text_onbodyjpos)
        
        text_pose_loss = reduce(text_pose_loss, 'b ... -> b (...)', 'mean')
        text_local_jpos_loss = reduce(text_local_jpos_loss, 'b ... -> b (...)', 'mean')
        
        return (body_pose_loss.mean()+body_local_jpos_loss.mean())*self.opt.bodystyleloss_weight+\
            (text_pose_loss.mean()+text_local_jpos_loss.mean())*self.opt.textstyleloss_weight
            
    def configure_optimizers(self):
        optimizer = AdamW(model.parameters(), lr=self.opt.lr, weight_decay=0.0002)
        return optimizer
    
    def fillbodymotion(self,body_motion,body_model_out):
        body_fullmotion = body_motion.clone().detach()
        for joint_idx in list(range(22,52)):
            cond_pos_dim_idx = joint_idx * 3 
            cond_rot_dim_idx = 52 * 3 + joint_idx * 6
            hand_list_pos_idx = (joint_idx-22)*3
            hand_list_rot_idx = 30 * 3 +(joint_idx-22)*6

            body_fullmotion[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = body_model_out[:, :, hand_list_pos_idx:hand_list_pos_idx+3].to(body_fullmotion.device)
            body_fullmotion[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = body_model_out[:, :, hand_list_rot_idx:hand_list_rot_idx+6].to(body_fullmotion.device)
            
        return body_fullmotion
    
    def training_step(self, batch, batch_idx):
        motion,seq_len,data_path,embed_text,clean_wist_handmotion,raw_text = batch
        
        body_motion = motion.clone()

        bs = torch.randperm(clean_wist_handmotion.shape[0])
        embed_text = embed_text[bs]
        clean_wist_handmotion = clean_wist_handmotion[bs]
        
        body_padding_mask = self.prep_padding_mask(body_motion, seq_len)
        text_padding_mask = self.prep_padding_mask(clean_wist_handmotion, seq_len)
        
        if self.opt.style_epochs< self.trainer.current_epoch:
            with torch.no_grad():
                loss_diffusion,text_model_out,body_model_out= self.model(text_cond = embed_text,
                                                                        body_x_start=body_motion,
                                                                        text_padding_mask=text_padding_mask,body_padding_mask=body_padding_mask,
                                                                        train_style=True,clean_wist_handmotion=clean_wist_handmotion)
                
                body_fullmotion = self.fillbodymotion(body_motion,body_model_out)
                text_onbody = self.set_texthand_tobody(text_hand=text_model_out,body_motion=body_fullmotion)
               
            style_model_out = self.model.style_forward(train_style=True,text_hand=text_onbody,body_hand=body_model_out)
            style_loss = self.recover_smpl_from_data(text_hand=text_onbody,body_hand=body_model_out,style_hand=style_model_out)
            loss_diffusion=0
        else:
            loss_diffusion,_,_= self.model(text_cond = embed_text,
                                            body_x_start=body_motion,
                                            text_padding_mask=text_padding_mask,body_padding_mask=body_padding_mask
                                            ,clean_wist_handmotion=clean_wist_handmotion)
            style_loss=0
        
        loss = loss_diffusion+style_loss
        if torch.isnan(loss).item():
            print('WARNING: NaN loss. Skipping to next data...')
            return None
        
        loss_logs = {}
        loss_logs["total_loss"] = loss
        loss_logs["diffusion_loss"] = loss_diffusion
        loss_logs["style_loss"] = style_loss
        
        return {"loss": loss,
                "loss_logs": loss_logs}
            
            
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema.update()
        if(outputs is not None):
            self.it += 1
            for k,v in outputs['loss_logs'].items():
                if k not in self.avg_loss_logs:
                    self.avg_loss_logs[k] = v
                else:
                    self.avg_loss_logs[k] += v
            
            if(self.it%self.print_every==0):
                print_current_loss(self.start_time,self.it,outputs['loss_logs'],
                                   self.trainer.current_epoch,inner_iter=batch_idx,
                                   lr=self.trainer.optimizers[0].param_groups[0]['lr'])
            
            if(self.it%self.log_every==0):
                for k,v in self.avg_loss_logs.items():
                    self.avg_loss_logs[k] = v/self.log_every
                self.log_dict(self.avg_loss_logs)
                self.avg_loss_logs = {}
            
            if self.it != 0 and self.it % self.save_and_sample_every == 0:                     
                self.motion_gen(val_data_dict=batch)
                

    def validation_step(self, batch, batch_idx):

        self.val_step_counter +=1
        self.it = self.val_step_counter
        
        if self.opt.mode =="eval":
            if self.val_step_counter> self.opt.validate_batch_size:
                return

        self.save_batch_idx = batch_idx
        
        if self.opt.costume_promot != None:
            self.motion_gen(val_data_dict=batch,costume_prompt=self.opt.costume_promot)
        else:
            self.motion_gen(val_data_dict=batch)
        
        if self.opt.mode =="gen":
            if self.opt.gen_times<=self.val_step_counter:
                quit()
        
        
    
    def motion_gen(self,val_data_dict = None,costume_prompt=None):
        self.ema.ema_model.eval()
        self.quantative_vis(val_data_dict=val_data_dict,costume_prompt = costume_prompt) 
        self.ema.ema_model.train()    
        
    def quantative_vis(self,val_data_dict = None,costume_prompt=None): 
        with torch.no_grad():
            #val_motion BS T D
            #val_seq_len BS 
            #val_data_path BS 
            val_motion,val_seq_len,val_data_path,embed_text,clean_wist_handmotion,raw_text = val_data_dict
            if costume_prompt is not None:

                text_descript = costume_prompt
                text_descript = text_descript.lower().replace(",","").replace(".","")
                
                val_data_text = np.tile(text_descript, self.opt.batch_size)
                embed_text = self.encode_text(val_data_text,val_motion.device)
                raw_text = val_data_text

            
            val_motion = val_motion.cuda(self.opt.device)
            val_seq_len = val_seq_len.cuda(self.opt.device)
            
            body_motion = val_motion.clone()
            text_motion = val_motion.clone()
            body_padding_mask = self.prep_padding_mask(body_motion, val_seq_len)
            text_padding_mask = self.prep_padding_mask(text_motion, val_seq_len)
            text_out,body_out = self.ema.ema_model.sample(body_motion,embed_text,text_padding_mask,body_padding_mask)
            text_onbody = self.set_texthand_tobody(text_hand=text_out,body_motion=body_motion)
            style_out = self.ema.ema_model.style_transfer_loop(body_x=body_out,text_x = text_onbody)
            
        if self.opt.mode == "eval":
            save_dict = {}
            save_dict["val_motion"] = val_motion.cpu().numpy()
            save_dict["val_seq_len"] = val_seq_len.cpu().numpy()
            save_dict["val_data_path"] = val_data_path
            save_dict["embed_text"] = embed_text.cpu().numpy()
            save_dict["clean_wist_handmotion"] = clean_wist_handmotion.cpu().numpy()
            save_dict["raw_text"] = raw_text
            save_dict["style_out"] = style_out.cpu().numpy()
            save_dict["bodymodel_out"] = body_out.cpu().numpy()
            save_dict["textmodel_out"] = text_onbody.cpu().numpy()
            sub_folder = os.path.join(self.evl_folder,str(self.val_epoch_counter))
            save_path = os.path.join(sub_folder, str(self.save_batch_idx)+".npy")
            os.makedirs(sub_folder,exist_ok=True)
            np.save(save_path,save_dict)
        else:
            style_fullmotion = self.fillbodymotion(body_motion,style_out)
            self.gen_vis_res(style_fullmotion, self.it, sub_folder_name="style")
    
    def gen_vis_res(self,all_res_list, step, sub_folder_name=None):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :self.joints_num*3].reshape(num_seq, -1, self.joints_num, 3)          
        global_jpos = self.train_ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, self.joints_num, 3))
        global_jpos[:,:,1]=0
        global_jpos = global_jpos.reshape(num_seq, -1, self.joints_num, 3) # N X T X self.joints_num X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        global_root_jpos_frame0 = global_root_jpos[:,0:1,None,:].clone()# N X T X self.joints_num X 3 
        global_root_jpos_frame0[:,:,:,2] -=0.95 
        global_jpos = global_jpos - global_root_jpos_frame0
        global_root_jpos = global_jpos[:, :, 0, :].clone()
        ###
        global_rot_6d = all_res_list[:, :, self.joints_num*3:].reshape(num_seq, -1, self.joints_num, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X self.joints_num X 3 X 3 
        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X self.joints_num X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X self.joints_num X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X self.joints_num X 3 

            pose_forunity = curr_local_rot_aa_rep.cpu().numpy().reshape(-1,156)
            tran_forunity = global_root_jpos[idx].cpu().numpy().reshape(-1,3)
            unity_motion = []
            for p, t in zip(pose_forunity, tran_forunity):
                s = ','.join(['%g' % v for v in p]) + '#' + \
                    ','.join(['%g' % v for v in t]) + '$'
                unity_motion.append(s)
            dest_mesh_vis_folder = os.path.join(self.vis_folder, sub_folder_name, str(step))
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)
            ###存储unity txt
            unity_save_file = os.path.join(dest_mesh_vis_folder, \
                            sub_folder_name+"_"+str(step)+"_"+str(idx)+".txt")
            unity_save_file = os.path.abspath(unity_save_file)
            with open(unity_save_file, 'a') as f:
                for i in range(len(unity_motion)):
                    f.write(unity_motion[i] + '\n')
                f.close()
                   
def cycle(dl):
    while True:
        for data in dl:
            yield data        
        
        
if __name__ == '__main__':
    opt = OptParse()
    pl.seed_everything(opt.opt.seed)
    #joints_num = 52*(opt.opt.output_mode=="BodyHand") + 30*(opt.opt.output_mode=="Hand") + 22*(opt.opt.output_mode=="Body")
    # wist_dim = 2*3+2*6
    # full_body_dim = 52 * 3 + 52 * 6 
    # body_dim = 22*3+22*6
    # hand_dim = 30*3+30*6
    
    ###text wistcond to hand 
    diffusion_model = CondGaussianDiffusion(d_model=opt.opt.d_model, \
            n_dec_layers=opt.opt.n_dec_layers, n_head=opt.opt.n_head, d_k=opt.opt.d_k, d_v=opt.opt.d_v, \
            max_timesteps=opt.opt.window+1, timesteps=1000, \
            objective="pred_x0", loss_type="l1", \
            batch_size=opt.opt.batch_size)
    
    checkpoint_callback = ModelCheckpoint(
            dirpath=opt.opt.checkpoint_dir, 
            save_top_k=opt.opt.save_top_k, 
            monitor="total_loss",
            save_last=opt.opt.save_last,
            every_n_train_steps=opt.opt.save_frequency)
    
    
    dataset = BodyHandDataset(opt = opt.opt,datasets_to_process=opt.opt.datasets_to_process)
    
    drop_last = opt.opt.mode == "train" 
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=opt.opt.batch_size,
                                            shuffle=True,
                                            drop_last = drop_last)
    val_dl = cycle(dataset)
    model = LitBodyhandMDM(opt=opt.opt,diffusion_model=diffusion_model,val_dl=val_dl,train_ds = dataset)
    
    opt.opt.num_epochs = 1 if opt.opt.mode != "train" else opt.opt.num_epochs
    trainer = pl.Trainer(accelerator="gpu",
                        devices=[opt.opt.device],
                        max_epochs=opt.opt.num_epochs,
                        precision=32,
                        default_root_dir=opt.opt.model_dir,
                        callbacks=[checkpoint_callback])
    
    ###run###
    if opt.opt.mode =="train":
        if opt.opt.rewake:
            trainer.strategy.strict_loading = False
            trainer.fit(model=model, train_dataloaders=dataloader,ckpt_path=opt.opt.resume_ckpt_path)
        else:
            trainer.fit(model=model, train_dataloaders=dataloader)
    elif opt.opt.mode =="gen":
        trainer.strategy.strict_loading = False
        trainer.validate(model=model, dataloaders=dataloader,ckpt_path=opt.opt.resume_ckpt_path)
    elif opt.opt.mode =="eval":
        model.opt.costume_promot = None
        trainer.strategy.strict_loading = False
        for i in range(opt.opt.validate_times):
            trainer.validate(model=model, dataloaders=dataloader,ckpt_path=opt.opt.resume_ckpt_path)
            model.val_epoch_counter+=1
        eval_path = model.evl_folder  
        evaulation(npy_folder=eval_path,dataloader=dataset)
        
        