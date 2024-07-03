import torch
from torch import nn

from model.transformer_module import Decoder,CrossDecoder

import math 

from tqdm.auto import tqdm

from einops import reduce

from inspect import isfunction

import torch.nn.functional as F

import pytorch3d.transforms as transforms 


from data.dataset import quat_ik_torch 

from lafan1.utils import rotate_at_frame_smplh
 
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
        
class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_feats, 
        d_model, 
        n_dec_layers, 
        n_head, 
        d_k, 
        d_v, 
        max_timesteps, 
        out_dim,
        cond_dim=None, 
        cross=False
    ):
        super().__init__()
        
        self.d_feats = d_feats 
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k 
        self.d_v = d_v 
        self.max_timesteps = max_timesteps 
        self.cond_dim = cond_dim

        # Input: BS X D X T 
        # Output: BS X T X D'
        
        if cross:
            self.motion_transformer = CrossDecoder(d_feats=self.d_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)
        else:
            self.motion_transformer = Decoder(d_feats=self.d_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True)

        self.linear_out = nn.Linear(self.d_model, out_dim)

        # For noise level t embedding
        dim = 64 
        time_dim = dim * 4

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )
        
        if cond_dim is not None:
            self.cond_embeding = nn.Linear(self.cond_dim, d_model)

    def forward(self, src, noise_t, padding_mask=None,cond=None):
        # src: BS X T X D
        # noise_t: int 
       
        noise_t_embed = self.time_mlp(noise_t) # BS X d_model 
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 
        
        bs = src.shape[0]
        num_steps = src.shape[1] + 1
        
        if cond is not None:
            cond_embed = self.cond_embeding(cond)# BS X T X d_model 
            embed = noise_t_embed + cond_embed
            
            # embed = noise_t_embed
            
            
            # ##把text vec放到每帧motion维度
            # src = torch.cat((cond_embed,src),dim=2)
            
        else: embed = noise_t_embed
        
        

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps)+1 # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T                                        
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=embed)
    
        output = self.linear_out(feat_pred[:, 1:]) # BS X T X D

        return output # predicted noise, the same size as the input 
    
    def forward_style(self, src,style, padding_mask=None):
        # src: BS X T X D
        # style: BS X T X D
        # noise_t: int 
       
        # noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model 
        
        bs = src.shape[0]
        num_steps = src.shape[1]
        

        if padding_mask is None:
            # In training, no need for masking 
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        data_input = src.transpose(1, 2).detach() # BS X D X T 
        style_input = style.transpose(1, 2).detach() # BS X D X T                                       
        feat_pred, _ = self.motion_transformer(data_input,style_input, padding_mask, pos_vec)
    
        output = self.linear_out(feat_pred) # BS X T X D

        return output # predicted noise, the same size as the input 




class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        d_model, 
        n_head, 
        n_dec_layers, 
        d_k, 
        d_v, 
        max_timesteps, 
        timesteps = 1000, 
        loss_type = 'l1', 
        objective = 'pred_noise', 
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        batch_size=None,
    ):
        super().__init__()

        wist_dim = 2*3+2*6
        full_body_dim = 52 * 3 + 52 * 6 
        # body_dim = 22*3+22*6
        hand_dim = 30*3+30*6
        
        
        self.text_denoise_fn = TransformerDiffusionModel(d_feats=hand_dim, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, out_dim=hand_dim,max_timesteps=max_timesteps,cond_dim=768) 
        

        self.body_denoise_fn = TransformerDiffusionModel(d_feats=full_body_dim, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, out_dim=hand_dim,max_timesteps=max_timesteps) 
        

        self.style_transfer_fn = TransformerDiffusionModel(d_feats=hand_dim, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, out_dim=hand_dim,max_timesteps=max_timesteps,cross=True) 
        
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x,x_cond, t, clip_denoised,cond=None,padding=None,diffusion_type="No"):
        if x_cond==None:
            x_all = x
        else:
            x_all = torch.cat((x, x_cond), dim=-1)
        
        if diffusion_type=="text":
            model_output = self.text_denoise_fn(x_all, t,cond=cond,padding_mask = padding)
        elif diffusion_type=="body":
            model_output = self.body_denoise_fn(x_all, t,cond=None,padding_mask = padding)
        else:
            raise ValueError('未知diffusion类型 '+diffusion_type)
        
        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, x_cond,t, clip_denoised=True,cond = None,padding=None,diffusion_type="No"):
        b, *_, device = *x.shape, x.device
        if diffusion_type=="text":
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, x_cond=x_cond,t=t, \
            clip_denoised=clip_denoised,cond=cond,padding=padding,diffusion_type="text")
        elif diffusion_type=="body":
            model_mean, _, model_log_variance = self.p_mean_variance(x=x, x_cond=x_cond, t=t, \
            clip_denoised=clip_denoised,cond=None,padding=padding,diffusion_type="body")
        else:
            raise ValueError('未知diffusion类型 '+diffusion_type)
        
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self,body_x_start, text_cond,text_padding=None,body_padding=None):
        device = self.betas.device

        
        
        b = body_x_start.shape[0]
        t = body_x_start.shape[1]
        
        
        text_x = torch.randn((b,t,270), device=device)#bs t d
        
        body_x_start_body = body_x_start.clone()
        body_x_start_body = torch.cat((body_x_start_body[:,:,0:22*3],body_x_start_body[:,:,52*3:52*3+22*6]),dim=2)
        body_x_start_hand = body_x_start.clone()
        body_x_start_hand = torch.cat((body_x_start_hand[:,:,22*3:52*3],body_x_start_hand[:,:,52*3+22*6:]),dim=2)
        
        body_x = torch.randn(body_x_start_hand.shape, device=device)#bs t d
        body_x_cond = body_x_start_body
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            # text_x = self.p_sample(text_x,text_x_cond, torch.full((b,), i, device=device, dtype=torch.long),cond = text_cond,padding=text_padding,diffusion_type="text")
            text_x = self.p_sample(text_x,None, torch.full((b,), i, device=device, dtype=torch.long),cond = text_cond,padding=text_padding,diffusion_type="text")
            body_x = self.p_sample(body_x,body_x_cond, torch.full((b,), i, device=device, dtype=torch.long),cond = None,padding=body_padding,diffusion_type="body")     

        
        return text_x,body_x
        
        
    
    @torch.no_grad()
    def style_transfer_loop(self,body_x, text_x):
        style_x = body_x
        for i in tqdm(reversed(range(0, 2)), desc='进行style transfer', total=self.num_timesteps):
            style_x = self.style_transfer_fn.forward_style(src=style_x,style=text_x,padding_mask=None)
        
        return style_x # BS X T X D
    
    @torch.no_grad()
    def p_sample_loop_betweening(self, shape, x_start, cond_mask,between_length ,padding_mask=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)
        
        
        x_cond = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask)    
            x = torch.cat((x_start[:,:between_length],x[:,between_length:]),dim=1) 

        return x # BS X T X D
    
    @torch.no_grad()
    def p_sample_loop_continue(self, shape, x_start, cond_mask,keep_length ,padding_mask=None):
        device = self.betas.device

        b = shape[0]
        #keep_length是保留前一段的长度
        motion_length = shape[1]
        x = torch.randn(shape, device=device)
        
        
        x_cond = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask)    
            x = torch.cat((x_start[:,(motion_length-keep_length):],x[:,keep_length:]),dim=1) 

        return x # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window(self, shape, x_start, cond_mask):
        device = self.betas.device

        b = shape[0]
        assert b == 1
        
        x_all = torch.randn(shape, device=device)
        x_cond_all = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        x_blocks = []
        x_cond_blocks = []
        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = x_start.shape[1]
        stride = self.window // 2
        for t_idx in range(0, num_steps, stride):
            x = x_all[0, t_idx:t_idx+self.window]
            x_cond = x_cond_all[0, t_idx:t_idx+self.window]

            x_blocks.append(x) # T X D 
            x_cond.append(x_cond) # T X D 

        last_window_x = None 
        last_window_cond = None 
        if x_blocks[-1].shape[0] != x_blocks[0].shape[0]:
            last_window_x = x_blocks[-1][None] # 1 X T X D 
            last_window_cond = x_cond_blocks[-1][None] 

            x_blocks = torch.stack(x_blocks[:-1]) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks[:-1]) # K X T X D 
        else:
            x_blocks = torch.stack(x_blocks) # K X T X D 
            x_cond_blocks = torch.stack(x_cond_blocks) # K X T X D 
       
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x_blocks = self.p_sample(x_blocks, torch.full((b,), i, device=device, dtype=torch.long), x_cond_blocks)    

        if last_window_x is not None:
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                last_window_x = self.p_sample(last_window_x, torch.full((b,), i, device=device, dtype=torch.long), last_window_cond)    

        # Convert from K X T X D to a single sequence.
        seq_res = None  
        # for t_idx in range(0, num_steps, stride):
        num_windows = x_blocks.shape[0]
        for w_idx in range(num_windows):
            if w_idx == 0:
                seq_res = x_blocks[w_idx] # T X D 
            else:
                seq_res = torch.cat((seq_res, x_blocks[self.window-stride:]), dim=0)

        if last_window_x is not None:
            seq_res = torch.cat((seq_res, last_window_x[self.window-stride:]), dim=0)

        return seq_res # BS X T X D

    @torch.no_grad()
    def p_sample_loop_sliding_window_w_canonical(self, ds, shape, global_head_jpos, global_head_jquat, cond_mask):
        # shape: BS X T X D 
        # global_head_jpos: BS X T X 3 
        # global_head_jquat: BS X T X 4 
        # cond_mask: BS X T X D 

        device = self.betas.device

        b = shape[0]
        # assert b == 1
        
        x_all = torch.randn(shape, device=device)

        whole_seq_aa_rep = None 
        whole_seq_root_pos = None 
        whole_seq_head_pos = None 

        # Divide to blocks to form a batch, then just need run model once to get all the results. 
        num_steps = global_head_jpos.shape[1]
        # stride = self.seq_len // 2
        overlap_frame_num = 10
        stride = self.seq_len - overlap_frame_num 
        for t_idx in range(0, num_steps, stride):
            curr_x = x_all[:, t_idx:t_idx+self.seq_len]

            if curr_x.shape[1] <= self.seq_len - stride:
                break 

            # Canonicalize current window 
            curr_global_head_quat = global_head_jquat[:, t_idx:t_idx+self.seq_len] # BS X T X 4
            curr_global_head_jpos = global_head_jpos[:, t_idx:t_idx+self.seq_len] # BS X T X 3 

            aligned_head_trans, aligned_head_quat, recover_rot_quat = \
            rotate_at_frame_smplh(curr_global_head_jpos.data.cpu().numpy(), \
            curr_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            aligned_head_trans = torch.from_numpy(aligned_head_trans).to(global_head_jpos.device)
            aligned_head_quat = torch.from_numpy(aligned_head_quat).to(global_head_jpos.device)

            move_to_zero_trans = aligned_head_trans[:, 0:1, :].clone() # Move the head joint x, y to 0,  BS X 1 X 3
            move_to_zero_trans[:, :, 2] = 0 

            aligned_head_trans = aligned_head_trans - move_to_zero_trans # BS X T X 3 

            aligned_head_rot_mat = transforms.quaternion_to_matrix(aligned_head_quat) # BS X T X 3 X 3 
            aligned_head_rot_6d = transforms.matrix_to_rotation_6d(aligned_head_rot_mat) # BS X T X 6  

            head_idx = 15 
            curr_x_start = torch.zeros(aligned_head_rot_6d.shape[0], \
            aligned_head_rot_6d.shape[1], 22*3+22*6).to(aligned_head_rot_6d.device)
            curr_x_start[:, :, head_idx*3:head_idx*3+3] = aligned_head_trans # BS X T X 3
            curr_x_start[:, :, 22*3+head_idx*6:22*3+head_idx*6+6] = aligned_head_rot_6d # BS X T X 6 

            # Normalize data to [-1, 1]
            normalized_jpos = ds.normalize_jpos_min_max(curr_x_start[:, :, :22*3].reshape(-1, 22, 3))
            curr_x_start[:, :, :22*3] = normalized_jpos.reshape(b, -1, 22*3) # BS X T X (22*3)

            curr_cond_mask = cond_mask[:, t_idx:t_idx+self.seq_len] # BS X T X D 
            curr_x_cond = curr_x_start * (1. - curr_cond_mask) + \
            curr_cond_mask * torch.randn_like(curr_x_start).to(curr_x_start.device)
       
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                curr_x = self.p_sample(curr_x, torch.full((b,), i, device=device, dtype=torch.long), curr_x_cond)    
                # Apply previous window prediction as additional condition, direcly replacement. 
                if t_idx > 0:
                    curr_x[:, :self.seq_len-stride, 22*3:] = prev_res_rot_6d.reshape(b, -1, 22*6)
                    curr_x[:, :self.seq_len-stride, :22*3] = prev_res_jpos.reshape(b, -1, 22*3)

            curr_seq_local_aa_rep, curr_seq_root_pos, curr_seq_head_pos = \
            self.convert_model_res_to_data(ds, curr_x, \
            recover_rot_quat, curr_global_head_jpos) 
            
            if t_idx == 0:
                whole_seq_aa_rep = curr_seq_local_aa_rep # BS X T X 22 X 3
                whole_seq_root_pos = curr_seq_root_pos # BS X T X 3 
                whole_seq_head_pos = curr_seq_head_pos # BS X T X 3 
            else:
                prev_last_pos = whole_seq_head_pos[:, -1:, :].clone() # BS X 1 X 3 
                curr_first_pos = curr_seq_head_pos[:, self.seq_len-stride-1:self.seq_len-stride, :].clone() # BS X 1 X 3
                
                move_trans = prev_last_pos - curr_first_pos # BS X 1 X 3 
                curr_seq_root_pos += move_trans # BS X T X 3 
                curr_seq_head_pos += move_trans 
                
                whole_seq_aa_rep = torch.cat((whole_seq_aa_rep, \
                curr_seq_local_aa_rep[:, self.seq_len-stride:]), dim=1)
                whole_seq_root_pos = torch.cat((whole_seq_root_pos, \
                curr_seq_root_pos[:, self.seq_len-stride:]), dim=1)
                whole_seq_head_pos = torch.cat((whole_seq_head_pos, \
                curr_seq_head_pos[:, self.seq_len-stride:]), dim=1)

            # Convert results to normalized representation for sampling in next window
            tmp_global_rot_quat, tmp_global_jpos = ds.fk_smpl(curr_seq_root_pos.reshape(-1, 3), \
            curr_seq_local_aa_rep.reshape(-1, 22, 3)) 
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            tmp_global_rot_quat = tmp_global_rot_quat.reshape(b, -1, 22, 4)
            tmp_global_jpos = tmp_global_jpos.reshape(b, -1, 22, 3)

            tmp_global_rot_quat = tmp_global_rot_quat[:, -self.seq_len+stride:].clone()
            tmp_global_jpos = tmp_global_jpos[:, -self.seq_len+stride:].clone()
            
            tmp_global_head_quat = tmp_global_rot_quat[:, :, 15, :] # BS X T X 4 
            tmp_global_head_jpos = tmp_global_jpos[:, :, 15, :] # BS X T X 3 

            tmp_aligned_head_trans, tmp_aligned_head_quat, tmp_recover_rot_quat = \
            rotate_at_frame_smplh(tmp_global_head_jpos.data.cpu().numpy(), \
            tmp_global_head_quat.data.cpu().numpy(), cano_t_idx=0)
            # BS X T' X 3, BS X T' X 4, BS X 1 X 1 X 4  

            tmp_aligned_head_trans = torch.from_numpy(tmp_aligned_head_trans).to(tmp_global_head_jpos.device)

            tmp_move_to_zero_trans = tmp_aligned_head_trans[:, 0:1, :].clone() 
            # Move the head joint x, y to 0,  BS X 1 X 3
            tmp_move_to_zero_trans[:, :, 2] *= 0 # 1 X 1 X 3 

            tmp_aligned_head_trans = tmp_aligned_head_trans - tmp_move_to_zero_trans # BS X T X 3 

            tmp_recover_rot_quat = torch.from_numpy(tmp_recover_rot_quat).float().to(tmp_global_rot_quat.device)

            tmp_global_jpos = transforms.quaternion_apply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_jpos.shape[1], \
            tmp_global_jpos.shape[2], 1), tmp_global_jpos) # BS X T X 22 X 3

            tmp_global_jpos -= tmp_move_to_zero_trans[:, :, None, :] 

            prev_res_jpos = tmp_global_jpos.clone() 
            prev_res_jpos = ds.normalize_jpos_min_max(prev_res_jpos.reshape(-1, 22, 3)).reshape(b, -1, 22, 3) # BS X T X 22 X 3 

            prev_res_global_rot_quat = transforms.quaternion_multiply(transforms.quaternion_invert(\
            tmp_recover_rot_quat).repeat(1, tmp_global_rot_quat.shape[1], \
            tmp_global_rot_quat.shape[2], 1), \
            tmp_global_rot_quat) # BS X T X 22 X 4
            prev_res_rot_mat = transforms.quaternion_to_matrix(prev_res_global_rot_quat) # BS X T X 22 X 3 X 3 
            prev_res_rot_6d = transforms.matrix_to_rotation_6d(prev_res_rot_mat) # BS X T X 22 X 6 

        return whole_seq_aa_rep, whole_seq_root_pos
        # T X 22 X 3, T X 3 

    def convert_model_res_to_data(self, ds, all_res_list, recover_rot_quat, curr_global_head_jpos):
        # all_res_list: BS X T X D 
        # recover_rot_quat: BS X 1 X 1 X 4 
        # curr_global_head_jpos: BS X T X 3 

        # De-normalize jpos 
        use_global_head_pos_for_root_trans = False 

        bs = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :22*3].reshape(bs, -1, 22, 3) # BS X T X 22 X 3 
      
        global_jpos = ds.de_normalize_jpos_min_max(normalized_global_jpos.reshape(-1, 22, 3)) # (BS*T) X 22 X 3
        global_jpos = global_jpos.reshape(bs, -1, 22, 3) # BS X T X 22 X 3 

        global_rot_6d = all_res_list[:, :, 22*3:] # BS X T X (22*6)
        
        bs, num_steps, _, _ = global_jpos.shape
        global_rot_6d = global_rot_6d.reshape(bs, num_steps, 22, 6) # BS X T X 22 X 6 
        
        global_root_jpos = global_jpos[:, :, 0, :] # BS X T X 3 

        head_idx = 15 
        global_head_jpos = global_jpos[:, :, head_idx, :] # BS X T X 3 

        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # BS X T X 22 X 3 X 3
        global_quat = transforms.matrix_to_quaternion(global_rot_mat) # BS X T X 22 X 4 
        recover_rot_quat = torch.from_numpy(recover_rot_quat).to(global_quat.device) # BS X 1 X 1 X 4 
        ori_global_quat = transforms.quaternion_multiply(recover_rot_quat, global_quat) # BS X T X 22 X 4 
        ori_global_root_jpos = global_root_jpos # BS X T X 3 
        ori_global_root_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        ori_global_root_jpos) # BS X T X 3 

        ori_global_head_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        global_head_jpos) # BS X T X 3 

        # Convert global join rotation to local joint rotation
        ori_global_rot_mat = transforms.quaternion_to_matrix(ori_global_quat) # BS X T X 22 X 3 X 3
        ori_local_rot_mat = quat_ik_torch(ori_global_rot_mat.reshape(-1, 22, 3, 3)).reshape(bs, -1, 22, 3, 3) # BS X T X 22 X 3 X 3 
        ori_local_aa_rep = transforms.matrix_to_axis_angle(ori_local_rot_mat) # BS X T X 22 X 3 

        if use_global_head_pos_for_root_trans: 
            zero_root_trans = torch.zeros(ori_local_aa_rep.shape[0], ori_local_aa_rep.shape[1], 3).to(ori_local_aa_rep.device).float()
            betas = torch.zeros(bs, 10).to(zero_root_trans.device).float()
            gender = ["male"] * bs 

            _, mesh_jnts = ds.fk_smpl(zero_root_trans.reshape(-1, 3), ori_local_aa_rep.reshape(-1, 22, 3))
            # (BS*T) X 22 X 4, (BS*T) X 22 X 3 
            mesh_jnts = mesh_jnts.reshape(bs, -1, 22, 3) # BS X T X 22 X 3 

            head_idx = 15 
            wo_root_trans_head_pos = mesh_jnts[:, :, head_idx, :] # BS X T X 3 

            calculated_root_trans = ori_global_head_jpos - wo_root_trans_head_pos # BS X T X 3 

            return ori_local_aa_rep, calculated_root_trans, ori_global_head_jpos

        return ori_local_aa_rep, ori_global_root_jpos, ori_global_head_jpos

    @torch.no_grad()
    def sample(self,body_x_start, text_cond,text_padding=None,body_padding=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.text_denoise_fn.eval()
        self.body_denoise_fn.eval() 
        self.style_transfer_fn.eval()
        
        text_x_out,body_x_out = self.p_sample_loop(body_x_start, text_cond,text_padding=text_padding,body_padding=body_padding)
        # BS X T X D
        self.text_denoise_fn.train()
        self.body_denoise_fn.train() 
        self.style_transfer_fn.train()
        return text_x_out,body_x_out
    
    
    @torch.no_grad()
    def sample_betweening(self, x_start, cond_mask, between_length,padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        sample_res = self.p_sample_loop_betweening(x_start.shape, \
                x_start, cond_mask,between_length)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res 
    
    @torch.no_grad()
    def sample_continue(self, x_start, keep_length,padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval()
        cond_mask = torch.ones_like(x_start).to(x_start.device) 
        sample_res = self.p_sample_loop_continue(x_start.shape, \
                x_start, cond_mask,keep_length)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res 

    @torch.no_grad()
    def sample_sliding_window(self, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    @torch.no_grad()
    def sample_sliding_window_w_canonical(self, ds, global_head_jpos, global_head_jquat, x_start, cond_mask):
        # If the sequence is longer than trained max window, divide 
        self.denoise_fn.eval()
        sample_res = self.p_sample_loop_sliding_window_w_canonical(ds, x_start.shape, \
                global_head_jpos, global_head_jquat, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    
    def p_losses(self, text_t,body_x_start,body_t, body_noise_hand = None,text_noise_hand=None, text_padding_mask=None,body_padding_mask=None,text_cond = None,train_style = False,clean_wist_handmotion=None):
        # x_start: BS X T X 52*3+52*6
        # cond_mask: BS X T X D, missing regions are 1, head pose conditioned regions are 0.  
        
        text_x_start_hand = clean_wist_handmotion
        # text_x_start_wist = torch.zeros(())
        
        text_noise_hand = default(text_noise_hand, lambda: torch.randn_like(text_x_start_hand))
        
        text_x_hand = self.q_sample(x_start=text_x_start_hand, t=text_t, noise=text_noise_hand) # noisy motion in noise level t.  noised hand

        body_x_start_body = body_x_start.clone()
        body_x_start_body = torch.cat((body_x_start_body[:,:,0:22*3],body_x_start_body[:,:,52*3:52*3+22*6]),dim=2)
        body_x_start_hand = body_x_start.clone()
        body_x_start_hand = torch.cat((body_x_start_hand[:,:,22*3:52*3],body_x_start_hand[:,:,52*3+22*6:]),dim=2)
        
        body_noise_hand = default(body_noise_hand, lambda: torch.randn_like(body_x_start_hand))
        
        body_x_hand = self.q_sample(x_start=body_x_start_hand, t=body_t, noise=body_noise_hand) # noisy motion in noise level t.  noised hand
        
        if train_style:
            self.text_denoise_fn.eval()
            self.body_denoise_fn.eval() 
        else:
            self.text_denoise_fn.train()
            self.body_denoise_fn.train()    
        
        text_x_all = text_x_hand
        text_model_out = self.text_denoise_fn(src=text_x_all, noise_t=text_t, padding_mask=text_padding_mask,cond=text_cond)
        
        
        body_x_all = torch.cat((body_x_hand, body_x_start_body), dim=-1)
        body_model_out = self.body_denoise_fn(src=body_x_all, noise_t=body_t, padding_mask=body_padding_mask,cond=None)
        
        
        if not train_style:
            if self.objective == 'pred_noise':
                text_target = text_noise_hand
                body_target = body_noise_hand
            elif self.objective == 'pred_x0':
                text_target = text_x_start_hand
                body_target = body_x_start_hand
            else:
                raise ValueError(f'unknown objective {self.objective}')

            # Predicting both head pose and other joints' pose. 
            if (text_padding_mask is not None) and (body_padding_mask is not None):
                texthand_loss = self.loss_fn(text_model_out, text_target, reduction = 'none') * text_padding_mask[:, 0, 1:][:, :, None]
                bodyhand_loss = self.loss_fn(body_model_out, body_target, reduction = 'none') * body_padding_mask[:, 0, 1:][:, :, None]
            else:
                texthand_loss = self.loss_fn(text_model_out, text_target, reduction = 'none') 
                bodyhand_loss = self.loss_fn(body_model_out, body_target, reduction = 'none') 
            
            texthand_loss = reduce(texthand_loss, 'b ... -> b (...)', 'mean')
            bodyhand_loss = reduce(bodyhand_loss, 'b ... -> b (...)', 'mean')

            texthand_loss = (texthand_loss * extract(self.p2_loss_weight, text_t, texthand_loss.shape)).mean()
            bodyhand_loss = (bodyhand_loss * extract(self.p2_loss_weight, body_t, bodyhand_loss.shape)).mean()
        else:
            texthand_loss =0
            bodyhand_loss=0
        
        
        
        
        return texthand_loss+bodyhand_loss, text_model_out,body_model_out
    
    
    def style_forward(self,train_style,text_hand,body_hand):
        if train_style:
            body_hand = body_hand.detach()
            text_hand = text_hand.detach()
            style_model_out = self.style_transfer_fn.forward_style(src=body_hand,style=text_hand,padding_mask=None)
        else:
            style_model_out = None
            
        return style_model_out
        

    def forward(self,text_cond,body_x_start,text_padding_mask=None,body_padding_mask=None,train_style = False,clean_wist_handmotion=None):
        # x_start: BS X T X D 
        # cond_mask: BS X T X D 
        bs = body_x_start.shape[0] 
        text_t = torch.randint(0, self.num_timesteps, (bs,), device=body_x_start.device).long()
        body_t = torch.randint(0, self.num_timesteps, (bs,), device=body_x_start.device).long()
        # print("t:{0}".format(t))
        curr_loss,text_model_out,body_model_out = self.p_losses(text_t=text_t,body_x_start=body_x_start,body_t=body_t, 
                                            text_padding_mask=text_padding_mask,body_padding_mask=body_padding_mask,text_cond=text_cond,train_style=train_style,clean_wist_handmotion = clean_wist_handmotion)
        
        return curr_loss,text_model_out,body_model_out
        