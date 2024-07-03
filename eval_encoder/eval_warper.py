import numpy as np
import torch
import torch.nn as nn
import clip
import os
from scipy import linalg

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device,dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.device=device
        
        self.dropout = nn.Dropout(p=dropout).cuda(device)

        pe = torch.zeros(max_len, d_model).cuda(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).cuda(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).cuda(device)
        pe[:, 0::2] = torch.sin(position * div_term).cuda(device)
        pe[:, 1::2] = torch.cos(position * div_term).cuda(device)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[1], :].unsqueeze(0)
        return self.dropout(x)

class MotionEncoder(nn.Module):
    def __init__(self,input_dim,latent_dim,ff_size,num_layers,num_heads,dropout,activation,device):
        super(MotionEncoder,self).__init__()
        self.device=device
        
        self.input_feats = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim)).cuda(device)

        self.embed_motion = nn.Linear(self.input_feats, self.latent_dim).cuda(device)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, device,self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True).cuda(device)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers).cuda(device)
        self.out_ln = nn.LayerNorm(self.latent_dim).cuda(device)
        self.out = nn.Linear(self.latent_dim, latent_dim).cuda(device)


    def forward(self, x):
        B, T, D  = x.shape

        x = x.reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)


        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        return motion_emb

loss_ce = nn.CrossEntropyLoss()
class Eval_Encoder(nn.Module):
    def __init__(self,device):
        super(Eval_Encoder, self).__init__()
        self.device = device
        
        self.hand_motion_encoder = MotionEncoder(input_dim=270,latent_dim=512,ff_size=1024,num_layers=4,num_heads=4,dropout=0.1,activation='gelu',device=device)
        
        self.body_motion_encoder = MotionEncoder(input_dim=198,latent_dim=512,ff_size=1024,num_layers=4,num_heads=4,dropout=0.1,activation='gelu',device=device)
        
        self.full_motion_encoder = MotionEncoder(input_dim=468,latent_dim=512,ff_size=1024,num_layers=4,num_heads=4,dropout=0.1,activation='gelu',device=device)
        self.latent_dim = 512
        
        self.combine_cond = nn.Linear(2*self.latent_dim, self.latent_dim).cuda(device)
        
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1])).cuda(device)

        set_requires_grad(self.token_embedding, False)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer,
            num_layers=8)
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, self.latent_dim).cuda(device)

        self.clip_training = "text_"
        self.l1_criterion = torch.nn.L1Loss(reduction='mean')
    
    def encode_embedtext(self, embed_text):
        embed_text = self.out(torch.from_numpy(embed_text).to(self.device)).to(self.device)

        embed_text = (embed_text / embed_text.norm(dim=-1, keepdim=True) * self.latent_scale).squeeze(dim=1)
        return embed_text
        
    def forward(self, embed_text,body_motion,hand_motion,full_motion):
        
        embed_text = self.out(torch.from_numpy(embed_text).to(body_motion.device)).to(body_motion.device)

        embed_text = (embed_text / embed_text.norm(dim=-1, keepdim=True) * self.latent_scale).squeeze(dim=1)
        
        embed_full_motion = self.encode_fullmotion(full_motion)
        mixed_clip_loss = self.compute_clip_losses(embed_text,embed_full_motion)

        return mixed_clip_loss
    
    def encode_text(self, text,motiondevice):
        device = next(self.parameters()).device
        raw_text = text

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        out = self.textTransEncoder(pe_tokens)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out).to(motiondevice)

        out = out / out.norm(dim=-1, keepdim=True) * self.latent_scale

        return out
    
    def encode_fullmotion(self, full_motion):
        embed_full_motion = self.full_motion_encoder(full_motion)
        embed_full_motion = embed_full_motion / embed_full_motion.norm(dim=-1, keepdim=True) * self.latent_scale

        return embed_full_motion
    
    def encode_bodymotion(self, body_motion):
        embed_body_motion = self.body_motion_encoder(body_motion)
        embed_body_motion = embed_body_motion / embed_body_motion.norm(dim=-1, keepdim=True) * self.latent_scale

        return embed_body_motion
    
    def encode_handmotion(self,hand_motion):
        embed_hand_motion = self.hand_motion_encoder(hand_motion)
        embed_hand_motion = embed_hand_motion / embed_hand_motion.norm(dim=-1, keepdim=True) * self.latent_scale

        return embed_hand_motion
    
    
    def compute_clip_losses(self, embed_cond,embed_hand_motion):
        mixed_clip_loss = 0.

        features = embed_cond
        motion_features = embed_hand_motion
        # normalized features
        features_norm = features / features.norm(dim=-1, keepdim=True)
        motion_features_norm = motion_features / motion_features.norm(dim=-1, keepdim=True)

        logit_scale = self.latent_scale ** 2
        logits_per_motion = logit_scale * motion_features_norm @ features_norm.t()
        logits_per_d = logits_per_motion.t()

        batch_size = motion_features.shape[0]
        ground_truth = torch.arange(batch_size, dtype=torch.long, device=motion_features.device)

        ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
        ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
        clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

        mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_fid(act1, act2, eps=1e-6):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    sigma1 += np.eye(sigma1.shape[0]) * eps
    sigma2 += np.eye(sigma2.shape[0]) * eps
    ssdiff = np.sum(np.square(mu1 - mu2)) 
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        covmean = linalg.sqrtm((sigma1 + eps).dot(sigma2 + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0*covmean)
    print("fid: "+str(fid) )
    return fid

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        correct_vec = (correct_vec | bool_mat[:, i])
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat

def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]
    activation = activation * 6
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm((activation[first_indices] - activation[second_indices])/2, axis=1)
    return dist.mean()

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def evaluate_diversity(activation):
    diversity = calculate_diversity(activation, 300)
    return diversity

def evaluate_multimodality(activation):
    diversity = calculate_multimodality(activation, 10)
    return diversity

def evaulateforours(encoder_path,results_folder,global_device,dataset):
    batch_list = []
    npy_folder = results_folder
    #############################################loadbatch
    for data_name in os.listdir(npy_folder):
        test_npy_path = os.path.join(npy_folder,data_name)
        data_ = np.load(test_npy_path, allow_pickle=True).item()
        if data_["val_motion"].shape[0]!=32:
            continue
        gt_fullmotion = torch.from_numpy(data_["val_motion"]).to(global_device)
        out_hand = torch.from_numpy(data_["style_out"]).to(global_device)
        body_hand = torch.from_numpy(data_["bodymodel_out"]).to(global_device)
        text_hand = torch.from_numpy(data_["textmodel_out"]).to(global_device)
        raw_text = data_["raw_text"]
        batch={}
        batch["gt_globalfull"] = gt_fullmotion
        batch["out_globalfull"] = torch.cat((gt_fullmotion[:,:,:22*3],out_hand[:,:,:30*3],gt_fullmotion[:,:,52*3:52*3+22*6],out_hand[:,:,30*3:]),dim=-1)
        batch["body_globalfull"] = torch.cat((gt_fullmotion[:,:,:22*3],body_hand[:,:,:30*3],gt_fullmotion[:,:,52*3:52*3+22*6],body_hand[:,:,30*3:]),dim=-1)
        batch["text_globalfull"] = torch.cat((gt_fullmotion[:,:,:22*3],text_hand[:,:,:30*3],gt_fullmotion[:,:,52*3:52*3+22*6],text_hand[:,:,30*3:]),dim=-1)
        batch["out_globalfull"][:, :, :52*3] = dataset.de_normalize_jpos_min_max(batch["out_globalfull"][:, :, :52*3].reshape(-1,52,3)).reshape(-1,150,52*3)
        batch["gt_globalfull"][:, :, :52*3] = dataset.de_normalize_jpos_min_max(batch["gt_globalfull"][:, :, :52*3].reshape(-1,52,3)).reshape(-1,150,52*3)
        batch["body_globalfull"][:, :, :52*3] = dataset.de_normalize_jpos_min_max(batch["body_globalfull"][:, :, :52*3].reshape(-1,52,3)).reshape(-1,150,52*3)
        batch["text_globalfull"][:, :, :52*3] = dataset.de_normalize_jpos_min_max(batch["text_globalfull"][:, :, :52*3].reshape(-1,52,3)).reshape(-1,150,52*3)
        batch["raw_text"] = raw_text
        batch["embed_text"] = data_["embed_text"]
        batch_list.append(batch)

    test_eval = Eval_Encoder(global_device)
    test_eval.load_state_dict(torch.load(encoder_path))

    embed_motion_list,embed_motion_gt_list,embed_text_list,embed_motion_body_list,embed_motion_text_list=[],[],[],[],[]
    all_motion_embeddings,all_motion_embeddings_gt,all_motion_embeddings_body,all_motion_embeddings_text=[],[],[],[]
    mm_dist_sum,mm_dist_sum_gt,mm_dist_sum_body,mm_dist_sum_text=0,0,0,0
    all_size,all_size_gt,all_size_body,all_size_text=0,0,0,0
    top_k_count,top_k_count_gt,top_k_count_body,top_k_count_text=0,0,0,0

    for batch in batch_list:
        embed_motion_gt = test_eval.encode_fullmotion(batch["gt_globalfull"]).detach().cpu().numpy()
        embed_text = test_eval.encode_embedtext(batch["embed_text"]).detach().cpu().numpy()
        embed_motion = test_eval.encode_fullmotion(batch["out_globalfull"]).detach().cpu().numpy()
        embed_motion_body = test_eval.encode_fullmotion(batch["body_globalfull"]).detach().cpu().numpy()
        embed_motion_text = test_eval.encode_fullmotion(batch["text_globalfull"]).detach().cpu().numpy()
        
        gt_dist_mat = euclidean_distance_matrix(embed_text,embed_motion_gt)
        mm_dist_sum_gt += gt_dist_mat.trace()
        argsmax_gt = np.argsort(gt_dist_mat, axis=1)
        top_k_mat_gt = calculate_top_k(argsmax_gt, top_k=3)
        top_k_count_gt += top_k_mat_gt.sum(axis=0)
        all_size_gt += embed_motion_gt.shape[0]
        all_motion_embeddings_gt.append(embed_motion_gt)
        
        dist_mat = euclidean_distance_matrix(embed_text,embed_motion)
        mm_dist_sum += dist_mat.trace()
        argsmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        top_k_count += top_k_mat.sum(axis=0)
        all_size += embed_motion.shape[0]
        all_motion_embeddings.append(embed_motion)

        body_dist_mat = euclidean_distance_matrix(embed_text,embed_motion_body)
        mm_dist_sum_body += body_dist_mat.trace()
        argsmax_body = np.argsort(body_dist_mat, axis=1)
        top_k_mat_body = calculate_top_k(argsmax_body, top_k=3)
        top_k_count_body += top_k_mat_body.sum(axis=0)
        all_size_body += embed_motion_body.shape[0]
        all_motion_embeddings_body.append(embed_motion_body)

        text_dist_mat = euclidean_distance_matrix(embed_text,embed_motion_text)
        mm_dist_sum_text += text_dist_mat.trace()
        argsmax_text = np.argsort(text_dist_mat, axis=1)
        top_k_mat_text = calculate_top_k(argsmax_text, top_k=3)
        top_k_count_text += top_k_mat_text.sum(axis=0)
        all_size_text += embed_motion_text.shape[0]
        all_motion_embeddings_text.append(embed_motion_text)
        
        embed_motion_gt_list.append(embed_motion_gt)
        embed_motion_list.append(embed_motion)
        embed_text_list.append(embed_text)
        embed_motion_body_list.append(embed_motion_body)
        embed_motion_text_list.append(embed_motion_text)

    
    print("---BOTH2Hands---")
    all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
    mm_dist = mm_dist_sum / all_size
    R_precision = top_k_count / all_size
    MM_Distance = mm_dist
    R_precision = R_precision
    print("R_p ")
    print(R_precision)
    print("mm_dist ")
    print(MM_Distance)
    activation= all_motion_embeddings
    embed_motion_gt_list = np.concatenate(embed_motion_gt_list,axis=0)
    embed_motion_list = np.concatenate(embed_motion_list,axis=0)
    embed_motion_body_list = np.concatenate(embed_motion_body_list,axis=0)
    embed_motion_text_list = np.concatenate(embed_motion_text_list,axis=0)
    embed_text_list = np.concatenate(embed_text_list,axis=0)
    fid_ours = calculate_fid(embed_motion_list,embed_text_list)
    print("diversity ")
    diversity = evaluate_diversity(activation)
    print(diversity)
    print("multimodality ")
    multimodality = evaluate_multimodality(activation.reshape(-1,32,512))
    print(multimodality)
    ours_numbers={}
    ours_numbers["R_precision"] = R_precision
    ours_numbers["MM_Distance"] = MM_Distance
    ours_numbers["FID"] = fid_ours
    ours_numbers["Diversity"] = diversity
    ours_numbers["Multimodality"] = multimodality
    
    print("---gt---")
    all_motion_embeddings_gt = np.concatenate(all_motion_embeddings_gt, axis=0)
    mm_dist_gt = mm_dist_sum_gt / all_size_gt
    R_precision_gt = top_k_count_gt / all_size_gt
    MM_Distance_gt = mm_dist_gt
    print("R_p ")
    print(R_precision_gt)
    print("mm_dist ")
    print(MM_Distance_gt)
    activation_gt= all_motion_embeddings_gt
    fid_gt = calculate_fid(embed_motion_gt_list,embed_text_list)
    print("diversity ")
    diversity_gt = evaluate_diversity(activation_gt)
    print(diversity_gt)
    print("multimodality ")
    multimodality_gt = evaluate_multimodality(activation_gt.reshape(-1,32,512))
    print(multimodality_gt)
    gt_numbers={}
    gt_numbers["R_precision"] = R_precision_gt
    gt_numbers["MM_Distance"] = MM_Distance_gt
    gt_numbers["FID"] = fid_gt
    gt_numbers["Diversity"] = diversity_gt
    gt_numbers["Multimodality"] = multimodality_gt
    
    return ours_numbers,gt_numbers


def evaulation(npy_folder,dataloader,encoder_path = "eval_encoder/encoder.pth"):
    ours_top1=[]
    ours_top2=[]
    ours_top3=[]
    ours_fid=[]
    ours_mmdist=[]
    ours_dicersity=[]
    ours_mmodality=[]

    gt_top1=[]
    gt_top2=[]
    gt_top3=[]
    gt_fid=[]
    gt_mmdist=[]
    gt_dicersity=[]
    gt_mmodality=[]

    all_ours_dixt = {}

    for npys in os.listdir(npy_folder):
        npys_path = os.path.join(npy_folder,npys)
        ours_numbers,gt_numbers =  evaulateforours(encoder_path,npys_path,global_device = 0,dataset=dataloader)
        ours_top1.append(ours_numbers["R_precision"][0])
        ours_top2.append(ours_numbers["R_precision"][1])
        ours_top3.append(ours_numbers["R_precision"][2])
        ours_fid.append(ours_numbers["FID"])
        ours_mmdist.append(ours_numbers["MM_Distance"])
        ours_dicersity.append(ours_numbers["Diversity"])
        ours_mmodality.append(ours_numbers["Multimodality"])
        
        gt_top1.append(gt_numbers["R_precision"][0])
        gt_top2.append(gt_numbers["R_precision"][1])
        gt_top3.append(gt_numbers["R_precision"][2])
        gt_fid.append(gt_numbers["FID"])
        gt_mmdist.append(gt_numbers["MM_Distance"])
        gt_dicersity.append(gt_numbers["Diversity"])
        gt_mmodality.append(gt_numbers["Multimodality"])
        


        
    all_ours_dixt["ours_top1"]=ours_top1
    all_ours_dixt["ours_top2"]=ours_top2
    all_ours_dixt["ours_top3"]=ours_top3
    all_ours_dixt["ours_fid"]=ours_fid
    all_ours_dixt["ours_mmdist"]=ours_mmdist
    all_ours_dixt["ours_dicersity"]=ours_dicersity
    all_ours_dixt["ours_mmodality"]=ours_mmodality

    all_ours_dixt["gt_top1"]=gt_top1
    all_ours_dixt["gt_top2"]=gt_top2
    all_ours_dixt["gt_top3"]=gt_top3
    all_ours_dixt["gt_fid"]=gt_fid
    all_ours_dixt["gt_mmdist"]=gt_mmdist
    all_ours_dixt["gt_dicersity"]=gt_dicersity
    all_ours_dixt["gt_mmodality"]=gt_mmodality

    for key,value in all_ours_dixt.items():
        print(key)
        print("mean")
        print(np.mean(value))
        print("std")
        print(np.std(value))
