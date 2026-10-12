import os


os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
import json
import math
import signal
import random
import copy
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    FLAIR_CONFIG, MVCINN_CONFIG, FUSION_CONFIG, TRAIN_CONFIG,
    RESUME_CONFIG,
    FLAIR_WEIGHTS, MVCINN_WEIGHTS, ensure_dirs,
    CHECKPOINT_DIR, LOG_DIR, BEST_WEIGHT_PATH, RESUME_CHECKPOINT_PATH,
    get_seed_weight_paths,
)
from data.unified_dataset import build_data_loaders
from models.flair_wrapper import FLAIRWrapper
from models.mvcinn_wrapper import MVCINNWrapper
from models.fusion_module import FusionModel
from utils.metrics import compute_all_metrics, format_metrics_str
from utils.visualization import plot_training_curves
from utils.checkpoint import (
    TrainingCheckpoint, BestWeightManager,
    build_training_state, restore_training_state, print_resume_info,
)




_interrupted = False


def _signal_handler(signum, frame):

    global _interrupted
    _interrupted = True
    print("\n检测到中断信号，将在当前 epoch 结束后安全退出...")


def seed_everything(seed=1001):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0, reduction='mean'):

        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, targets):

        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha,
            label_smoothing=self.label_smoothing, reduction='none',
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def ordinal_loss(logits, labels, num_classes=5):

    probs = F.softmax(logits, dim=1)
    class_indices = torch.arange(num_classes, device=labels.device).float()
    mean_class = (probs * class_indices).sum(dim=1)
    return F.mse_loss(mean_class, labels.float())


def _diagnose_forward_nan(model, data_loader, device):

    model.eval()
    try:
        batch = next(iter(data_loader))
    except StopIteration:
        return None

    flair_input = batch["flair_input"].to(device)
    mvcinn_input = batch["mvcinn_input"].to(device)

    with torch.no_grad():
        
        mvcinn_output = model.mvcinn(mvcinn_input, return_multi_scale=True)
        conv_logits = mvcinn_output["conv_logits"]
        trans_logits = mvcinn_output["trans_logits"]
        trans_features = mvcinn_output["trans_features"]
        multi_scale = mvcinn_output.get("multi_scale_features")
        for name, logits in [("conv_logits", conv_logits), ("trans_logits", trans_logits)]:
            if not torch.isfinite(logits).all():
                print(f"  [NaN] MVCINN {name}")
                return f"MVCINN.{name}"
        if not torch.isfinite(trans_features).all():
            print("  [NaN] MVCINN trans_features")
            return "MVCINN.trans_features"
        
        if multi_scale is not None:
            for i, ms in enumerate(multi_scale):
                if ms is not None and not torch.isfinite(ms).all():
                    print(f"  [NaN] MVCINN multi_scale[{i}]")
                    return f"MVCINN.multi_scale_{i}"

        
        flair_output = model.flair(flair_input)
        vis_embed = flair_output["vis_embed"]  
        if not torch.isfinite(vis_embed).all():
            print("  [NaN] FLAIR visual features (vis_embed)")
            return "FLAIR.vision_model"

        
        class_embeds = flair_output["class_text_embeds"]
        if not torch.isfinite(class_embeds).all():
            print("  [NaN] FLAIR text embeddings (class_embeds)")
            return "FLAIR.text_model"

        zero_shot_logits = flair_output["zero_shot_logits"]
        if not torch.isfinite(zero_shot_logits).all():
            print("  [NaN] FLAIR zero_shot_logits")
            return "FLAIR.zero_shot_logits"

        
        B = mvcinn_input.size(0)

        
        vis_embed_norm = F.normalize(vis_embed, dim=-1)
        flair_logits = F.linear(vis_embed_norm, class_embeds) * model.flair_scale + model.flair_bias
        if not torch.isfinite(flair_logits).all():
            print("  [NaN] flair_classifier output")
            return "flair_classifier"

        
        if model.fusion_module is not None and trans_features is not None:
            
            for v in range(model.num_view):
                v_feat = trans_features[v::model.num_view]
                fused_feat, attn_w = model.fusion_module(v_feat, class_embeds, vis_embed)
                if not torch.isfinite(fused_feat).all():
                    print(f"  [NaN] CrossModalAttention output (view={v})")
                    return "CrossModalAttention"

            
            fused_cls_list = []
            for v in range(model.num_view):
                v_feat = trans_features[v::model.num_view]
                fused_feat, _ = model.fusion_module(v_feat, class_embeds, vis_embed)
                cls_pooled = fused_feat[:, 0]  
                if not torch.isfinite(cls_pooled).all():
                    print(f"  [NaN] CLS token pooling output (view={v})")
                    return "CLS_pooling"
                fused_cls_list.append(cls_pooled)

            
            view_cls = torch.stack(fused_cls_list, dim=1)
            if model.view_transformer is not None:
                fused_cls, _ = model.view_transformer(view_cls)
            else:
                fused_cls = view_cls.mean(1)
            fused_cls_refined = fused_cls

            
            deep_fusion_logits = None
            if model.feature_fusion is not None:
                deep_fused_feat = model.feature_fusion(trans_features, vis_embed)
                if not torch.isfinite(deep_fused_feat).all():
                    print("  [NaN] FeatureLevelFusion output")
                    return "FeatureLevelFusion"
                deep_fusion_logits = model.deep_fusion_head(deep_fused_feat)
                if not torch.isfinite(deep_fusion_logits).all():
                    print("  [NaN] deep_fusion_head output")
                    return "deep_fusion_head"

            
            vis_proj_feat = model.vis_proj_norm(model.vis_proj(vis_embed))
            concat_feat = torch.cat([fused_cls_refined, vis_proj_feat], dim=-1)
            concat_feat = model.residual_norm(concat_feat)
            fusion_logits = model.residual_head(concat_feat)
            if not torch.isfinite(fusion_logits).all():
                print("  [NaN] residual_head output")
                return "residual_head"

            
            conv_trans_cross_logits = None
            if multi_scale is not None and len(multi_scale) >= 3:
                conv_1_feat, conv_4_feat, conv_8_feat = multi_scale
                conv_trans_cross_logits, _ = model.conv_trans_cross(
                    conv_8_feat, trans_features,
                )
                if not torch.isfinite(conv_trans_cross_logits).all():
                    print("  [NaN] ConvTransCross output")
                    return "ConvTransCross"

            
            
            
            if deep_fusion_logits is None:
                deep_fusion_logits = fusion_logits  
            fused_logits, conf_weights = model.conf_fusion(
                flair_logits, fusion_logits, deep_fusion_logits,
            )
            if not torch.isfinite(fused_logits).all():
                print("  [NaN] conf_fusion output")
                return "conf_fusion"

    model.train()
    return None


def _validate_model_weights(model):

    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            print(f"  [NaN 检测] 参数 {name} 包含 NaN 或 Inf")
            return True
    return False


def build_model(device):

    print("=" * 60)
    print("=" * 60)

    
    load_pretrained = TRAIN_CONFIG.get("load_pretrained", True)
    flair_weights = FLAIR_WEIGHTS if load_pretrained and os.path.exists(FLAIR_WEIGHTS) else None
    flair_wrapper = FLAIRWrapper(
        weights_path=flair_weights,
        freeze_vision=False,  
        freeze_text=True,     
        proj_dim=FLAIR_CONFIG["proj_dim"],
        image_size=FLAIR_CONFIG["image_size"],
        multi_view=FLAIR_CONFIG.get("multi_view", True),       
        num_view=FLAIR_CONFIG.get("num_view", 4),
        coop_enabled=FLAIR_CONFIG.get("coop_enabled", True),   
        coop_init_std=FLAIR_CONFIG.get("coop_init_std", 0.02),
    )

    
    mvcinn_weights = MVCINN_WEIGHTS if load_pretrained and os.path.exists(MVCINN_WEIGHTS) else None
    mvcinn_wrapper = MVCINNWrapper(
        weights_path=mvcinn_weights,
        num_classes=MVCINN_CONFIG["num_classes"],
        depth=MVCINN_CONFIG["depth"],
        num_heads=MVCINN_CONFIG["num_heads"],
        drop_rate=MVCINN_CONFIG["drop_rate"],
        drop_path_rate=MVCINN_CONFIG["drop_path_rate"],
        freeze_backbone=False,  
    )

    
    fusion_model = FusionModel(
        flair_wrapper=flair_wrapper,
        mvcinn_wrapper=mvcinn_wrapper,
        text_dim=FUSION_CONFIG["text_dim"],
        vis_dim=FUSION_CONFIG["vis_dim"],
        flair_vis_dim=FUSION_CONFIG["flair_vis_dim"],
        num_heads=FUSION_CONFIG["num_heads"],
        num_classes=MVCINN_CONFIG["num_classes"],
        fusion_dropout=FUSION_CONFIG["fusion_dropout"],
        use_cross_attention=FUSION_CONFIG["use_cross_attention"],
        
        use_view_attention=FUSION_CONFIG.get("use_view_attention", True),
        view_attn_layers=FUSION_CONFIG.get("view_attn_layers", 2),
        view_attn_heads=FUSION_CONFIG.get("view_attn_heads", 8),
        view_attn_dropout=FUSION_CONFIG.get("view_attn_dropout", 0.1),
        
        fusion_head_dropout1=FUSION_CONFIG.get("fusion_head_dropout1", 0.2),
        fusion_head_dropout2=FUSION_CONFIG.get("fusion_head_dropout2", 0.1),
        
        pcol_lambda=FUSION_CONFIG.get("pcol_lambda", 0.1),
    ).to(device)

    return fusion_model


def build_optimizer(model):

    
    lr_unified = TRAIN_CONFIG.get("lr_unified", 1e-5)

    
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    
    total_trainable = sum(p.numel() for p in trainable_params)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr_unified,
        weight_decay=TRAIN_CONFIG["weight_decay"],
        eps=1e-4,
    )
    return optimizer


def build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs,
                                  T_0=4, T_mult=2):

    from torch.optim.lr_scheduler import SequentialLR

    warmup_epochs = max(warmup_epochs, 0)

    
    def warmup_lambda(epoch):

        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, warmup_lambda)
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=0,
    )

    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler


def mixup_data(images_flair, images_mvcinn, labels, alpha=0.2):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(labels.size(0)).to(labels.device)
    
    mixed_flair = lam * images_flair + (1 - lam) * images_flair[index]

    
    if lam >= 0.5:
        
        mixed_mvcinn = lam * images_mvcinn + (1 - lam) * images_mvcinn[index]
    else:
        
        V = images_mvcinn.shape[1]
        lam_views = np.clip(np.random.beta(alpha, alpha, size=(V,)), 0.1, 0.9).astype(np.float32)
        lam_views_t = torch.from_numpy(lam_views).to(images_mvcinn.device)
        lam_views_t = lam_views_t.view(1, V, 1, 1, 1)
        mixed_mvcinn = lam_views_t * images_mvcinn + (1 - lam_views_t) * images_mvcinn[index]

    return mixed_flair, mixed_mvcinn, labels, labels[index], lam


def cutmix_data(images_flair, images_mvcinn, labels, alpha=1.0):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(labels.size(0)).to(labels.device)

    
    B, V, C, H_m, W_m = images_mvcinn.shape
    _, _, _, H_f, W_f = images_flair.shape

    
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h_m = int(H_m * cut_ratio)
    cut_w_m = int(W_m * cut_ratio)
    cut_h_f = int(H_f * cut_ratio)
    cut_w_f = int(W_f * cut_ratio)

    
    cy_m = np.random.randint(cut_h_m // 2, H_m - cut_h_m // 2 + 1)
    cx_m = np.random.randint(cut_w_m // 2, W_m - cut_w_m // 2 + 1)
    cy_f = int(cy_m * H_f / H_m)
    cx_f = int(cx_m * W_f / W_m)

    
    mixed_mvcinn = images_mvcinn.clone()
    mixed_flair = images_flair.clone()

    y1_m, y2_m = cy_m - cut_h_m // 2, cy_m + cut_h_m // 2
    x1_m, x2_m = cx_m - cut_w_m // 2, cx_m + cut_w_m // 2
    y1_f, y2_f = cy_f - cut_h_f // 2, cy_f + cut_h_f // 2
    x1_f, x2_f = cx_f - cut_w_f // 2, cx_f + cut_w_f // 2

    mixed_mvcinn[:, :, :, y1_m:y2_m, x1_m:x2_m] = images_mvcinn[index, :, :, y1_m:y2_m, x1_m:x2_m]
    mixed_flair[:, :, :, y1_f:y2_f, x1_f:x2_f] = images_flair[index, :, :, y1_f:y2_f, x1_f:x2_f]

    
    lam = 1.0 - (cut_h_m * cut_w_m) / (H_m * W_m)

    return mixed_flair, mixed_mvcinn, labels, labels[index], lam


def unwrap_model(model):

    return model.module if isinstance(model, nn.DataParallel) else model


class ModelEMA:


    def __init__(self, model, decay=0.999):

        self.decay = decay
        
        inner = unwrap_model(model)
        self.state = {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}

    @torch.no_grad()
    def update(self, model):

        inner = unwrap_model(model)
        model_state = inner.state_dict()
        for k, v in self.state.items():
            if k not in model_state:
                continue
            model_v = model_state[k].detach().cpu()
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(model_v, alpha=1 - self.decay)
            else:
                
                v.copy_(model_v)

    @torch.no_grad()
    def apply_to_model(self, model):

        inner = unwrap_model(model)
        backup = {k: v.detach().cpu().clone() for k, v in inner.state_dict().items()}
        model_state = inner.state_dict()
        for k in model_state.keys():
            if k in self.state:
                model_state[k].copy_(self.state[k].to(model_state[k].device))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):

        inner = unwrap_model(model)
        model_state = inner.state_dict()
        for k in model_state.keys():
            if k in backup:
                model_state[k].copy_(backup[k].to(model_state[k].device))


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device,
                    epoch=0, total_epochs=1, amp=True, scaler=None,
                    aux_weight=0.5, class_weights=None, pcol_weight=0.5, ema=None):


    global _interrupted
    model.train()
    if scaler is None and amp:
        scaler = GradScaler('cuda')

    
    model_inner = unwrap_model(model)

    

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    nan_count = 0  
    pcol_sum = 0.0  
    pcol_count = 0

    pbar = tqdm(loader, desc=f"[单阶段训练] Epoch {epoch+1}")
    for batch_idx, batch in enumerate(pbar):
        if _interrupted:
            break

        flair_input = batch["flair_input"].to(device, non_blocking=True)
        mvcinn_input = batch["mvcinn_input"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        
        cutmix_prob = TRAIN_CONFIG.get("cutmix_prob", 0.0)
        cutmix_alpha = TRAIN_CONFIG.get("cutmix_alpha", 1.0)
        use_cutmix = (cutmix_prob > 0 and np.random.random() < cutmix_prob)

        if use_cutmix:
            flair_input, mvcinn_input, labels_a, labels_b, lam = cutmix_data(
                flair_input, mvcinn_input, labels, cutmix_alpha,
            )
        elif TRAIN_CONFIG["mixup_alpha"] > 0:
            flair_input, mvcinn_input, labels_a, labels_b, lam = mixup_data(
                flair_input, mvcinn_input, labels, TRAIN_CONFIG["mixup_alpha"],
            )
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        
        
        def compute_aux_loss(outputs):

            def _mixup_aux(logits_head):

                return (lam * criterion(logits_head, labels_a) +
                        (1 - lam) * criterion(logits_head, labels_b))
            aux = torch.tensor(0.0, device=labels_a.device)
            
            if outputs.get("trans_logits") is not None:
                aux = aux + 0.1 * _mixup_aux(outputs["trans_logits"])
            
            if outputs.get("flair_logits") is not None:
                aux = aux + 0.5 * _mixup_aux(outputs["flair_logits"])
            
            if outputs.get("fusion_logits") is not None:
                aux = aux + 0.5 * _mixup_aux(outputs["fusion_logits"])
            
            conv_aux_w = FUSION_CONFIG.get("conv_aux_weight", 0.1)
            if conv_aux_w > 0 and outputs.get("conv_logits") is not None:
                aux = aux + conv_aux_w * _mixup_aux(outputs["conv_logits"])
            
            deep_fusion_aux_w = FUSION_CONFIG.get("deep_fusion_aux_weight", 0.3)
            if deep_fusion_aux_w > 0 and outputs.get("deep_fusion_logits") is not None:
                aux = aux + deep_fusion_aux_w * _mixup_aux(outputs["deep_fusion_logits"])
            
            conv_trans_cross_aux_w = FUSION_CONFIG.get("conv_trans_cross_aux_weight", 0.3)
            if conv_trans_cross_aux_w > 0 and outputs.get("conv_trans_cross_logits") is not None:
                aux = aux + conv_trans_cross_aux_w * _mixup_aux(outputs["conv_trans_cross_logits"])
            return aux

        
        def compute_extra_loss(outputs):

            extra = torch.tensor(0.0, device=labels_a.device)

            
            ordinal_w = FUSION_CONFIG.get("ordinal_loss_lambda", 0.2)
            if ordinal_w > 0:
                fused_logits = outputs.get("fused_logits")
                if fused_logits is not None:
                    
                    ord_loss_a = ordinal_loss(fused_logits, labels_a, num_classes=5)
                    ord_loss_b = ordinal_loss(fused_logits, labels_b, num_classes=5)
                    extra = extra + ordinal_w * (lam * ord_loss_a + (1 - lam) * ord_loss_b)

            
            
            disagree_w = FUSION_CONFIG.get("disagree_lambda", 0.1)
            if disagree_w > 0:
                fused_logits_tensor = outputs.get("fused_logits")
                if fused_logits_tensor is not None:
                    fused_prob = F.softmax(fused_logits_tensor, dim=1)
                    fused_correct_prob = fused_prob.gather(1, labels_a.unsqueeze(1)).squeeze(1)
                    
                    branch_keys = ["flair_logits", "fusion_logits", "deep_fusion_logits"]
                    for bkey in branch_keys:
                        blogits = outputs.get(bkey)
                        if blogits is not None and isinstance(blogits, torch.Tensor):
                            bprob = F.softmax(blogits, dim=1)
                            b_correct_prob = bprob.gather(1, labels_a.unsqueeze(1)).squeeze(1)
                            
                            disagree = F.relu(b_correct_prob - fused_correct_prob).mean()
                            extra = extra + disagree_w * disagree

            return extra

        
        
        def compute_pcol_loss(outputs):

            fused_cls_refined = outputs.get("fused_cls_refined")
            if fused_cls_refined is None:
                return torch.tensor(0.0, device=labels_a.device)
            pcol_loss, _, _ = model_inner.pcol(
                fused_cls_refined, labels_a, class_weights,
            )
            return pcol_loss

        
        if amp and scaler is not None:
            with autocast('cuda'):
                outputs = model(flair_input, mvcinn_input)
                logits = outputs["fused_logits"]

                if batch_idx == 0 and epoch == 0:
                    print(f"\n  [诊断] logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                    print(f"  [诊断] conv_logits range: [{outputs['conv_logits'].min():.4f}, {outputs['conv_logits'].max():.4f}]")
                    print(f"  [诊断] trans_logits range: [{outputs['trans_logits'].min():.4f}, {outputs['trans_logits'].max():.4f}]")
                    print(f"  [诊断] flair_logits range: [{outputs['flair_logits'].min():.4f}, {outputs['flair_logits'].max():.4f}]")
                    if outputs.get("fusion_logits") is not None:
                        print(f"  [诊断] fusion_logits range: [{outputs['fusion_logits'].min():.4f}, {outputs['fusion_logits'].max():.4f}]")
                    print(f"  [诊断] labels range: [{labels_a.min()}, {labels_a.max()}]")

                
                loss = (lam * criterion(logits, labels_a) +
                        (1 - lam) * criterion(logits, labels_b))
                
                aux_loss = compute_aux_loss(outputs)
                
                pcol_loss = compute_pcol_loss(outputs)
                
                extra_loss = compute_extra_loss(outputs)
                loss = loss + aux_weight * aux_loss + pcol_weight * pcol_loss + extra_loss
            loss_val = loss.item()
            if not np.isfinite(loss_val):
                
                nan_count += 1
                outputs = model(flair_input, mvcinn_input)
                logits = outputs["fused_logits"]
                loss = (lam * criterion(logits, labels_a) +
                        (1 - lam) * criterion(logits, labels_b))
                aux_loss = compute_aux_loss(outputs)
                pcol_loss = compute_pcol_loss(outputs)
                extra_loss = compute_extra_loss(outputs)
                loss = loss + aux_weight * aux_loss + pcol_weight * pcol_loss + extra_loss
                loss_val = loss.item()

                if np.isfinite(loss_val):
                    
                    loss.backward()
                    
                    
                    grad_has_nan = False
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grad_has_nan = True
                            break
                    if grad_has_nan:
                        if nan_count <= 5:
                            print(f"\n  [警告] 检测到 NaN 梯度，跳过 optimizer.step() 防止权重污染")
                        optimizer.zero_grad()
                        pbar.set_postfix({
                            "loss": "nan",
                            "acc": f"{total_correct/max(total_samples,1):.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        })
                        continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    
                    if nan_count <= 5:
                        print(f"\n  [警告] 检测到 NaN loss (batch {nan_count})，跳过")
                    optimizer.zero_grad()
                    pbar.set_postfix({
                        "loss": "nan",
                        "acc": f"{total_correct/max(total_samples,1):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })
                    continue
            else:
                
                
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(flair_input, mvcinn_input)
            logits = outputs["fused_logits"]

            
            if batch_idx == 0 and epoch == 0:
                preds_diag = logits.argmax(dim=1)
                print(f"\n  [诊断-fp32] fused_logits range: [{logits.min():.4f}, {logits.max():.4f}]")
                print(f"  [诊断-fp32] fused_logits样本: {logits[0].tolist()}")
                print(f"  [诊断-fp32] preds分布: {torch.bincount(preds_diag, minlength=5).tolist()}")
                print(f"  [诊断-fp32] labels分布: {torch.bincount(labels_a, minlength=5).tolist()}")
                print(f"  [诊断-fp32] conv_logits range: [{outputs['conv_logits'].min():.4f}, {outputs['conv_logits'].max():.4f}]")
                print(f"  [诊断-fp32] flair_logits range: [{outputs['flair_logits'].min():.4f}, {outputs['flair_logits'].max():.4f}]")
                if outputs.get("fusion_logits") is not None:
                    print(f"  [诊断-fp32] fusion_logits range: [{outputs['fusion_logits'].min():.4f}, {outputs['fusion_logits'].max():.4f}]")

            loss = (lam * criterion(logits, labels_a) +
                    (1 - lam) * criterion(logits, labels_b))
            aux_loss = compute_aux_loss(outputs)
            pcol_loss = compute_pcol_loss(outputs)
            extra_loss = compute_extra_loss(outputs)
            loss = loss + aux_weight * aux_loss + pcol_weight * pcol_loss + extra_loss
            loss_val = loss.item()

            if np.isfinite(loss_val):
                loss.backward()
                
                grad_has_nan = False
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        grad_has_nan = True
                        break
                if grad_has_nan:
                    optimizer.zero_grad()
                    pbar.set_postfix({
                        "loss": "nan",
                        "acc": f"{total_correct/max(total_samples,1):.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    })
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            else:
                nan_count += 1
                optimizer.zero_grad()
                pbar.set_postfix({
                    "loss": "nan",
                    "acc": f"{total_correct/max(total_samples,1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })
                continue

        
        if ema is not None:
            ema.update(model)
        total_loss += loss_val
        
        pcol_sum += float(pcol_loss.detach().cpu())
        pcol_count += 1
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "acc": f"{total_correct/max(total_samples,1):.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })


    avg_loss = total_loss / max(len(loader) - nan_count, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    pcol_avg = pcol_sum / max(pcol_count, 1)
    return avg_loss, accuracy, scaler, pcol_avg


@torch.no_grad()
def validate(model, loader, criterion, device):

    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    branch_preds = {
        "conv": [], "trans": [], "flair": [],
        "fusion": [], "deep_fusion": [],
        "conv_trans_cross": [],  
    }
    
    monitor_stats = {
        "global_feat_norm": 0.0,
        "flair_scale_val": 0.0,
        "raw_flair_abs": 0.0,       
        "conv_logit_abs": 0.0,      
        "trans_logit_abs": 0.0,
        "flair_logit_abs": 0.0,
        "deep_fusion_logit_abs": 0.0,  
        "fusion_logit_abs": 0.0,
        "conv_trans_cross_logit_abs": 0.0,  
        "coop_prompt_norm": 0.0,    
        "view_weights": None,       
        "attn_dist": None,          
        "conf_weights": None,       
    }

    
    inner_model = model.module if hasattr(model, 'module') else model

    for batch in tqdm(loader, desc="验证中"):
        flair_input = batch["flair_input"].to(device, non_blocking=True)
        mvcinn_input = batch["mvcinn_input"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        outputs = inner_model(flair_input, mvcinn_input)
        logits = outputs["fused_logits"]

        loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        
        for branch in branch_preds.keys():
            key = f"{branch}_logits"
            if key in outputs and outputs[key] is not None:
                branch_pred = outputs[key].argmax(dim=1).cpu().numpy()
                branch_preds[branch].append(branch_pred)

        
        
        def _to_float(val):

            if isinstance(val, torch.Tensor):
                return val.mean().item()
            return val if val is not None else 0.0

        monitor_stats["global_feat_norm"] = _to_float(outputs.get("global_feat_norm", 0.0))
        monitor_stats["flair_scale_val"] = _to_float(outputs.get("flair_scale_val", 0.0))
        
        raw_flair = outputs.get("raw_flair_logits")
        if raw_flair is not None:
            monitor_stats["raw_flair_abs"] = raw_flair.abs().mean().item()
        
        monitor_stats["conv_logit_abs"] = _to_float(outputs.get("conv_logit_abs", 0.0))
        monitor_stats["trans_logit_abs"] = _to_float(outputs.get("trans_logit_abs", 0.0))
        monitor_stats["flair_logit_abs"] = _to_float(outputs.get("flair_logit_abs", 0.0))
        
        monitor_stats["deep_fusion_logit_abs"] = _to_float(outputs.get("deep_fusion_logit_abs", 0.0))
        monitor_stats["fusion_logit_abs"] = _to_float(outputs.get("fusion_logit_abs", 0.0))
        monitor_stats["fused_logit_abs"] = _to_float(outputs.get("fused_logit_abs", 0.0))
        
        monitor_stats["conv_trans_cross_logit_abs"] = _to_float(outputs.get("conv_trans_cross_logit_abs", 0.0))
        
        monitor_stats["coop_prompt_norm"] = _to_float(outputs.get("coop_prompt_norm", 0.0))
        
        vw = outputs.get("view_weights")
        if vw is not None and isinstance(vw, torch.Tensor) and vw.dim() >= 2:
            monitor_stats["view_weights"] = vw.mean(0).detach().cpu().tolist()
        
        aw = outputs.get("attn_weights")
        if aw is not None and isinstance(aw, torch.Tensor) and aw.dim() >= 2:
            
            monitor_stats["attn_dist"] = aw.mean(dim=(0, 1)).detach().cpu().tolist()
        
        gw = outputs.get("conf_weights")
        if gw is not None and isinstance(gw, torch.Tensor):
            monitor_stats["conf_weights"] = gw.mean(0).detach().cpu().tolist()

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    metrics = compute_all_metrics(labels, preds, probs)
    avg_loss = total_loss / max(len(loader), 1)
    accuracy = metrics["accuracy"]

    
    branch_acc = {}
    for branch, preds_list in branch_preds.items():
        if preds_list:
            branch_pred = np.concatenate(preds_list)
            branch_acc[branch] = float((branch_pred == labels).mean())
        else:
            branch_acc[branch] = 0.0

    return avg_loss, accuracy, metrics, branch_acc, monitor_stats


def check_resume_available(resume_checkpoint):

    if not resume_checkpoint.exists():
        return None

    state_dict = resume_checkpoint.load()
    if state_dict is None:
        print("警告: 检查点文件存在但验证失败，将从头开始训练")
        
        damaged_path = resume_checkpoint.save_path + ".damaged"
        try:
            shutil = __import__("shutil")
            shutil.move(resume_checkpoint.save_path, damaged_path)
            print(f"  损坏检查点已备份至: {damaged_path}")
        except Exception:
            pass
        return None

    print_resume_info(state_dict)
    return state_dict


def main():

    global _interrupted

    
    
    
    
    import argparse
    parser = argparse.ArgumentParser(description="单阶段训练（支持多种子集成）")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（不传则使用 config 默认 seed=1001）。"
             "传入后权重保存为 best_seed{seed}.pth，便于多种子集成",
    )
    args = parser.parse_args()

    
    if args.seed is not None:
        
        TRAIN_CONFIG["seed"] = args.seed
        
        global BEST_WEIGHT_PATH, RESUME_CHECKPOINT_PATH
        BEST_WEIGHT_PATH, RESUME_CHECKPOINT_PATH = get_seed_weight_paths(args.seed)
        print(f"\n[多种子集成模式] seed={args.seed}")
        print(f"  最佳权重路径: {BEST_WEIGHT_PATH}")
        print(f"  断点续训路径: {RESUME_CHECKPOINT_PATH}")

    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    
    seed_everything(TRAIN_CONFIG["seed"])
    ensure_dirs()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"随机种子: {TRAIN_CONFIG['seed']}")

    
    print("\n构建数据加载器...")
    train_loader, val_loader, test_loader = build_data_loaders(TRAIN_CONFIG)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")

    
    model = build_model(device)

    
    has_nan = _validate_model_weights(model)
    if has_nan:
        sys.exit(1)

    
    print("\n前向传播诊断...")
    nan_source = _diagnose_forward_nan(model, train_loader, device)
    if nan_source:
        sys.exit(1)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    
    
    
    
    historical_best = -float("inf")  
    if os.path.exists(BEST_WEIGHT_PATH):
        try:
            checkpoint = torch.load(BEST_WEIGHT_PATH, map_location="cpu", weights_only=False)
            old_best_score = checkpoint.get("best_score", None)
        finally:
            pass



    
    
    best_weight_mgr = BestWeightManager(BEST_WEIGHT_PATH, historical_best=historical_best)

    
    resume_checkpoint = TrainingCheckpoint(
        RESUME_CHECKPOINT_PATH,
        max_keep=RESUME_CONFIG["max_keep"],
        validate_checksum=RESUME_CONFIG["validate_checksum"],
    )

    
    resume_state = None
    


    
    optimizer = build_optimizer(model)
    total_epochs = TRAIN_CONFIG["total_epochs"]
    warmup_epochs = TRAIN_CONFIG["warmup_epochs"]
    val_interval = max(1, int(TRAIN_CONFIG.get("val_interval", 2)))
    scheduler = build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs)
    scaler = GradScaler('cuda') if TRAIN_CONFIG["amp"] else None

    
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    
    best_score = historical_best if (historical_best is not None and historical_best > 0) else -float("inf")
    if resume_state is not None:
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_epoch = resume_state.get("global_epoch", 0)
        history_restored = resume_state.get("history", {})
        if history_restored:
            history["train_loss"] = history_restored.get("train_loss", [])
            history["val_loss"] = history_restored.get("val_loss", [])
            history["train_acc"] = history_restored.get("train_acc", [])
            history["val_acc"] = history_restored.get("val_acc", [])
        
        if os.path.exists(BEST_WEIGHT_PATH):
            best_info = best_weight_mgr.load(map_location=device)
            if best_info is not None:
                best_score = best_info.get("best_score", -float("inf"))


    
    class_weights = None
    if TRAIN_CONFIG.get("use_class_weight", False):
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, "get_class_weights_tensor"):
            class_weights = train_dataset.get_class_weights_tensor().to(device)
            print(f"  类别权重: {class_weights.tolist()}")
    
    focal_gamma = TRAIN_CONFIG.get("focal_gamma", 0)
    if focal_gamma > 0:
        criterion = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=TRAIN_CONFIG["label_smoothing"],
        )

    else:
        criterion = nn.CrossEntropyLoss(
            label_smoothing=TRAIN_CONFIG["label_smoothing"],
            weight=class_weights,
        )
    aux_weight = TRAIN_CONFIG["aux_weight"]

    
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n启用 DataParallel 多 GPU 训练（{num_gpus} GPUs: {list(range(num_gpus))}）")
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
    else:
        print(f"\n单 GPU 训练（GPU 数量: {num_gpus}）")

    
    print(f"\n{'='*60}")
    print(f"开始单阶段训练（{total_epochs} epochs, warmup={warmup_epochs}）")
    print(f"辅助损失权重: {aux_weight}")
    print(f"{'='*60}")

    counter = 0
    patience = TRAIN_CONFIG["patience"]

    
    ema = ModelEMA(model, decay=0.999)


    for epoch in range(start_epoch, total_epochs):
        if _interrupted:
            break

        
        
        if hasattr(criterion, 'label_smoothing'):
            progress = epoch / max(total_epochs, 1)
            if progress < 0.5:
                target_ls = TRAIN_CONFIG.get("label_smoothing_warmup", 0.1)
            else:
                target_ls = TRAIN_CONFIG.get("label_smoothing", 0.05)
            criterion.label_smoothing = target_ls

        
        model_inner = unwrap_model(model)

        
        train_loss, train_acc, scaler, pcol_avg = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, epoch=epoch, total_epochs=total_epochs,
            amp=TRAIN_CONFIG["amp"], scaler=scaler,
            aux_weight=aux_weight,
            class_weights=class_weights,  
            pcol_weight=FUSION_CONFIG.get("pcol_weight", 0.5),  
            ema=ema,  
        )

        
        if scheduler is not None:
            scheduler.step()

        if _interrupted:
            
            print(f"\n  在 epoch {epoch+1} 中断，保存训练状态...")
            state = build_training_state(
                model_inner, optimizer, scheduler, "单阶段", epoch + 1,
                best_score, history, scaler,
            )
            resume_checkpoint.save(state)
            break

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if (epoch + 1) % val_interval == 0:
            ema_backup = ema.apply_to_model(model)

            val_loss, val_acc, val_metrics, branch_acc, monitor_stats = validate(
                model, val_loader, criterion, device,
            )

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            current_score = val_metrics["qwk"] + val_acc
            current_qwk = val_metrics["qwk"]

            print(f"[单阶段训练] Epoch {epoch+1}/{total_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                  f"val_qwk={val_metrics['qwk']:.4f}, score={current_score:.4f}")

            vw = monitor_stats.get("view_weights")
            if vw is not None and isinstance(vw, list):
                vw_str = " ".join(f"{w:.4f}" for w in vw)

            ad = monitor_stats.get("attn_dist")
            if ad is not None and isinstance(ad, list):
                ad_str = " ".join(f"{v:.4f}" for v in ad)

            gw = monitor_stats.get("conf_weights")
            if gw is not None and isinstance(gw, list):
                gw_str = " ".join(f"{w:.4f}" for w in gw)

            if best_weight_mgr is not None and best_weight_mgr.should_save(current_score):
                best_weight_mgr.save(
                    model_state_dict=copy.deepcopy(model_inner.state_dict()),
                    current_score=current_score,
                    epoch=epoch + 1,
                    stage="单阶段",
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict(),
                    scaler_state_dict=scaler.state_dict() if scaler else None,
                    extra_info={
                        "val_acc": val_acc,
                        "val_qwk": current_qwk,
                        "val_metrics": {
                            k: v for k, v in val_metrics.items()
                            if k != "confusion_matrix"
                        },
                    },
                )
                best_score = current_score
                counter = 0
                print(f"  >>> 更新最佳权重: best.pth (score={current_score:.4f})")
            else:
                counter += 1
                if counter >= patience:
                    print(f"  [信息] 已连续 {patience} 次验证无提升（早停已禁用，继续训练）")

            ema.restore(model, ema_backup)
        else:
            history["val_loss"].append(None)
            history["val_acc"].append(None)
            print(f"[单阶段训练] Epoch {epoch+1}/{total_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")

        
        if RESUME_CONFIG["save_interval"] > 0:
            if (epoch + 1) % RESUME_CONFIG["save_interval"] == 0:
                state = build_training_state(
                    model_inner, optimizer, scheduler, "单阶段", epoch + 1,
                    best_score, history, scaler,
                )
                resume_checkpoint.save(state)
                print(f"  训练状态: {RESUME_CHECKPOINT_PATH}")

    
    if not _interrupted:
        
        print(f"\n加载最佳权重: {BEST_WEIGHT_PATH}")
        best_info = best_weight_mgr.load(map_location=device)
        if best_info is not None:
            
            
            unwrap_model(model).load_state_dict(best_info["model_state_dict"], strict=False)


        
        plot_training_curves(
            history,
            save_path=os.path.join(LOG_DIR, "training_curves.png"),
        )
        print(f"训练曲线保存至: {LOG_DIR}/training_curves.png")

        
        print("\n" + "=" * 60)
        print("最终测试集评估")
        print("=" * 60)
        criterion = nn.CrossEntropyLoss(label_smoothing=TRAIN_CONFIG["label_smoothing"])
        test_loss, test_acc, test_metrics, _, _ = validate(
            model, test_loader, criterion, device,
        )
        print(format_metrics_str(test_metrics))

        
        metrics_path = os.path.join(LOG_DIR, "final_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        print(f"评估指标保存至: {metrics_path}")

        
        if os.path.exists(RESUME_CHECKPOINT_PATH):
            os.remove(RESUME_CHECKPOINT_PATH)
            print(f"训练完成，已清理断点续训文件: {RESUME_CHECKPOINT_PATH}")

        print("\n训练完成！")
        print(f"最佳权重: {BEST_WEIGHT_PATH}")
    else:
        print(f"\n训练中断，状态已保存。下次运行将自动恢复。")
        print(f"最佳权重: {BEST_WEIGHT_PATH}")
        print(f"断点续训: {RESUME_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
