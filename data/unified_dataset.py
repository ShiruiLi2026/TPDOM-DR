import os
import random
import numpy as np
import pandas as pd
import cv2

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform
from torchvision.transforms import functional as F_t




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class UnifiedFundusDataset(Dataset):

    def __init__(self, data_roots, csv_paths, num_view=4,
                 flair_size=512, mvcinn_size=224,
                 view_fill_strategy="copy",
                 is_train=True, balance=False,
                 randaugment_n=0, randaugment_m=9,
                 random_erasing_p=0.0, random_erasing_scale=(0.02, 0.2),
                 single_view=False, img_ext=None):

        super().__init__()
        self.num_view = num_view
        self.flair_size = flair_size
        self.mvcinn_size = mvcinn_size
        self.view_fill_strategy = view_fill_strategy
        self.is_train = is_train
        self.single_view = single_view
        self.img_ext = img_ext

        
        self.samples = []
        for root, csv_path in zip(data_roots, csv_paths):
            if csv_path and os.path.exists(csv_path):
                self.samples.extend(self._load_csv(root, csv_path))

        
        self.view_groups = self._group_views_by_patient()

        
        self.sample_weights = self._compute_sample_weights() if balance else None

        
        self.randaugment = None
        if is_train and randaugment_n > 0:
            self.randaugment = transform.RandAugment(
                num_ops=randaugment_n, magnitude=randaugment_m,
            )
        self.random_erasing = None
        if is_train and random_erasing_p > 0:
            self.random_erasing = transform.RandomErasing(
                p=random_erasing_p, scale=random_erasing_scale,
            )

        
        
        
        self.transform_train = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((mvcinn_size, mvcinn_size)),
            transform.RandomResizedCrop(mvcinn_size, scale=(0.8, 1.0)),
            transform.RandomRotation(15),
            transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
        ])
        self.transform_test = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((mvcinn_size, mvcinn_size)),
            transform.ToTensor(),
        ])

        
        
        self.flair_transform_train = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((flair_size, flair_size)),
            transform.RandomResizedCrop(flair_size, scale=(0.8, 1.0)),
            transform.RandomRotation(15),
            transform.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
            transform.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_csv(self, root, csv_path):

        samples = []

        
        is_txt = csv_path.lower().endswith(".txt")
        if is_txt:
            
            with open(csv_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    img_id, level_str = parts[0], parts[1]
                    try:
                        level = int(level_str)
                    except ValueError:
                        continue
                    
                    
                    if level >= 5:
                        continue
                    img_name = img_id if img_id.endswith((".jpg", ".jpeg", ".png")) else img_id + ".jpg"
                    img_path = os.path.join(root, img_name)
                    raw_id = img_id.rsplit(".", 1)[0]  
                    patient_id = self._extract_patient_id(raw_id)
                    view_idx = self._extract_view_idx(raw_id)
                    samples.append({
                        "image_path": img_path,
                        "level": level,
                        "patient_id": patient_id,
                        "view_idx": view_idx,
                        "raw_id": raw_id,
                    })
            return samples

        
        df = pd.read_csv(csv_path)
        
        view_cols = [c for c in df.columns if c.lower() in ("id1", "id2", "id3", "id4")]
        if "level" in df.columns and len(view_cols) >= 4:
            
            for _, row in df.iterrows():
                level = int(row["level"])
                patient_id = str(row.get("id", ""))
                for vcol in view_cols:
                    img_id = str(row[vcol])
                    img_name = img_id if img_id.endswith((".jpg", ".jpeg", ".png")) else img_id + ".jpg"
                    img_path = os.path.join(root, img_name)
                    view_idx = self._extract_view_idx(img_id)
                    samples.append({
                        "image_path": img_path,
                        "level": level,
                        "patient_id": patient_id,  
                        "view_idx": view_idx,
                        "raw_id": img_id,
                    })
            return samples

        
        
        cols_lower = {c.lower().strip(): c for c in df.columns}
        
        id_col = None
        for key in ("id", "image", "image name", "image_name", "filename", "name"):
            if key in cols_lower:
                id_col = cols_lower[key]
                break
        if id_col is None:
            id_col = df.columns[0]
        
        level_col = None
        for key in ("level", "retinopathy grade", "grade", "label"):
            if key in cols_lower:
                level_col = cols_lower[key]
                break
        if level_col is None:
            level_col = df.columns[1]
        for _, row in df.iterrows():
            img_id = str(row[id_col]).strip()
            level = int(row[level_col])
            
            level = min(level, 4)
            if img_id.endswith((".jpg", ".jpeg", ".png")):
                img_name = img_id
            elif self.img_ext is not None:
                img_name = img_id + self.img_ext
            else:
                img_name = img_id + ".jpg"
            img_path = os.path.join(root, img_name)
            if self.single_view:
                
                patient_id = img_id
                view_idx = 0
            else:
                patient_id = self._extract_patient_id(img_id)
                view_idx = self._extract_view_idx(img_id)
            samples.append({
                "image_path": img_path,
                "level": level,
                "patient_id": patient_id,
                "view_idx": view_idx,
                "raw_id": img_id,
            })
        return samples

    def _extract_patient_id(self, img_id):
        
        
        parts = img_id.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").split("_")
        if len(parts) >= 2:
            
            return "_".join(parts[:-1])
        return img_id

    def _extract_view_idx(self, img_id):

        parts = img_id.replace(".jpg", "").replace(".jpeg", "").replace(".png", "").split("_")
        if len(parts) >= 1:
            try:
                return int(parts[-1])
            except ValueError:
                return 0
        return 0

    def _group_views_by_patient(self):

        groups = {}
        for s in self.samples:
            pid = s["patient_id"]
            if pid not in groups:
                groups[pid] = {"views": [], "level": s["level"]}
            groups[pid]["views"].append(s["image_path"])

        
        view_groups = []
        for pid, info in groups.items():
            view_groups.append({
                "patient_id": pid,
                "views": info["views"],
                "level": info["level"],
            })
        return view_groups

    def _compute_sample_weights(self):

        labels = [g["level"] for g in self.view_groups]
        class_counts = np.bincount(labels, minlength=5)
        class_weights = 1.0 / (class_counts + 1e-8)
        return np.array([class_weights[l] for l in labels])

    def get_class_counts(self):

        labels = [g["level"] for g in self.view_groups]
        return np.bincount(labels, minlength=5)

    def get_class_weights_tensor(self):

        counts = self.get_class_counts()
        weights = 1.0 / np.sqrt(counts.astype(np.float32) + 1e-8)
        
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def _fill_views(self, views):

        if len(views) >= self.num_view:
            return views[:self.num_view]

        if self.view_fill_strategy == "copy":
            
            filled = views.copy()
            while len(filled) < self.num_view:
                filled.append(random.choice(views))
            return filled
        elif self.view_fill_strategy == "repeat":
            
            filled = []
            for i in range(self.num_view):
                filled.append(views[i % len(views)])
            return filled
        else:
            
            filled = views.copy()
            while len(filled) < self.num_view:
                filled.append(None)
            return filled

    def _load_image(self, img_path):

        if img_path is None:
            
            return np.zeros((self.mvcinn_size, self.mvcinn_size, 3), dtype=np.uint8)
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((self.mvcinn_size, self.mvcinn_size, 3), dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _prepare_flair_input(self, main_img):

        img = main_img.astype(np.float32)
        if img.max() > 0:
            img /= 255.0
        
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)
        
        h, w = img.shape[-2:]
        max_size = max(h, w)
        scale = max_size / self.flair_size
        new_h, new_w = int(h / scale), int(w / scale)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode="bilinear"
        ).squeeze(0)
        
        pad_h = self.flair_size - new_h
        pad_w = self.flair_size - new_w
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h, 0, 0))
        
        img = (img - _IMAGENET_MEAN_TENSOR) / _IMAGENET_STD_TENSOR
        return img

    def __len__(self):
        return len(self.view_groups)

    def __getitem__(self, idx):

        group = self.view_groups[idx]
        views = group["views"]
        label = group["level"]

        
        views = self._fill_views(views)

        
        imgs = [self._load_image(p) for p in views]

        if self.is_train:
            
            import random as _rng
            from PIL import Image

            
            
            
            flip_h = _rng.random() > 0.5
            angle = _rng.uniform(-10, 10)
            brightness = _rng.uniform(0.9, 1.1)
            contrast = _rng.uniform(0.9, 1.1)
            saturation = _rng.uniform(0.9, 1.1)
            hue = _rng.uniform(-0.05, 0.05)

            def apply_aug(img_np, target_size, use_imagenet_norm=False):
                img_pil = Image.fromarray(img_np)
                img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
                
                img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
                
                img_pil = F_t.adjust_brightness(img_pil, brightness)
                img_pil = F_t.adjust_contrast(img_pil, contrast)
                img_pil = F_t.adjust_saturation(img_pil, saturation)
                img_pil = F_t.adjust_hue(img_pil, hue)
                
                if flip_h:
                    img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                
                if self.randaugment is not None:
                    img_pil = self.randaugment(img_pil)
                img_tensor = F_t.to_tensor(img_pil)
                
                if self.random_erasing is not None:
                    img_tensor = self.random_erasing(img_tensor)
                
                if use_imagenet_norm:
                    img_tensor = F_t.normalize(img_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
                return img_tensor

            
            mvcinn_views = [apply_aug(img, self.mvcinn_size, use_imagenet_norm=False) for img in imgs]
            mvcinn_input = torch.stack(mvcinn_views)  

            
            flair_views = [apply_aug(img, self.flair_size, use_imagenet_norm=True) for img in imgs]
            flair_input = torch.stack(flair_views)  

        else:
            
            
            flair_views = [self._prepare_flair_input(img) for img in imgs]
            flair_input = torch.stack(flair_views)  
            transform_fn = self.transform_test
            mvcinn_views = [transform_fn(img) for img in imgs]
            mvcinn_input = torch.stack(mvcinn_views)  

        return {
            "flair_input": flair_input,
            "mvcinn_input": mvcinn_input,
            "label": label,
            "patient_id": group["patient_id"],
        }


def build_data_loaders(config):

    from torch.utils.data import DataLoader, WeightedRandomSampler
    from configs.config import (
        MFIDDR_TRAIN, MFIDDR_TEST, DDR_TRAIN,
        MFIDDR_TRAIN_CSV, MFIDDR_TEST_CSV, DDR_TRAIN_TXT,
        DATA_CONFIG, TRAIN_CONFIG,
    )

    num_view = DATA_CONFIG["num_view"]
    flair_size = DATA_CONFIG["flair_size"]
    mvcinn_size = DATA_CONFIG["mvcinn_size"]
    fill_strategy = DATA_CONFIG["view_fill_strategy"]
    
    randaugment_n = DATA_CONFIG.get("randaugment_n", 0)
    randaugment_m = DATA_CONFIG.get("randaugment_m", 9)
    random_erasing_p = DATA_CONFIG.get("random_erasing_p", 0.0)
    random_erasing_scale = DATA_CONFIG.get("random_erasing_scale", (0.02, 0.2))

    
    train_roots = [MFIDDR_TRAIN]
    train_csvs = [MFIDDR_TRAIN_CSV]

    
    valid_pairs = [(r, c) for r, c in zip(train_roots, train_csvs)
                   if c is None or os.path.exists(c)]

    train_dataset = UnifiedFundusDataset(
        data_roots=[r for r, _ in valid_pairs],
        csv_paths=[c for _, c in valid_pairs],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=True,
        balance=DATA_CONFIG["balance"],
        randaugment_n=randaugment_n, randaugment_m=randaugment_m,
        random_erasing_p=random_erasing_p, random_erasing_scale=random_erasing_scale,
    )

    val_dataset = UnifiedFundusDataset(
        data_roots=[r for r, _ in valid_pairs],
        csv_paths=[c for _, c in valid_pairs],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=False, balance=False,
    )

    total_size = len(train_dataset)
    val_ratio = float(config.get("val_ratio", 0.2))
    if total_size < 2:
        raise ValueError("MFIDDR 训练集至少需要 2 个样本才能划分训练集和验证集")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须位于 0 和 1 之间")
    val_size = min(max(1, int(total_size * val_ratio)), total_size - 1)
    generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    indices = torch.randperm(total_size, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_groups = train_dataset.view_groups
    val_groups = val_dataset.view_groups
    train_dataset.view_groups = [train_groups[i] for i in train_indices]
    val_dataset.view_groups = [val_groups[i] for i in val_indices]
    train_dataset.sample_weights = (
        train_dataset._compute_sample_weights() if DATA_CONFIG["balance"] else None
    )

    
    test_dataset = UnifiedFundusDataset(
        data_roots=[MFIDDR_TEST],
        csv_paths=[MFIDDR_TEST_CSV],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=False, balance=False,
    )

    
    sampler = None
    if train_dataset.sample_weights is not None:
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            len(train_dataset),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        sampler=sampler, shuffle=(sampler is None),
        num_workers=TRAIN_CONFIG["num_workers"], pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False,
    )

    return train_loader, val_loader, test_loader


def build_idrid_test_loader(config):

    from torch.utils.data import DataLoader
    from configs.config import (
        IDRID_TEST_IMAGES, IDRID_TEST_CSV,
        DATA_CONFIG, TRAIN_CONFIG,
    )

    num_view = DATA_CONFIG["num_view"]
    flair_size = DATA_CONFIG["flair_size"]
    mvcinn_size = DATA_CONFIG["mvcinn_size"]
    fill_strategy = DATA_CONFIG["view_fill_strategy"]

    test_dataset = UnifiedFundusDataset(
        data_roots=[IDRID_TEST_IMAGES],
        csv_paths=[IDRID_TEST_CSV],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=False, balance=False,
        single_view=True,  
    )

    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False,
    )
    return test_loader


def build_ddr_test_loader(config):

    from torch.utils.data import DataLoader
    from configs.config import (
        DDR_TEST, DDR_TEST_TXT,
        DATA_CONFIG, TRAIN_CONFIG,
    )

    num_view = DATA_CONFIG["num_view"]
    flair_size = DATA_CONFIG["flair_size"]
    mvcinn_size = DATA_CONFIG["mvcinn_size"]
    fill_strategy = DATA_CONFIG["view_fill_strategy"]

    test_dataset = UnifiedFundusDataset(
        data_roots=[DDR_TEST],
        csv_paths=[DDR_TEST_TXT],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=False, balance=False,
        single_view=True,  
    )

    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False,
    )
    return test_loader


def build_aptos_test_loader(config):

    from torch.utils.data import DataLoader
    from configs.config import (
        APTOS_TEST, APTOS_TEST_CSV,
        DATA_CONFIG, TRAIN_CONFIG,
    )

    num_view = DATA_CONFIG["num_view"]
    flair_size = DATA_CONFIG["flair_size"]
    mvcinn_size = DATA_CONFIG["mvcinn_size"]
    fill_strategy = DATA_CONFIG["view_fill_strategy"]

    test_dataset = UnifiedFundusDataset(
        data_roots=[APTOS_TEST],
        csv_paths=[APTOS_TEST_CSV],
        num_view=num_view, flair_size=flair_size, mvcinn_size=mvcinn_size,
        view_fill_strategy=fill_strategy, is_train=False, balance=False,
        single_view=True,   
        img_ext=".png",     
    )

    test_loader = DataLoader(
        test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False, num_workers=TRAIN_CONFIG["num_workers"],
        pin_memory=True, drop_last=False,
    )
    return test_loader
