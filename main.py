"""
Основной файл с решением соревнования
Здесь должен быть весь ваш код для создания предсказаний
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce
        return focal_loss.mean()

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 16)
        inp = 17 + 16 + 3  # len(num_cols) + embedding + actions

        self.net = nn.Sequential(
            nn.Linear(inp, 768), nn.LayerNorm(768), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(768, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.45),
            nn.Linear(512, 384), nn.LayerNorm(384), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(384, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.35),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, nums, cats, acts):
        e = self.emb(cats[:, 0])
        x = torch.cat([nums, e, acts], dim=1)
        return self.net(x).squeeze(-1)

class BanditDataset(Dataset):
    def __init__(self, nums, cats, acts, targets=None):
        self.nums = torch.from_numpy(nums).float()
        self.cats = torch.from_numpy(cats).long()
        self.acts = torch.from_numpy(acts).float()
        self.targets = None if targets is None else torch.from_numpy(targets).float()

    def __len__(self): 
        return len(self.nums)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.nums[idx], self.cats[idx], self.acts[idx], self.targets[idx]
        return self.nums[idx], self.cats[idx], self.acts[idx]

def get_seg_num(x):
    try:
        return int(str(x).split(')')[0].strip())
    except:
        return 1

def snips(q, actions, rewards, temp):
    pi = np.exp(q / temp)
    pi /= pi.sum(axis=1, keepdims=True)
    w = pi[np.arange(len(actions)), actions] / (1/3.0)
    return (w * rewards).sum() / w.sum() if w.sum() > 0 else 0

def train_model(train_data, num_cols, train_cats, train_acts, y, train_idx=None, full=False, device=device):
    if full:
        ds = BanditDataset(train_data[num_cols].values, train_cats, train_acts, y)
    else:
        ds = BanditDataset(train_data.loc[train_idx, num_cols].values, train_cats[train_idx], 
                          train_acts[train_idx], y[train_idx])
    
    loader = DataLoader(ds, batch_size=8192, shuffle=True, pin_memory=True, num_workers=0)

    model = QNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0018, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=0.0005)
    
    scaler = GradScaler() if device.type == 'cuda' else None
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    
    epochs = 60
    swa_start = 40

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            nums, cats, acts, targets = [x.to(device) for x in batch]
            with autocast(device_type=device.type if device.type == 'cuda' else 'cpu'):
                logits = model(nums, cats, acts)
                loss = criterion(logits, targets)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        print(f'Epoch {epoch+1:02d} loss: {total_loss/len(loader):.6f}')

    if full:
        swa_model.update_parameters(model)
    return swa_model

def get_q(model, nums, cats, device=device):
    model.eval()
    nums_t = torch.from_numpy(nums).float().to(device)
    cats_t = torch.from_numpy(cats).long().to(device)
    q = np.zeros((len(nums), 3))
    with torch.no_grad():
        for a in range(3):
            acts = torch.zeros(len(nums), 3).to(device)
            acts[:, a] = 1.0
            preds = []
            for i in range(0, len(nums), 16384):
                with autocast(device_type=device.type if device.type == 'cuda' else 'cpu'):
                    logits = model(nums_t[i:i+16384], cats_t[i:i+16384], acts[i:i+16384])
                preds.append(torch.sigmoid(logits).cpu().numpy())
            q[:, a] = np.concatenate(preds).ravel()
    return q

def create_submission(predictions):
    """
    Пропишите здесь создание файла submission.csv в папку results
    !!! ВНИМАНИЕ !!! ФАЙЛ должен иметь именно такого названия
    """
    import os
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    predictions.to_csv(submission_path, index=False)
    
    print(f"Submission файл сохранен: {submission_path}")
    
    return submission_path

def main():
    """
    Главная функция программы
    
    Вы можете изменять эту функцию под свои нужды,
    но обязательно вызовите create_submission() в конце!
    """
    print("=" * 50)
    print("Запуск решения соревнования")
    print("=" * 50)
    
    # Загрузка данных
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    
    # Предобработка данных
    train['zip_code'] = train['zip_code'].str.replace('Surburban', 'Suburban').str.strip()
    test['zip_code'] = test['zip_code'].str.replace('Surburban', 'Suburban').str.strip()

    train['hist_seg'] = train['history_segment'].apply(get_seg_num)
    test['hist_seg'] = test['history_segment'].apply(get_seg_num)

    train['log_history'] = np.log1p(train['history'])
    test['log_history'] = np.log1p(test['history'])

    train['mens_log'] = train['mens'] * train['log_history']
    train['womens_log'] = train['womens'] * train['log_history']
    test['mens_log'] = test['mens'] * test['log_history']
    test['womens_log'] = test['womens'] * test['log_history']

    train['mens_womens'] = train['mens'] * train['womens']
    test['mens_womens'] = test['mens'] * test['womens']

    train['any_category'] = (train['mens'] + train['womens']).clip(upper=1)
    test['any_category'] = (test['mens'] + test['womens']).clip(upper=1)

    train['recency_sq'] = train['recency'] ** 2
    test['recency_sq'] = test['recency'] ** 2

    train = pd.get_dummies(train, columns=['zip_code', 'channel'], dtype=float)
    test = pd.get_dummies(test, columns=['zip_code', 'channel'], dtype=float)

    train, test = train.align(test, join='left', axis=1, fill_value=0.0)

    action_map = {'Mens E-Mail': 0, 'Womens E-Mail': 1, 'No E-Mail': 2}
    train['action'] = train['segment'].map(action_map)

    num_cols = ['recency', 'log_history', 'mens', 'womens', 'newbie', 'hist_seg',
                'mens_log', 'womens_log', 'mens_womens', 'any_category', 'recency_sq',
                'zip_code_Rural', 'zip_code_Suburban', 'zip_code_Urban',
                'channel_Multichannel', 'channel_Phone', 'channel_Web']

    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    train_cats = train[['hist_seg']].values.astype(np.int64)
    test_cats = test[['hist_seg']].values.astype(np.int64)

    train_acts = np.eye(3)[train['action'].values]
    y = train['visit'].values.astype(np.float32)

    # Разделение на train/validation
    train_idx, val_idx = train_test_split(
        np.arange(len(train)), 
        test_size=0.2, 
        random_state=42, 
        stratify=train['action']
    )

    val_nums = train.loc[val_idx, num_cols].values
    val_cats = train_cats[val_idx]
    val_acts = train['action'].values[val_idx]
    val_r = y[val_idx]

    # Обучение моделей
    print("Обучение модели на train/validation split...")
    swa_model_val = train_model(train, num_cols, train_cats, train_acts, y, train_idx, full=False, device=device)
    
    print("Обучение модели на всех данных...")
    swa_model = train_model(train, num_cols, train_cats, train_acts, y, full=True, device=device)

    # Валидация и подбор температуры
    print("Подбор оптимальной температуры...")
    val_q = get_q(swa_model_val, val_nums, val_cats, device=device)

    best_temp = 0.015
    best_score = 0
    for t in np.concatenate([np.linspace(0.005, 0.04, 100), np.linspace(0.041, 0.15, 50)]):
        sc = snips(val_q, val_acts, val_r, t)
        if sc > best_score:
            best_score = sc
            best_temp = t

    print(f"Лучшая температура: {best_temp:.4f}, лучший score: {best_score:.6f}")

    # Предсказание на тестовых данных
    print("Создание предсказаний...")
    test_q = get_q(swa_model, test[num_cols].values, test_cats, device=device)

    probs = np.exp(test_q / best_temp)
    probs /= probs.sum(axis=1, keepdims=True)

    # Создание submission файла
    submission = pd.DataFrame({
        'id': test['id'],
        'p_mens_email': probs[:, 0],
        'p_womens_email': probs[:, 1],
        'p_no_email': probs[:, 2]
    })
    
    # Создание submission файла (ОБЯЗАТЕЛЬНО!)
    create_submission(submission)
    
    print("=" * 50)
    print("Выполнение завершено успешно!")
    print("=" * 50)

if __name__ == "__main__":
    main()