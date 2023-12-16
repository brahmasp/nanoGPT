import numpy as np
import torch
import os
import pickle
from model import GPTConfig, GPT, RewardPredictor
import wandb
import time

# Goal capture a reward model that can be used to train nanoGPT
out_reward_dir = 'out_reward_model/reward_model.pt'
dataset_name = 'bernard_char'
meta_path = os.path.join('data', dataset_name,  'meta.pkl')


meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])




data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
train_data = train_data[:num_train_char] # crop to desired length

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Get a batch of data using the split # 
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        pg_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            pg_loss = compute_policy_gradient_loss(logits, reward_char)
            losses[k] = loss.item()
            pg_losses[k] = pg_loss.item()
        out[split + "_loss"] = losses.mean()
        out[split + "_pg_loss"] = pg_losses.mean()
    model.train()
    return out

def compute_policy_gradient_loss(logits, reward_char, use_heuristic_reward=True):
    prediction = torch.argmax(logits, dim=-1)
    if use_heuristic_reward: 
        reward = get_heuristic_reward(prediction)
    else: 
        reward = trained_reward_model(prediction)

    log_logits = torch.nn.functional.log_softmax(logits, dim=-1)
    log_prob_char = log_logits[:, :, reward_char]

    policy_gradient_loss = -torch.mean(log_prob_char * reward.unsqueeze(-1))
    return policy_gradient_loss

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # st

model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
reward_char = 57

state_dict = torch.load(out_reward_dir, map_location=device)
reward_model = RewardPredictor(block_size=256)
reward_model.to(device)
reward_model.load_state_dict(state_dict)

if wandb_log:
    wandb.init(project='cs839', name='policy_gradient_' + str(time.time()), config=config)


def get_heuristic_reward(prediction):
    reward = torch.sum(prediction == reward_char, dim=1).float()/block_size
    return reward

def trained_reward_model(prediction):
    reward = reward_model(X.float())
    return reward

min_loss = float('inf')
if __name__ == "__main__":
    for iter_num in range(num_train_batches): 
        lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        X, Y = get_batch('train')
        logits, loss = model(X, Y)
        prediction = torch.argmax(logits, dim=-1)        
        policy_gradient_loss = compute_policy_gradient_loss(logits, reward_char, use_heuristic_reward=False)
        total_loss = policy_gradient_loss * 5.0 + loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if  iter_num  % eval_interval == 0:
            losses = estimate_loss()
            print(f"Iteration {iter_num} | loss {losses['train_pg_loss'].item():.5f} | val loss {losses['val_pg_loss'].item():.5f}")
            if wandb_log: 
                wandb.log({
                    "train/loss": losses["train_loss"].item(),
                    "val/loss": losses["val_loss"].item(),
                    "train/pg_loss": losses["train_pg_loss"].item(),
                    "val/pg_loss": losses["val_pg_loss"].item(),
                    "train_policy_gradient_loss": policy_gradient_loss.item(),
                    "lr": lr,
                })

            # If Losses < Minimm loss then save the model # 
            if losses['val_loss'] < min_loss:
                if save_model: 
                    os.makedirs(model_folder, exist_ok=True)
                    ckpt_dict = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "config": config,
                    }
                    torch.save(ckpt_dict,  os.path.join(model_folder, 'ckpt.pt'))