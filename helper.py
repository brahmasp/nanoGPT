import numpy as np
import torch

# Filter out sentiment token from the input # 
def filter_sentiment_token(input_ids, sentiment_token):
    return np.array([x for x in input_ids if x != sentiment_token])

# Reward Model using prediction and prefix # 
def reward_model(prediction, prefix):
    pass

# we want to be able to obtain reward model 
def compute_policy_gradient_loss(logits, prefix, reward_char, use_heuristic_reward=True):
    prediction = torch.argmax(logits, dim=-1)
    reward = reward_model(prediction, prefix)

    log_logits = torch.nn.functional.log_softmax(logits, dim=-1)
    log_prob_char = log_logits[:, :, reward_char]

    policy_gradient_loss = -torch.mean(log_prob_char * reward.unsqueeze(-1))
    return policy_gradient_loss


    