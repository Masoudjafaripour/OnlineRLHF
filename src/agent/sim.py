import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
# =====================
# 1. Transformer Components
# =====================

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.atten = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        atten_out, _ = self.atten(x, x, x)
        x = self.norm1(x + atten_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, depth):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, num_heads) for _ in range(depth)])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# =====================
# 2. Simulated Human Preference Model
# =====================

def simulate_preference(traj_a, traj_b, irrationality=0.1):
    """Returns 1 if A preferred over B, else 0 (simulate human preference with noise)."""
    score_a = traj_a.sum().item()
    score_b = traj_b.sum().item()
    delta = score_a - score_b
    prob = torch.sigmoid(torch.tensor(delta))  # soft preference
    prob = prob * (1 - irrationality) + 0.5 * irrationality  # noise
    return 1 if random.random() < prob.item() else 0

# =====================
# 3. Reward Model from Preferences
# =====================

class SimpleRewardModel(nn.Module):
    """Score a trajectory (sequence of tokens)"""
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.score = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids):
        emb = self.embedding(input_ids)  # [B, T, D]
        pooled = emb.mean(dim=1)
        return self.score(pooled).squeeze(-1)

# =====================
# 4. Preference-Based RLHF Loop
# =====================

def preference_rlhf_loop(actor, reward_model, optimizer, tokenizer, vocab_size, seq_len=16, batch_size=4):
    actor.train()
    reward_model.eval()

    input_ids_a = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids_b = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Simulate human preference labels (1 if A preferred, 0 if B preferred)
    with torch.no_grad():
        rewards_a = reward_model(input_ids_a)
        rewards_b = reward_model(input_ids_b)

    preferences = []
    for ra, rb in zip(rewards_a, rewards_b):
        preferences.append(1 if ra > rb else 0)

    # Train reward model on preferences (optional: skipped in this version)

    logits_a = actor(input_ids_a)
    logits_b = actor(input_ids_b)

    # log_probs_a = F.log_softmax(logits_a, dim=-1).mean(dim=1)
    # log_probs_b = F.log_softmax(logits_b, dim=-1).mean(dim=1)

    log_probs_a = F.log_softmax(logits_a, dim=-1)
    log_probs_b = F.log_softmax(logits_b, dim=-1)

    # Get log-probs of the actual tokens used (input_ids)
    selected_log_probs_a = log_probs_a.gather(2, input_ids_a.unsqueeze(-1)).squeeze(-1)
    selected_log_probs_b = log_probs_b.gather(2, input_ids_b.unsqueeze(-1)).squeeze(-1)

    # One total log-prob per sequence
    log_probs_a = selected_log_probs_a.sum(dim=1)
    log_probs_b = selected_log_probs_b.sum(dim=1)


    # Simulate DPO-style loss using preferences
    chosen_logprobs = torch.where(
        torch.tensor(preferences).bool(),
        log_probs_a,
        log_probs_b
    )
    rejected_logprobs = torch.where(
        torch.tensor(preferences).bool(),
        log_probs_b,
        log_probs_a
    )

    loss = -torch.log(torch.sigmoid(chosen_logprobs - rejected_logprobs)).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # print(f"[RLHF-Preference] Loss: {loss.item():.4f}")
    return loss.item()

# =====================
# 5. Main Experiment Loop
# =====================

class DummyTokenizer:
    pad_token_id = 0

def run_experiment():
    torch.manual_seed(42)
    vocab_size = 1000
    dim = 64
    seq_len = 16
    model = MiniGPT(vocab_size=vocab_size, dim=dim, num_heads=4, depth=2)
    reward_model = SimpleRewardModel(vocab_size=vocab_size, dim=dim)
    tokenizer = DummyTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    Loss = []
    for step in range(1000):
        loss = preference_rlhf_loop(model, reward_model, optimizer, tokenizer, vocab_size=vocab_size, seq_len=seq_len)
        Loss.append(loss)
    
    plt.plot(Loss)
    plt.show()

if __name__ == "__main__":
    run_experiment()
