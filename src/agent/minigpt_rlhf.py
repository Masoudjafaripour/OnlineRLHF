import torch
import torch.nn as nn
import torch.nn.functional as F

# ==
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
        atten_out, _ = self.atten(x,x,x)
        x = self.norm1(atten_out + x)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)
    
# =====
class MiniGPT(nn.Module): # what if it inherent from Transformer block? 
    def __init__(self, vocab_size, dim, num_heads, depth):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleLsit([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)
    
# ======== 
class DummyTokenizer:
    pad_token_id = 0

tokenizer = DummyTokenizer()

# ====
def sft_loop(model, input_ids, optimizer):
    model.train()
    logits = model(input_ids)
    labels = input_ids.clone()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    loss = criterion(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1)
    )
    # cross(logist, labels)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(loss.item())


# =======
def dpo_loss(logprobs_chosen, logprobs_rejected, beta=0.1):
    return -torch.log(torch.sigmoid(beta * (logprobs_chosen - logprobs_rejected))).mean()

def dpo_loop(model, chosen_input, rejected_input, optimizer):
    model.train()
    logits_chosen = model(chosen_input)
    logits_rejected = model(rejected_input)

    log_prob_chosen = F.log_softmax(logits_chosen, dim=-1).mean(dim=1)
    log_prob_rejected = F.log_softmax(logits_rejected, dim=-1).mean(dim=1)

    loss = dpo_loss(log_prob_chosen, log_prob_rejected)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    
# ====
def rlhf_loop(actor, reward_model, ref_model, input_ids, optimizer, k1_coef=0.1):
    actor.train()
    logits = actor(input_ids)
    log_probs = F.log_softmax(logits, dim=1)

    with torch.no_grad():
        ref_logits = ref_model(input_ids)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        rewards = reward_model(input_ids).unsqueeze(-1)
        k1 = (log_probs - ref_log_probs).sum(dim=-1, keepdim=True)

        advantages = rewards - k1_coef * k1

    selected_log_probs = log_probs.sum(dim=-1, keepdim=True)
    loss = -(selected_log_probs * advantages).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# log
# rew -->> k1 = (log - ref_log)
# def
vocab_size = 1000
model = MiniGPT(vocab_size=vocab_size, dim=64, num_heads=4, depth=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
input_ids = torch.randint(0, vocab_size, (4, 16))

# sim
dpo_loop(model, input_ids, input_ids, optimizer)
rlhf_loop(model, lambda x:torch.ones(x.shape[0]), model, input_ids, optimizer)