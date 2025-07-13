import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPNameModel(nn.Module):
    def __init__(self, vocab_size, context_length=3, embed_dim=10, hidden_size=100):
        super().__init__()
        self.context_length = context_length
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * context_length, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, targets):
        # x shape: (batch_size, context_length)
        batch_size = x.shape[0]
        x_emb = self.embed(x)  # (batch_size, context_length, embed_dim)
        x_flat = x_emb.view(
            batch_size, -1
        )  # flatten to (batch_size, context_length * embed_dim)
        h = F.relu(self.fc1(x_flat))  # hidden layer with ReLU
        logits = self.fc2(h)  # output layer
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, start_context="...", max_len=20, stoi=None, itos=None):
        assert stoi is not None and itos is not None, "stoi and itos must be provided"
        assert len(start_context) >= self.context_length, (
            f"start_context must be greater than or equal to context length {self.context_length}"
        )

        context = [stoi[ch] for ch in start_context]
        result = []
        for _ in range(max_len):
            x = torch.tensor(
                [context[-self.context_length :]]
            )  # ensure correct context length
            logits, _ = self.forward(x, torch.tensor([0]))  # dummy target
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1).item()
            if itos[next_ix] == ".":
                break
            result.append(next_ix)
            context.append(next_ix)

        return "".join(itos[ix] for ix in result)  # return generated name
