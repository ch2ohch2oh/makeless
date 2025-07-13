import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramNameModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets):
        logits = self.embed(x)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, start_token=".", max_len=20, stoi=None, itos=None):
        assert stoi is not None and itos is not None, "stoi and itos must be provided"
        result = [stoi[start_token]]
        while len(result) < max_len:
            logits = self.embed(torch.tensor([result[-1]]))
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1).item()
            result.append(next_ix)
            if itos[next_ix] == ".":
                break
        return "".join(itos[ix] for ix in result[1:-1])  # remove start/end
