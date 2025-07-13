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

    def generate(self, start_token=".", max_len=20):
        result = [stoi[start_token]]
        while len(result) < max_len:
            logits = self.embed(torch.tensor([result[-1]]))
            probs = F.softmax(logits, dim=-1)
            next_ix = torch.multinomial(probs, num_samples=1).item()
            result.append(next_ix)
            if itos[next_ix] == ".":
                break
        return "".join(itos[ix] for ix in result[1:-1])  # remove start/end


def load_names(file_path="data/names.txt"):
    with open(file_path, "r") as f:
        names = f.read().split()
    print(f"Loaded {len(names)} names from {file_path}")
    return names


if __name__ == "__main__":
    SEP = "."

    names = load_names()

    chars = [SEP] + sorted(set("".join(names)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    assert vocab_size == 27

    xs, ys = [], []
    text = SEP.join(names)
    for i in range(len(text) - 1):
        x_char = text[i]
        y_char = text[i + 1]
        xs.append(stoi[x_char])
        ys.append(stoi[y_char])

    x = torch.tensor(xs)  # input chars
    y = torch.tensor(ys)  # target chars
    print(f"Input tensor shape: {x.shape}, Target tensor shape: {y.shape}")
    print(f"x[:10]: {x[:10]}\ny[:10]: {y[:10]}")

    model = BigramNameModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    print("\nTraining loop:")
    for epoch in range(200):
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\nGenerated names:")
    for _ in range(10):
        print(model.generate())

    # Print next char probabilities for a given input char
    input_char = "."
    input_ix = torch.tensor([stoi[input_char]])
    logits = model.embed(input_ix)
    probs = F.softmax(logits, dim=-1)
    total_prob = 0.0
    print(f"\nNext character probabilities after '{input_char}':")
    for next_char in chars:
        next_ix = stoi[next_char]
        prob = probs[0, next_ix].item()
        total_prob += prob
        print(f"P({input_char} => {next_char}) = {prob:.4f}")
    print(f"Total probability: {total_prob:.4f}")
    assert abs(total_prob - 1.0) < 1e-4, "Probabilities do not sum to 1!"
