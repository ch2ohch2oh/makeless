import random

import matplotlib
import torch
import torch.nn.functional as F

from bigram import BigramNameModel
from mlp import MLPNameModel

import matplotlib.pyplot as plt

# matplotlib.use("Agg")

random.seed(42)


def load_names(file_path="data/names.txt"):
    with open(file_path, "r") as f:
        names = f.read().split()
    print(f"Loaded {len(names)} names from {file_path}")
    return names


def create_train_set(names, sep=".", context_length=1, stoi=None):
    assert stoi is not None, "stoi must be provided"
    print(f"Creating training set with context length={context_length} sep='{sep}'")
    x_chars, y_chars = [], []
    for name in names:
        name = name + sep
        context = sep * context_length
        for i in range(len(name)):
            x_chars.append([stoi[ch] for ch in context])
            y_chars.append(stoi[name[i]])
            context = context[1:] + name[i]  # shift context
    xs = torch.tensor(x_chars)  # input chars
    ys = torch.tensor(y_chars)  # target chars

    itos = {i: ch for ch, i in stoi.items()}
    print(f"xs shape: {xs.shape}, ys shape: {ys.shape}")
    print("First 10 examples:")
    for i in range(10):
        x_text = "".join(itos[c] for c in x_chars[i])
        print(f"{x_text} => {itos[y_chars[i]]} \tstoi: {xs[i]} => {ys[i]}")
    return xs, ys


def create_vocab(names, sep_char="."):
    chars = [sep_char] + sorted(set("".join(names)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    assert vocab_size == 27, f"Expected vocab size 27, got {vocab_size}"
    print(f"Vocabulary size: {vocab_size}")
    return chars, stoi, itos


def train_bigram_model():
    SEP = "."

    names = load_names()
    # Shuffle names to ensure randomness
    random.shuffle(names)

    chars, stoi, itos = create_vocab(names, sep_char=SEP)
    vocab_size = len(stoi)

    xs, ys = create_train_set(names, sep=SEP, context_length=1, stoi=stoi)
    xs = xs.squeeze(-1)

    model = BigramNameModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    print("\nTraining loop:")
    for epoch in range(200):
        logits, loss = model(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\nGenerated names:")
    for _ in range(10):
        print(model.generate(stoi=stoi, itos=itos))

    # Print next char probabilities for a given input char
    input_char = "n"
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


def train_mlp_model():
    SEP = "."

    names = load_names()
    random.shuffle(names)

    num_train = int(0.8 * len(names))
    train_names = names[:num_train]
    val_names = names[num_train:]
    print(f"Train names: {len(train_names)}, Val names: {len(val_names)}")

    chars, stoi, itos = create_vocab(train_names, sep_char=SEP)

    context_length = 3

    xs, ys = create_train_set(names, sep=SEP, context_length=context_length, stoi=stoi)

    model = MLPNameModel(
        vocab_size=len(stoi),
        context_length=context_length,
        embed_dim=2,
        hidden_size=50,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

    batch_size = 100

    print("\nTraining loop:")
    loss_history = []
    for batch_idx in range(2000):
        batch_indices = torch.randint(0, xs.shape[0], (batch_size,))
        xb = xs[batch_indices]
        yb = ys[batch_indices]
        logits, loss = model(xb, yb)
        loss_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(loss_history)
    plt.xlabel("Batch Index")
    plt.ylabel(loss_history)
    plt.title("Training Loss over Batches")
    plt.show()
    # plt.savefig("mlp_loss_plot.png")

    print("\nGenerated names:")
    for _ in range(10):
        print(
            model.generate(
                start_context=SEP * context_length, max_len=20, stoi=stoi, itos=itos
            )
        )


if __name__ == "__main__":
    train_mlp_model()
