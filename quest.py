import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import sys
import random

# --- Helper Functions for The Game ---
def slow_print(text, delay=0.03):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def ask_choice(question, options):
    print(f"\n{question}")
    for i, opt in enumerate(options):
        print(f"  [{i+1}] {opt}")

    while True:
        try:
            choice = int(input("\nSelect an option (number): "))
            if 1 <= choice <= len(options):
                return choice - 1
            print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a number.")

def clear_screen():
    print("\n" * 50)

# --- The Quest Begins ---

clear_screen()
print("""
  ____        _
 |  _ \      | |
 | |_) | __ _| |__  _   _
 |  _ < / _` | '_ \| | | |
 | |_) | (_| | |_) | |_| |
 |____/ \__,_|_.__/ \__, |
  / ____|                      |___/
 | (___   ___  _ __  _ __   ___| |_
  \___ \ / _ \| '_ \| '_ \ / _ \ __|
  ____) | (_) | | | | | | |  __/ |_
 |_____/ \___/|_| |_|_| |_|\___|\__|

      ~ The Training Quest ~
""")

slow_print("Welcome, Apprentice AI Researcher.")
slow_print("Your goal: Train a neural network to speak like Shakespeare.")
slow_print("This model will be a tiny ancestor of Claude Sonnet.")
time.sleep(1)

# --- Level 1: The Data ---
print("\n=== LEVEL 1: PREPARING THE DATA ===")
slow_print("First, we must load the ancient texts (Shakespeare).")
slow_print("Computers don't understand words, only numbers.")
slow_print("We will build a vocabulary mapping characters to integers.")

try:
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: input.txt not found. Please run the curl command first.")
    sys.exit(1)

chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"\n[SYSTEM] Dataset loaded. Length: {len(text)} characters.")
print(f"[SYSTEM] Vocabulary size: {vocab_size} unique characters.")

slow_print("\nLet's see an example of tokenization:")
sample_text = "To be, or not to be"
print(f" Original: '{sample_text}'")
print(f" Tokenized: {encode(sample_text)}")

input("\nPress Enter to proceed to Model Architecture...")

# --- Level 2: The Model ---
clear_screen()
print("\n=== LEVEL 2: CONSTRUCTING THE BRAIN ===")
slow_print("We need to design the neural network architecture.")
slow_print("You have a choice of architecture complexity.")
print("\n(Note: Modern LLMs use Transformers, but we start with RNNs which are easier to train on CPUs)")

choice = ask_choice("Choose your model architecture:", [
    "Simple RNN (Recurrent Neural Network) - Fast but forgetful",
    "GRU (Gated Recurrent Unit) - Balanced performance",
    "LSTM (Long Short-Term Memory) - Classic, best memory for this scale"
])

# Hyperparameters based on choice
if choice == 0:
    model_type = "RNN"
    hidden_size = 128
    n_layers = 1
elif choice == 1:
    model_type = "GRU"
    hidden_size = 192
    n_layers = 2
else:
    model_type = "LSTM"
    hidden_size = 256
    n_layers = 2

print(f"\n[SYSTEM] Initializing {model_type} architecture...")

# Data loader
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 64 # Context length
batch_size = 32 # How many independent sequences to process in parallel

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Model Definition
class BabySonnet(nn.Module):
    def __init__(self, vocab_size, hidden_size, model_type):
        super().__init__()
        self.model_type = model_type
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        if model_type == "RNN":
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, batch_first=True)

        self.ln_f = nn.LayerNorm(hidden_size) # Final layer norm
        self.head = nn.Linear(hidden_size, vocab_size) # Language model head

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embed tokens
        tok_emb = self.token_embedding(idx) # (B,T,C)

        # RNN/LSTM processing
        x, _ = self.rnn(tok_emb)

        # Layer Norm
        x = self.ln_f(x)

        logits = self.head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop context if needed
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'

print(f"[SYSTEM] Using device: {device}")

model = BabySonnet(vocab_size, hidden_size, model_type)
m = model.to(device)
print(f"[SYSTEM] Model {model_type} created with {sum(p.numel() for p in m.parameters())/1e3:.1f}k parameters.")

input("\nPress Enter to begin the Training Ritual...")

# --- Level 3: Training ---
clear_screen()
print("\n=== LEVEL 3: THE TRAINING RITUAL ===")
slow_print("We now begin the optimization loop.")
slow_print("1. Forward Pass: The model guesses the next character.")
slow_print("2. Loss Calculation: We measure how wrong it was.")
slow_print("3. Backward Pass (Backprop): We calculate gradients.")
slow_print("4. Optimizer Step: We update the weights to reduce error.")

# Optimizer
learning_rate = 3e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

max_iters = 500 # Short training for demo
eval_interval = 100

print(f"\n[SYSTEM] Starting training for {max_iters} iterations...")
start_time = time.time()

try:
    for iter in range(max_iters):
        # Sample a batch of data
        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)

        # Evaluate the loss
        logits, loss = model(xb, yb)

        # Optimize
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Progress bar
        if iter % 10 == 0:
            sys.stdout.write(f"\rIter {iter}/{max_iters} | Loss: {loss.item():.4f} | {'#' * (iter // 20)}")
            sys.stdout.flush()
            time.sleep(0.01) # Just for visual effect

    print(f"\n\n[SYSTEM] Training complete! Final Loss: {loss.item():.4f}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")

except KeyboardInterrupt:
    print("\n[SYSTEM] Training interrupted by user.")

input("\nPress Enter to see what the baby model has learned...")

# --- Level 4: Generation ---
clear_screen()
print("\n=== LEVEL 4: THE INVOCATION ===")
slow_print("Now we ask the model to generate new text based on what it learned.")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nGenerating text (this might take a moment)...\n")
print("-" * 40)
generated_output = decode(m.generate(context, max_new_tokens=500)[0].tolist())
slow_print(generated_output, delay=0.005)
print("-" * 40)

slow_print("\nIt may not be perfect English yet (it needs hours of training, not seconds).")
slow_print("But you have successfully built and trained a neural network from scratch!")
print("\n=== QUEST COMPLETE ===")
