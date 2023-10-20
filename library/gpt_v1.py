import torch
import torch.nn as nn
from torch.nn import functional as F

# Define constants for the model's hyperparameters
BATCH_SZ = 32  # Number of sequences processed concurrently
CONTEXT_LENGTH = 8  # Maximum context length for predictions
MAX_EPOCHS = 3000
CHECKPOINT_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCHES = 200
# -----

torch.manual_seed(1337)

# Load the Shakespeare dataset
with open(
    "/Users/dhruv.b/test_code/attention-is-all-you-need/datasets/tinyshakespeare_input.txt",
    "r",
    encoding="utf-8",
) as file:
    shakespeare_data = file.read()

# Extract unique characters from the data
unique_chars = sorted(list(set(shakespeare_data)))
vocab_len = len(unique_chars)

# Mapping characters to integers and vice versa
char_to_int = {ch: i for i, ch in enumerate(unique_chars)}
int_to_char = {i: ch for i, ch in enumerate(unique_chars)}
encode_string = lambda s: [char_to_int[c] for c in s]
decode_indices = lambda indices: "".join([int_to_char[i] for i in indices])

# Splitting data into training and validation sets
encoded_data = torch.tensor(encode_string(shakespeare_data), dtype=torch.long)
split_idx = int(0.9 * len(encoded_data))
train_set = encoded_data[:split_idx]
val_set = encoded_data[split_idx:]


# Function to fetch random batches of data
def fetch_data_batch(data_type):
    dataset = train_set if data_type == "train" else val_set
    random_idx = torch.randint(len(dataset) - CONTEXT_LENGTH, (BATCH_SZ,))
    input_data = torch.stack([dataset[i : i + CONTEXT_LENGTH] for i in random_idx])
    target_data = torch.stack(
        [dataset[i + 1 : i + CONTEXT_LENGTH + 1] for i in random_idx]
    )
    return input_data.to(DEVICE), target_data.to(DEVICE)


# Function to estimate loss on training and validation sets
@torch.no_grad()
def compute_avg_loss():
    avg_losses = {}
    m.eval()
    for data_type in ["train", "val"]:
        loss_values = torch.zeros(EVAL_BATCHES)
        for k in range(EVAL_BATCHES):
            X, Y = fetch_data_batch(data_type)
            _, batch_loss = m(X, Y)
            loss_values[k] = batch_loss.item()
        avg_losses[data_type] = loss_values.mean()
    m.train()
    return avg_losses


# Define the bigram language model
class BigramLM(nn.Module):
    def __init__(self, vocab_len):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_len, vocab_len)

    def forward(self, input_indices, target_indices=None):
        prediction_logits = self.embedding_table(input_indices)

        if target_indices is None:
            return_loss = None
        else:
            B, T, C = prediction_logits.shape
            prediction_logits = prediction_logits.view(B * T, C)
            target_indices = target_indices.view(B * T)
            return_loss = F.cross_entropy(prediction_logits, target_indices)

        return prediction_logits, return_loss

    def sample_sequence(self, input_indices, token_count):
        for _ in range(token_count):
            pred_logits, _ = self(input_indices)
            pred_logits = pred_logits[:, -1, :]
            token_probs = F.softmax(pred_logits, dim=-1)
            next_token = torch.multinomial(token_probs, num_samples=1)
            input_indices = torch.cat((input_indices, next_token), dim=1)
        return input_indices


m = BigramLM(vocab_len).to(DEVICE)
optim = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

for epoch in range(MAX_EPOCHS):
    if epoch % CHECKPOINT_INTERVAL == 0:
        losses = compute_avg_loss()
        print(
            f"Iteration {epoch}: Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}"
        )

    X_batch, Y_batch = fetch_data_batch("train")
    _, train_loss = m(X_batch, Y_batch)
    optim.zero_grad(set_to_none=True)
    train_loss.backward()
    optim.step()

initial_context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode_indices(m.sample_sequence(initial_context, 500)[0].tolist()))
