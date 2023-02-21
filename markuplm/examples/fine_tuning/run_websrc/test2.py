import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
import nevergrad as ng
import random

from datasets import load_dataset

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)

# Define some hyperparameters
batch_size = 32
learning_rate = 2e-5
num_epochs = 3

# Load data
# imdb = load_dataset("imdb")
train_data = ["This is a positive sentence.", "This is a negative sentence.", "This is also a negative sentence."]
train_labels = [1, 0, 0]

# Tokenize data
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
train_encodings = tokenizer(train_data, truncation=True, padding=True)

train_dataset = list(zip(train_encodings["input_ids"], train_encodings["attention_mask"], train_labels))
print (train_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


# Split model parameters into those optimized with AdamW and those optimized with Nevergrad
adamw_params = [n for n, p in model.named_parameters() if n.startswith('bert')]
nevergrad_params = [n for n, p in model.named_parameters() if n.startswith('classifier')]
adamw_optimizer = optim.AdamW(params=[p for n, p in model.named_parameters() if n in adamw_params], lr=learning_rate)
# nevergrad_optimizer = optimizerlib.registry["TwoPointsDE"](instrumentation=len(nevergrad_params), budget=1000)
# nevergrad_optimizer = ng.optimizers.TwoPointsDE(parametrization=ng.p.Array(init=0), budget=1000)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Train model
for epoch in range(num_epochs):
    for batch in train_loader:
        print (batch)
        input_ids, attention_mask, labels = batch
        adamw_optimizer.zero_grad()
        # nevergrad_params_values = nevergrad_optimizer.ask()
        # if isinstance(nevergrad_params_values, tuple):
        #     nevergrad_params_values = nevergrad_params_values[0]
        # model_state_dict = model.state_dict()
        # for i, n in enumerate(nevergrad_params):
        #     param_shape = model_state_dict[n].shape
        #     param_values = torch.tensor(nevergrad_params_values[i].value, dtype=torch.float32).numpy()
        #     if len(param_shape) == 1:
        #         model_state_dict[n] = torch.tensor(param_values, dtype=torch.float32).view_as(model_state_dict[n])
        #     else:
        #         num_param = model_state_dict[n].numel()
        #         index = torch.tensor(list(range(num_param)))
        #         model_state_dict[n] = torch.index_select(torch.tensor(param_values, dtype=torch.float32), dim=0, index=index).view(param_shape)
        # model.load_state_dict(model_state_dict)
        model.train()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        adamw_optimizer.step()
        # nevergrad_optimizer.tell(nevergrad_params_values, -loss.detach().item())
    print(f"Epoch {epoch+1} loss: {loss.item()}")

