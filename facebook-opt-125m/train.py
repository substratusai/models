from PromptData import PromptData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
from infer import Inferer

import sys

data_path = sys.argv[1]

tokenizer = torch.load('/model/tokenizer.pt')
model = torch.load('/model/model.pt')

ds = PromptData(data_path, tokenizer)
dl = DataLoader(ds, batch_size=1)

epochs = 3

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def train(dat, model, optim):
    for i in tqdm.tqdm(range(epochs)):
        for X, a in dat:
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        #torch.save(model.state_dict(), "model_state.pt")

print("training .... ")
model.train()
optim = Adam(model.parameters(), lr=1e-3)
train(dl, model, optim)

torch.save(model, '/trained/model.pt')

print(Inferer(model, tokenizer, device).infer("What is your favorite color?", 10))
