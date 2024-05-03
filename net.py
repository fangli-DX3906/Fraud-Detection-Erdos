import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchviz import make_dot

# data-parse
with open('data.pkl', 'rb') as file:
    data_dict = pickle.load(file)

original_data = data_dict['original_sample']
under_sample = data_dict['under_sample']
over_sample = data_dict['over_sample']
which_data_to_train = over_sample
which_data_to_test = original_data


# data-loader
class CreditFraud(Dataset):
    def __init__(self, data: dict, training=True):
        if training:
            x_ind = 0
            y_ind = 2
        else:
            x_ind = 1
            y_ind = 3

        try:
            self.X_train = data[x_ind]
            self.y_train = data[y_ind]
        except:
            raise ValueError

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        x, y = self.X_train[idx, :], self.y_train[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# net work structure
class CardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# set up the parameter
input_dim = 32
hidden_dim = 16
output_dim = 1
epochs = 10

# initialzation for net, optimizer, and dataloader
carddata = CreditFraud(which_data_to_train)
train_loader = DataLoader(carddata, batch_size=2, shuffle=True)

carddata_for_test = CreditFraud(which_data_to_test, False)
test_loader = DataLoader(carddata_for_test, batch_size=2, shuffle=True)

model = CardNN(input_dim, hidden_dim, output_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train the model
count = 0
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        count += 1
        print(f'in epoch={epoch}, count={count}')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float().squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader.dataset)}")

# test the model
# model = torch.load('over_sampled.pth')
model.eval()
correct = 0
total = 0
count = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        count += 1
        print(f'count={count}')
        outputs = model(inputs)
        predicted = torch.round(torch.sigmoid(outputs))
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).sum().item()
accuracy = correct / total
print(f"Test Accuracy: {accuracy}")

# another way to test the accuracy
N = 1000
T = original_data[1].shape[0]
n = []
for i in range(100):
    positive = 0
    for j in range(N):
        idx = random.choice(range(T))
        x = torch.tensor(original_data[1][idx, :], dtype=torch.float32)
        y = torch.tensor(original_data[3][idx], dtype=torch.float32)
        predicted = torch.round(torch.sigmoid(model(x))).squeeze()
        positive += (y == predicted).item()
    print(f"Test Accuracy: {positive / N}")
    n.append(positive / N)

plt.plot(n)
plt.title('Accuracy: the whole Sample')
plt.show()

# positive and negative sample
x_test_plus_ind = np.where(original_data[3] == 1)
x_test_minus_ind = np.where(original_data[3] == 0)
x_test_plus = original_data[1][x_test_plus_ind[0], :]
x_test_minus = original_data[1][x_test_minus_ind[0], :]
y_test_plus = original_data[3][x_test_plus_ind[0]]
y_test_minus = original_data[3][x_test_minus_ind[0]]

# test default sample
N = 1000
T = x_test_plus.shape[0]
m = []
for j in range(100):
    positive = 0
    for i in range(N):
        idx = random.choice(range(T))
        x = torch.tensor(x_test_plus[idx, :], dtype=torch.float32)
        y = torch.tensor(y_test_plus[idx], dtype=torch.float32)
        predicted = torch.round(torch.sigmoid(model(x))).squeeze()
        positive += (y == predicted).item()
    print(f"Test Accuracy: {positive / N}")
    m.append(positive / N)

plt.plot(m)
plt.title('Accuracy: the Positive Sample')
plt.show()

# save the trained model
torch.save(model, 'over_sampled.pth')

# visualize
output = model(x)
mg = make_dot(output, params=dict(model.named_parameters()))
mg.render("model_graph", format="png")
