import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Implementação do MLP com PyTorch
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.hidden_layer = nn.Linear(2, 2)
        self.output_layer = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# Dados de entrada e saída
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Convertendo os dados para tensores
inputs = torch.tensor(inputs, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)

# Instanciando o modelo MLP
model = XOR_MLP()

# Definindo a função de custo e o otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Treinamento do modelo
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Testando o modelo treinado
with torch.no_grad():
    outputs = model(inputs)
    predicted = (outputs > 0.5).float()
    for input, output, target in zip(inputs, predicted, targets):
        print(f'Input: {input.tolist()}, Predicted: {output.tolist()[0]}, Target: {target.tolist()[0]}')
