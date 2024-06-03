import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Função de ativação sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Implementação do MLP manual
def manual_xor_mlp():
    # Dados de entrada e saída para o problema XOR
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # Pesos para a camada oculta e camada de saída
    hidden_weights = np.array([[0.5, 0.5], [0.5, 0.5]])
    output_weights = np.array([[0.5], [-0.5]])

    # Bias para a camada oculta e camada de saída
    hidden_bias = np.array([0, 0])
    output_bias = np.array([0])

    # Forward pass
    hidden_layer_output = sigmoid(np.dot(inputs, hidden_weights.T) + hidden_bias)
    output = sigmoid(np.dot(hidden_layer_output, output_weights.T) + output_bias)

    # Imprimir resultados
    for input, out, target in zip(inputs, output, targets):
        print(f'Input: {input.tolist()}, Predicted: {out.tolist()[0]}, Target: {target.tolist()[0]}')

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

# Treinamento e teste do modelo PyTorch
def pytorch_xor_mlp():
    # Dados de entrada e saída para o problema XOR
    inputs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

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

if __name__ == "__main__":
    print("Manual XOR MLP:")
    manual_xor_mlp()

    print("\nPyTorch XOR MLP:")
    pytorch_xor_mlp()
