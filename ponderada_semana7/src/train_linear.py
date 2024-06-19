from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Carregando o dataset e separando os dados de treino e de teste
(x_treino, y_treino), (x_teste, y_teste) = mnist.load_data()

# Normalização dos dados de entrada
x_treino_norm = x_treino / x_treino.max()
x_teste_norm = x_teste / x_teste.max()

# Reshape dos dados de entrada para um vetor
x_treino_norm = x_treino_norm.reshape(len(x_treino_norm), 28 * 28)
x_teste_norm = x_teste_norm.reshape(len(x_teste_norm), 28 * 28)

# Transformar os labels em one-hot encoding
y_treino_cat = to_categorical(y_treino)
y_teste_cat = to_categorical(y_teste)

# Criação do modelo linear
model = Sequential()
model.add(Dense(512, input_dim=28 * 28, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compilação do modelo
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
model.fit(x_treino_norm, y_treino_cat, epochs=10, validation_data=(x_teste_norm, y_teste_cat))

# Salvando os pesos do modelo
model.save('modelos/pesos_linear.h5')
