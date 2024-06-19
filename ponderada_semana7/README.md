# Ponderada Semana 07 | Detecção de Algarismos

## Estrutura do Projeto

- 'venv/': Ambiente virtual para isolamento das dependências do projeto
- 'modelos/': Diretório contendo os modelos de treinamento
- 'static/imagens/': Diretório para armazenar imagens de exemplo
- 'templates': Diretório contendo o template HTML
- 'train_convolucional.py': Script para treinamento do modelo convolucional.
- 'train_linear.py': Script para treinamento do modelo linear.
- 'app.py': Backend em Flask.
- 'requirements.txt': Arquivo de dependências
- 'README.md': Documentação do projeto


## Instruções

### Configuração do Ambiente Virtual

1. Crie e ative um ambiente virtual:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

### Treinamento dos Modelos

1. Execute o script de treinamento do modelo convolucional:
    ```bash
    python train_convolucional.py
    ```

2. Execute o script de treinamento do modelo linear:
    ```bash
    python train_linear.py
    ```

### Executando o Backend

1. Inicie o servidor Flask:
    ```bash
    python app.py
    ```

2. Acesse a interface de usuário em `http://127.0.0.1:5000/` para enviar uma imagem de algarismo e receber a predição.

## Comparação dos Modelos

Os resultados da comparação entre os modelos convolucional e linear são apresentados abaixo:

- **Tempo de Treinamento:**
    - Modelo Convolucional: X segundos
    - Modelo Linear: Y segundos

- **Tempo de Inferência:**
    - Modelo Convolucional: X segundos
    - Modelo Linear: Y segundos

- **Desempenho:**
    - Modelo Convolucional: Acurácia - X%, Perda - Y
    - Modelo Linear: Acurácia - X%, Perda - Y