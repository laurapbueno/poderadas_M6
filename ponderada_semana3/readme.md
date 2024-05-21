## Movimentos do Robô

o controle do robô será feito pelas teclas de seta localizadas no lado inferior direito do teclado. A tecla 'espaço' mata o nó de comunicação com o ROS e a tecla 'S' realiza a parada de emergência do sistema.

## Pré-requisitos

1. É necessário que o ROS esteja instalado e com o sistema de trabalho configurado na sua máquina

2. Coloque o seguinte comando em sem terminal para preparar a área do computador
    - comando que roda o WEBOTS (ambiente virtual para testar o código do robô)
    ```
    ros2 launch webots_ros2_turtlebot robot_launch.py
    ```

3. Prepare a área de trabalho utilizando os seguintes comandos:
    ```bash
    colcon build
    source install/local_setup.bash
    ```

4. Instale todas as dependências
    ```bash
    python3 -m pip install -r requirements.txt
    ```

## Instalando o Webots

1. Abra o terminal e insira os seguintes comandos:
    ```bash
    sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-turtlebot3*

    sudo apt install ros-humble-rmw-cyclonedds-cpp

    echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
    ```

2. Depois de rodar esses comandos, insira:
    ```bash
    sudo mkdir -p /etc/apt/keyrings
    cd /etc/apt/keyrings
    sudo wget -q https://cyberbotics.com/Cyberbotics.asc
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/Cyberbotics.asc] https://cyberbotics.com/debian binary-amd64/" | sudo tee /etc/apt/sources.list.d/Cyberbotics.list
    ```
    ```bash
    sudo apt update
    sudo apt install webots
    sudo apt install ros-humble-webots-ros2
    ```

## Executar o Programa

Assim que o ambiente estiver preparado, você pode executar o programa executando o seguinte comando:

```bash
python3 movimento.py
```
- OBS: é importante estar no local da pasta certa; por isso, utilize os comandos do terminal para chegar até a pasta
'../ponderada_semana3/src/backend' 