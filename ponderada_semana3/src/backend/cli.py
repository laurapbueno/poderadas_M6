import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import typer
import inquirer

app = typer.Typer()

class TeleopRobo(Node):
    def __init__(self):
        super().__init__('turtlebot3_teleop')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.twist_msg = Twist()

    def send_cmd_vel(self, linear_velo, angular_velo):
        self.twist_msg.linear.x = linear_velo
        self.twist_msg.angular.z = angular_velo
        self.publisher_.publish(self.twist_msg)
        print(f"Velocidade Linear: {linear_vel}0, Velocidade Angular: {angular_vel0}")

@app.command()
def control():
    rclpy.init()
    node = TeleopRobo()
    print("Controle do Turtle")
    questions = [inquirer.List(
        name='command',
        message='Selecione uma opção:',
        choices=['Frente', 'Trás', 'Esquerda', 'Direita', 'Parada de Emergência', 'Sair'])]
    try:
        while True:
            command = inquirer.prompt(questions)['command']
            match command:
                case 'Sair':
                    break
                case 'Frente':
                    node.send_cmd_vel(0.2, 0.0) 
                case 'Trás':
                    node.send_cmd_vel(-0.2, 0.0)
                case 'Esquerda':
                    node.send_cmd_vel(0.0, 0.5)
                case 'Direita':
                    node.send_cmd_vel(0.0, -0.5)
                case 'Parada de Emergência':
                    node.send_cmd_vel(0.0, 0.0)
    except Exception as e:
        print('Parada de Emergência, encerrando movimentação do robô')
        node.send_cmd_vel(0.0, 0.0) 
    finally:
        node.send_cmd_vel(0.0, 0.0)
        rclpy.shutdown()

if __name__ == "__main__":
    app()