#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
import sys, select, tty, termios

BURGER_MAX_LIN_VEL = 0.22
BURGER_MAX_ANG_VEL = 2.84
LIN_VEL_STEP_SIZE = 0.01
ANG_VEL_STEP_SIZE = 0.1

msg = """
Controle seu robô
---------------------------
Movimentação:
    ↑
←       →
    ↓

↑ : aumentar linear velocity (Burger : ~ 0.22)
↓ : diminuir linear velocity (Burger : ~ 0.22)
← : aumentar angular velocity (Burger : ~ 2.84)
→ : diminuir angular velocity (Burger : ~ 2.84)

tecla 's' : parada de Emergência
tecla de espaço : encerrar o nó

CTRL-C para sair
"""

e = """
Communications Failed
"""

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
        if key == '\x1b':
            key += sys.stdin.read(2)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def vels(target_linear_vel, target_angular_vel):
    return f"currently:\tlinear vel {target_linear_vel}\t angular vel {target_angular_vel}"

def makeSimpleProfile(output, input, slop):
    if input > output:
        output = min(input, output + slop)
    elif input < output:
        output = max(input, output - slop)
    else:
        output = input
    return output

def constrain(input, low, high):
    if input < low:
        input = low
    elif input > high:
        input = high
    return input

def checkLinearLimitVelocity(vel):
    return constrain(vel, -BURGER_MAX_LIN_VEL, BURGER_MAX_LIN_VEL)

def checkAngularLimitVelocity(vel):
    return constrain(vel, -BURGER_MAX_ANG_VEL, BURGER_MAX_ANG_VEL)


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 30)
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.control_linear_vel = 0.0
        self.control_angular_vel = 0.0
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info(msg)
        self.timer = self.create_timer(0.1, self.update)

        self.stop_service = self.create_service(Empty, 'stop_robot', self.stop_robot_callback)

    def stop_robot_callback(self, request, response):
        self.target_linear_vel = 0.0
        self.target_angular_vel = 0.0
        self.control_linear_vel = 0.0
        self.control_angular_vel = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        self.publisher.publish(twist)
        self.get_logger().info("Robot stopped via service call")
        rclpy.shutdown()
        return response

    def update(self):
        key = getKey(self.settings)
        if key == '\x1b[A': 
            self.target_linear_vel = checkLinearLimitVelocity(self.target_linear_vel + LIN_VEL_STEP_SIZE)
            self.get_logger().info(vels(self.target_linear_vel, self.target_angular_vel))
        elif key == '\x1b[B':  
            self.target_linear_vel = checkLinearLimitVelocity(self.target_linear_vel - LIN_VEL_STEP_SIZE)
            self.get_logger().info(vels(self.target_linear_vel, self.target_angular_vel))
        elif key == '\x1b[D':  
            self.target_angular_vel = checkAngularLimitVelocity(self.target_angular_vel + ANG_VEL_STEP_SIZE)
            self.get_logger().info(vels(self.target_linear_vel, self.target_angular_vel))
        elif key == '\x1b[C':  
            self.target_angular_vel = checkAngularLimitVelocity(self.target_angular_vel - ANG_VEL_STEP_SIZE)
            self.get_logger().info(vels(self.target_linear_vel, self.target_angular_vel))
        elif key == 's':  
            self.target_linear_vel = 0.0
            self.target_angular_vel = 0.0
            self.control_linear_vel = 0.0
            self.control_angular_vel = 0.0
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            self.get_logger().info("Robo parado pela tecla pressionada 's'")
        elif key == ' ':  
            rclpy.shutdown()  
            self.get_logger().info("Desligamento do nó pressionando a tecla 'espaço'")
        else:
            if key == '\x03':  
                rclpy.shutdown()
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
                return

        twist = Twist()
        self.control_linear_vel = makeSimpleProfile(self.control_linear_vel, self.target_linear_vel, (LIN_VEL_STEP_SIZE / 2.0))
        twist.linear.x = self.control_linear_vel
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        self.control_angular_vel = makeSimpleProfile(self.control_angular_vel, self.target_angular_vel, (ANG_VEL_STEP_SIZE / 2.0))
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = self.control_angular_vel

        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)  
    node = TeleopNode()  
    try:
        rclpy.spin(node)  
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()  
        rclpy.shutdown()  
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node.settings)  

if __name__ == '__main__':
    main()  