import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from std_srvs.srv import Empty

# Definir la arquitectura de la red neuronal para A2C
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc_actor = nn.Linear(64, action_size)
        self.fc_critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99  # factor de descuento
        self.lr = 0.001  # tasa de aprendizaje
        self.model = ActorCritic(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.model(state_tensor)
        action = torch.multinomial(action_probs, 1)
        return action.item()

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def learn(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        _, current_state_value = self.model(state_tensor)
        _, next_state_value = self.model(next_state_tensor)

        target = reward + self.gamma * next_state_value * (1 - int(done))
        advantage = target - current_state_value

        action_probs, state_value = self.model(state_tensor)
        log_probs = torch.log(action_probs.squeeze(0)[action])

        actor_loss = -log_probs * advantage
        critic_loss = F.smooth_l1_loss(state_value, torch.tensor([target]))

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class A2C_Turtlebot:
    def __init__(self):
        rospy.init_node('a2c_turtlebot')
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = 3  # Número de acciones posibles (avanzar, girar izquierda, girar derecha)
        self.state_size = 2  # Tamaño del espacio de estados (posición x, posición y)

        self.agent = A2CAgent(self.state_size, self.action_space)
        self.current_state = None
        self.next_state = None
        self.current_action = None
        self.crashed = False
        self.total_reward = 0  # Inicializar la recompensa acumulada
        self.episode_start_position = None

    def odom_callback(self, odom_msg):
        # Obtener posición actual del robot
        # Suponiendo que se extrae la posición x e y del mensaje de odometría
        x_position = odom_msg.pose.pose.position.x
        y_position = odom_msg.pose.pose.position.y

        # Guardar la posición inicial al comienzo del episodio
        if self.episode_start_position is None:
            self.episode_start_position = (x_position, y_position)

        self.current_state = (x_position, y_position)

    def laser_callback(self, laser_msg):
        # Procesar datos del sensor láser para determinar la proximidad a las paredes
        # Suponiendo que se calcula la distancia mínima al obstáculo
        min_distance = min(laser_msg.ranges)

        # Definir recompensas y penalizaciones
        reward = 0
        if min_distance < 0.19:
            self.crashed = True  # Marcar que se ha chocado

        # Acumular recompensa
        if not self.crashed:
            reward += 1  # Recompensa por seguir en movimiento

        # Aprender con A2C
        if self.current_state is not None and self.next_state is not None and self.current_action is not None:
            self.agent.learn(self.current_state, self.current_action, reward, self.next_state, self.crashed)

    def reset_simulation(self):
        # Reiniciar la simulación llamando al servicio de reset de Gazebo
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            self.reset_simulation_service()
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def run(self, num_epochs):
        for epoch in range(num_epochs):
            self.reset_simulation()
            rospy.loginfo(f"Starting epoch {epoch + 1}")
            self.current_state = None
            self.next_state = None
            self.current_action = None
            self.crashed = False  # Reiniciar la bandera de choque
            self.total_reward = 0  # Reiniciar la recompensa acumulada
            self.episode_start_position = None  # Reiniciar la posición inicial del episodio
            while not rospy.is_shutdown() and not self.crashed:
                if self.current_state is not None:
                    self.current_action = self.agent.act(self.current_state)
                    # Ejecutar acción basada en la política actual
                    self.execute_action(self.current_action)
                    self.rate.sleep()

            # Calcular la recompensa basada en la diferencia de posición desde el inicio del episodio
            if self.episode_start_position:
                current_position = self.current_state
                start_position = self.episode_start_position
                episode_reward = self.calculate_position_reward(current_position, start_position)
                self.total_reward += episode_reward

            # Imprimir la recompensa acumulada al final de la época
            rospy.loginfo(f"Epoch {epoch + 1} ended. Total reward: {self.total_reward}")
            f = open("a2c.txt", "a")
            f.write(str(self.total_reward) + ",")
            f.close()

    def calculate_position_reward(self, current_position, start_position):
        # Calcular la recompensa basada en la diferencia de posición desde el inicio del episodio
        # Aquí puedes definir tu propia función de recompensa basada en la diferencia de posición
        # Ejemplo: recompensa proporcional a la distancia recorrida desde el inicio del episodio
        start_x, start_y = start_position
        current_x, current_y = current_position
        distance_traveled = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
        return distance_traveled

    def execute_action(self, action):
        # Definir las acciones posibles (ej. publicar velocidades en /cmd_vel)
        twist_msg = Twist()
        if action == 0:  # Avanzar
            twist_msg.linear.x = 0.2
            twist_msg.angular.z = 0.0
        elif action == 1:  # Girar a la izquierda (solo eje z)
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.5
        elif action == 2:  # Girar a la derecha (solo eje z)
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = -0.5

        self.velocity_publisher.publish(twist_msg)

if __name__ == '__main__':
    try:
        a2c_turtlebot = A2C_Turtlebot()
        a2c_turtlebot.run(num_epochs=50)  # Ejemplo: entrenar durante 10 épocas
        a2c_turtlebot.agent.save_model('a2c_model.pth')
    except rospy.ROSInterruptException:
        pass
