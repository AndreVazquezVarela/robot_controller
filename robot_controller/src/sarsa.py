import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import math
import random
import numpy as np
from std_srvs.srv import Empty
import pickle

class SARSAagent:
    def __init__(self, action_space, state_size):
        self.action_space = action_space
        self.state_size = state_size
        self.alpha = 0.1  # tasa de aprendizaje
        self.gamma = 0.9  # factor de descuento
        self.epsilon = 1.0  # exploración inicial
        self.epsilon_decay = 0.995  # factor de decaimiento de la exploración
        self.epsilon_min = 0.01  # exploración mínima
        self.q_table = np.zeros((state_size[0], state_size[1], action_space))
        self.last_position = None  # Para almacenar la posición al inicio de la época

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state[0], state[1], action]
        next_value = self.q_table[next_state[0], next_state[1], next_action]
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_value)
        self.q_table[state[0], state[1], action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

class SARSA_Turtlebot:
    def __init__(self):
        rospy.init_node('sarsa_turtlebot')
        self.velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_subscriber = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_subscriber = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.rate = rospy.Rate(10)  # 10 Hz
        self.reset_simulation_service = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        self.action_space = 3  # Número de acciones posibles (avanzar, girar izquierda, girar derecha)
        self.state_size = (10, 10)  # Tamaño del espacio de estados (bins para la posición x e y)

        self.agent = SARSAagent(self.action_space, self.state_size)
        self.current_state = None
        self.next_state = None
        self.current_action = None
        self.next_action = None
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

        # Convertir posición a estado (bins)
        state_x = int(x_position * 10)
        state_y = int(y_position * 10)
        self.current_state = (state_x, state_y)

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

        # Actualizar Q-Table
        if self.current_state is not None and self.next_state is not None:
            self.agent.update_q_table(self.current_state, self.current_action, reward, self.next_state, self.next_action)
            self.agent.decay_epsilon()

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
            self.next_action = None
            self.crashed = False  # Reiniciar la bandera de choque
            self.total_reward = 0  # Reiniciar la recompensa acumulada
            self.episode_start_position = None  # Reiniciar la posición inicial del episodio
            while not rospy.is_shutdown() and not self.crashed:
                if self.current_state is not None:
                    self.current_action = self.agent.choose_action(self.current_state)
                    # Ejecutar acción basada en la política actual
                    self.execute_action(self.current_action)
                    self.rate.sleep()

            # Calcular la recompensa basada en la diferencia de posición desde el inicio del episodio
            if self.episode_start_position:
                current_position = (self.current_state[0] / 10.0, self.current_state[1] / 10.0)
                start_position = (self.episode_start_position[0], self.episode_start_position[1])
                episode_reward = self.calculate_position_reward(current_position, start_position)
                self.total_reward += episode_reward

            # Imprimir la recompensa acumulada al final de la época
            rospy.loginfo(f"Epoch {epoch + 1} ended. Total reward: {self.total_reward}")
            f = open("sarsa.txt", "a")
            f.write(str(self.total_reward) + ",")
            f.close()
        self.agent.save_model('sarsa_model.pkl')

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
        sarsa_turtlebot = SARSA_Turtlebot()
        sarsa_turtlebot.run(num_epochs=50)  # Ejemplo: entrenar durante 10 épocas
    except rospy.ROSInterruptException:
        pass
