# Quick write-up of actor critic algorithm

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 

import gym

class Critic:
	def __init__(self, session, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = name

		self._build_net()

	def _build_net(self, h1_size=100, h2_size=100, lr=1e-3):
		with tf.variable_scope(self.name):
			self._traj = tf.placeholder(tf.float32, [None, self.input_size])
			W1 = tf.get_variable("W1", shape=[self.input_size, h1_size], initializer=tf.contrib.layers.xavier_initializer())
			b1 = tf.get_variable("b1", [h1_size], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.relu(tf.matmul(self._traj, W1) + b1)
			W2 = tf.get_variable("W2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.get_variable("b2", shape=[h2_size], initializer=tf.constant_initializer(0.0))
			layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
			W3 = tf.get_variable("W3", shape=[h2_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			b3 = tf.get_variable("b3", shape=[self.output_size], initializer=tf.constant_initializer(0.0))

			self._C = tf.nn.relu(tf.matmul(layer2, W3) + b3)
			self._Q = tf.placeholder(tf.float32, [None, 1])
		
		self._lossC = tf.reduce_mean(self._Q-self._C)
		self._trainC = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._lossC)

	def critique(self, state_action):
		traj = np.reshape(state_action, [1, self.input_size])
		return self.session.run(self._C, feed_dict={self._traj: traj})

	def update(self, state_action, Q):
		traj = state_action
		return self.session.run([self._lossC, self._trainC], feed_dict={self._traj:traj, self._Q:Q})

class Actor:
	def __init__(self, session, input_size, output_size, name):
		self.session = session
		self.input_size = input_size
		self.output_size = output_size
		self.name = name

		self._build_net()

	def _build_net(self, h1_size=100, h2_size=100, lr=1e-3, lamb=0):
		with tf.variable_scope(self.name):
			self._state = tf.placeholder(tf.float32, [None, self.input_size])
			self._action = tf.placeholder(tf.float32, [None, self.output_size])
			W1 = tf.get_variable("W1", shape=[self.input_size, h1_size], initializer=tf.contrib.layers.xavier_initializer())
			b1 = tf.get_variable("b1", shape=[h1_size], initializer=tf.constant_initializer(0.0))
			layer1 = tf.nn.tanh(tf.matmul(self._state, W1) + b1)
			W2 = tf.get_variable("W2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.get_variable("b2", shape=[h2_size], initializer=tf.constant_initializer(0.0))
			layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
			W3 = tf.get_variable("W3", shape=[h2_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			b3 = tf.get_variable("b3", shape=[self.output_size], initializer=tf.constant_initializer(0.0))
			self._PI = tf.nn.softmax(tf.matmul(layer2, W3) + b3)
			self._A = tf.placeholder(tf.float32, [None, 1])

		self._pa = tf.reduce_max(tf.multiply(self._PI, self._action))

		self._lossPI = -(tf.reduce_mean(tf.log(self._pa)*self._A))
		self._trainPI = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._lossPI)

	def act(self, state):
		x = np.reshape(state, [1, self.input_size])
		return self.session.run(self._PI, feed_dict={self._state: x})
		

	def update(self, state, action_mat, A):
		return self.session.run([self._lossPI, self._trainPI], feed_dict={self._state:state, self._action:action_mat, self._A:A})

def discounted_Q(r_traj, dc_factor):
	q_sum = 0
	for i in range(len(r_traj)):
		q_sum += r_traj[i] * np.power(dc_factor, i)
	return q_sum


def main():

	env = gym.make('CartPole-v0')
	# env = gym.make('MountainCar-v0')
	print(env.spec.id)
	
	input_size_C = env.observation_space.shape[0] + env.action_space.n
	output_size_C = 1
	input_size_A = env.observation_space.shape[0]
	output_size_A = env.action_space.n

	max_episode = 2000
	
	with tf.Session() as sess:
		critic = Critic(sess, input_size_C, output_size_C, "Critic")
		actor = Actor(sess, input_size_A, output_size_A, "Actor")
		tf.global_variables_initializer().run()

		for i in range(max_episode):
			done = False
			state = env.reset()

			traj = np.empty(0).reshape(0, input_size_C)
			state_stack = np.empty(0).reshape(0, input_size_A)
			action_stack = np.empty(0).reshape(0, output_size_A)
			reward_stack = np.empty(0).reshape(0, 1)
			Q_stack = np.empty(0).reshape(0, 1)
			A_stack = np.empty(0).reshape(0, 1)

			step_counter = 0
			while True:
				step_counter += 1
				if i%50==0:
					env.render()
				action_prob = actor.act(state)[0]
				# print(action_prob)
				# a = [1, 0] if(action_prob[0] > action_prob[1]) else [0, 1]
				a = np.random.choice(output_size_A, p=action_prob)
				temp_a = np.zeros(output_size_A)
				temp_a[a] = 1

				traj = np.vstack([traj, np.concatenate((state,temp_a))])
				state_stack = np.vstack([state_stack, state])
				action_stack = np.vstack([action_stack, temp_a])

				# state, reward, done, _ = env.step(a[1])
				state, reward, done, _ = env.step(a)
				if done:
					reward -= 100
					# if(step_counter==200):
					# 	reward -= 100
					# else:
					# 	reward += 100
				reward_stack = np.vstack([reward_stack, reward])
				Q_value = discounted_Q(reward_stack, 0.95)
				Q_stack = np.vstack([Q_stack, Q_value])
				A_stack = np.vstack([A_stack, Q_value-critic.critique(np.concatenate((state,temp_a)))])

				if done or step_counter > 200:
					print("Score of {}".format(len(traj)))
					# if(len(traj)<=10):
					# 	print(action_stack)

					# discriminator update
					critic.update(traj, Q_stack)
					# using updated discriminator, calculate the Q value
					# actor update
					actor.update(state_stack, action_stack, A_stack)
					break
					

if __name__ == "__main__":
	main()