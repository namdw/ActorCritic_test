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
			layer1 = tf.nn.tanh(tf.matmul(self._traj, W1) + b1)
			W2 = tf.get_variable("W2", shape=[h1_size, h2_size], initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.get_variable("b2", shape=[h2_size], initializer=tf.constant_initializer(0.0))
			layer2 = tf.nn.tanh(tf.matmul(layer1, W2) + b2)
			W3 = tf.get_variable("W3", shape=[h2_size, self.output_size], initializer=tf.contrib.layers.xavier_initializer())
			b3 = tf.get_variable("b3", shape=[self.output_size], initializer=tf.constant_initializer(0.0))
			self._C = tf.nn.sigmoid(tf.matmul(layer2, W3) + b3)
		
		self._lossC = -(tf.reduce_mean(tf.log(self._C))+tf.reduce_mean(tf.log(1-self._DE)))
		self._trainC = tf.train.AdamOptimizer(learning_rate=lr).minimize(self._lossC)

	def critique(self, state_action, expt):
		traj = np.reshape(state_action, [1, self.input_size])
		return self.session.run(self._C, feed_dict={self._traj: traj})

	def update(self, state_action, stateE_actionE):

		traj = state_action
		return self.session.run([self._lossC, self._trainC], feed_dict={self._traj:traj})

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
		

	def update(self, state, action_mat, Q):
		return self.session.run([self._lossPI, self._trainPI], feed_dict={self._state:state, self._action:action_mat, self._Q:Q})


def main():

	env = gym.make('CartPole-v0')
	
	input_size_C = env.observation_space.shape[0] + env.action_space.n
	output_size_C = 1
	input_size_A = env.observation_space.shape[0]
	output_size_A = env.action_space.n

	max_iter = 5
	max_episode = 100
	
	with tf.Session() as sess:
		critic = Critic(sess, input_size_C, output_size_C, "Critic")
		actor = PInet(sess, input_size_A, output_size_A, "Actor")
		tf.global_variables_initializer().run()

		for _ in range(max_iter):
			# re-initializing the training environment
			for _ in range(max_episode):
				done = False
				state = env.reset()

				traj = np.empty(0).reshape(0, input_size_C)
				state_stack = np.empty(0).reshape(0, input_size_A)
				action_stack = np.empty(0).reshape(0, output_size_A)
				Q_stack = np.empty(0).reshape(0, 1)

				step_counter = 0
				while True:
					step_counter += 1
					env.render()
					action_prob = actor.act(state)[0]
					# print(action_prob)
					a = [1, 0] if(action_prob[0] > action_prob[1]) else [0, 1]
					
					traj = np.vstack([traj, np.concatenate((state,a))])
					state_stack = np.vstack([state_stack, state])
					action_stack = np.vstack([action_stack, a])

					state, reward, done, _ = env.step(a[1])

					if done or step_counter > 300:
						print("Score of {}".format(len(traj)))
						# if(len(traj)<=10):
						# 	print(action_stack)

						# discriminator update
						critic.update(traj)
						# using updated discriminator, calculate the Q value
						D_SUM = 0
						D_ct = 0
						for j in range(traj.shape[0]+1,0,-1):
							C_ct += 1
							C_SUM += np.log(discriminator.discriminate(traj[j-2], 0))
							A = C_SUM/C_ct
							A_stack = np.vstack([A_stack, A])
						A_stack = np.flipud(A_stack)
						# actor update
						actor.update(state_stack, action_stack, A_stack)
						break
					

if __name__ == "__main__":
	main()