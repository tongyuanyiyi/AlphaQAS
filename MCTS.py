import numpy as np
import pdb
import cirq

# Complementary Code J = 2, M = 2, N = 8
c_p = 1
from typing import Dict, List, Optional, Union

# pulse compression radar
# c_p = 1/np.sqrt(2)

class Node():
	def __init__(self, selfPlay, cummulativeMove, num_Moves, n_actions, evaluating_fn, calc_reward, fromlink, env, parent=None):
		# a node has a state (of length num_Moves), a parent
		self.selfPlay = selfPlay
		self.move = cummulativeMove  # 生成当前状态，所需要的动作
		self.num_Moves = num_Moves   # 添加门所需要步骤
		self.n_actions = n_actions   # 动作数
		self.evaluating_fn = evaluating_fn # 神经网络
		self.calc_reward = calc_reward  # 奖励函数
		self.fromlink = fromlink  #生成当前状态的前一步动作
		self.parent = parent   #父节点
		# a node has n_actions edges
		self.N_sa = np.zeros(self.n_actions) # N(s,a)
		self.W_sa = np.zeros(self.n_actions) # W(s,a)
		self.Q_sa = np.zeros(self.n_actions) # Q(s,a)
		# evaluate this node using DNN, to get self.Prior_sa and value v
		self.env = env
		self.qubits = cirq.LineQubit.range(2)
		self.state_observables = self.get_default_observables()
		self.error_observables = 0.001
		self.error_gates = 0.001
		self.state = self.get_obs(maybe_add_noise=False)

		if self.terminalNode() == True:
			self.Prior_sa = np.zeros([1, n_actions]) - 2 # of no use
			circuit_gates = []
			for mov in self.move:
				circuit_gates.append(self.env.action_gates[int(mov)])
			_, self.value, _ = self.calc_reward(circuit_gates,self.qubits, self.error_gates, self.error_observables, False)
		else:
			# 如果新添加的节点不是最终节点，使用神经网络来估计该节点的奖励。(可以换一种方法来估计节点的奖励)
			self.Prior_sa, self.value = self.evaluating_fn(self.state.reshape([1,len(self.state)]), self.selfPlay)
			self.reserve_Pr = self.Prior_sa

		# add noise to root node
		if self.selfPlay == 1 and self.parent == None:
			self.root_noise()

		# a node has at most n_actions children, initialize to 0 children
		self.children = []
		self.seenChildIndex = []

	def get_cirq_circuit(self, circuit_gates, maybe_add_noise=False):
		circuit = cirq.Circuit(cirq.I(qubit) for qubit in self.qubits)
		for gate in circuit_gates:
			circuit.append(gate)
			if maybe_add_noise and (self.error_gates is not None):
				noise_gate = cirq.depolarize(
					self.error_gates).on_each(*gate._qubits)
				circuit.append(noise_gate)
		if maybe_add_noise and (self.error_observables is not None):
			noise_observable = cirq.bit_flip(
				self.error_observables).on_each(*self.qubits)
			circuit.append(noise_observable)
		return circuit

	def get_default_observables(self) -> List[cirq.GateOperation]:
		observables = []
		for qubit in self.qubits:
			observables += [
				cirq.X(qubit),
				cirq.Y(qubit),
				cirq.Z(qubit),
			]
		return observables

	def get_obs(self ,  maybe_add_noise=False):
		circuit_gates = []
		for mov in self.move:

			circuit_gates.append(self.env.action_gates[int(mov)])
		circuit = self.get_cirq_circuit(circuit_gates, maybe_add_noise)
		obs = cirq.Simulator().simulate_expectation_values(
			circuit, observables=self.state_observables)
		return np.array(obs).real
	def root_noise(self):
		self.Prior_sa = 0.75*self.reserve_Pr + 0.25*np.random.dirichlet(alpha0*np.ones(len(self.reserve_Pr[0])))

	def terminalNode(self):
		if len(self.move) == self.num_Moves : # the height of the tree <= num_Moves
			return True
		return False

	def add_child(self, cummulativeMove, env, fromlink):
		child = Node(self.selfPlay, cummulativeMove, self.num_Moves, self.n_actions, self.evaluating_fn, self.calc_reward, fromlink, env,  self)
		self.children.append(child)

	def __repr__(self):
		s="I am a node\nmy move is %s\nI have %d children"%(self.move, len(self.children))
		return s

def bestMove(node):
	QplusU = node.Q_sa + c_p * np.sqrt(np.sum(node.N_sa)) * node.Prior_sa[0] / (1 + node.N_sa)
	# QplusU = node.Q_sa + c_p * np.sqrt(np.sum(node.N_sa))  / (1 + node.N_sa)

	bestvalue = np.max(QplusU)
	bestmoves = []

	for index, value in enumerate(QplusU):
		if value == bestvalue:
			bestmoves.append(index)

	try:
		moveIndex = np.random.choice(bestmoves)
	except:
		pdb.set_trace()

	return moveIndex

def cal_piVec(N_sa, tau = 0):
	if tau == 1:
		probs = N_sa/np.sum(N_sa)
	elif tau == 0:
		probs = np.zeros(len(N_sa))
		probs[np.argmax(N_sa)] = 1
	else:
		probs = softmax(1.0/tau * np.log(N_sa))

	return probs

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def Backup(node, value):
	# we update the edges, rather than nodes
	while node.parent != None:
		pos = node.fromlink
		node = node.parent
		node.N_sa[pos] += 1
		node.W_sa[pos] += value
		node.Q_sa[pos] = node.W_sa[pos]/node.N_sa[pos]
		# node.Q_sa[pos] = node.Q_sa[pos] + (value - node.Q_sa[pos]) / node.N_sa[pos] # running average
	return

def TreePolicy(node, VisitedState, flag, env):
	# keep searching along the existing tree until terminal state - Two cases when return:
	# 1. find an unseen child, then return this child
	# 2. search to the bottom of the tree, then return the arriving node
	while node.terminalNode() == False:
		# calculate Q+U to choose an edge, this edge leads to a child
		nextmoveIndex = bestMove(node) # move with maximum Q+U
		if nextmoveIndex in node.seenChildIndex:
			# nextState is already in Tree - return the child and continute searching
			node = node.children[node.seenChildIndex.index(nextmoveIndex)]
		else:
			node.seenChildIndex.append(nextmoveIndex)
			# unseen state - first add a child, then return this new leaf node
			cummulativeMove = np.append(node.move, nextmoveIndex)
			node.add_child(cummulativeMove, env, fromlink = nextmoveIndex)

			if flag == 1:
				VisitedState.store(cummulativeMove)

			return node.children[-1] # return the newly-added leaf node
	return node

def MCTS_main(args, VisitedState, stepSize, n_steps, evaluating_fn, calc_reward, selfPlay):
    global alpha0
    alpha0 = args.alpha
    flag = selfPlay * args.recordState

    # initial state
    cummulativeMove = np.array([])
    temp_store = []

    # Run one episode
    eachstep = 0
    while eachstep < n_steps:
        # temperature parameters in MCTS
        if eachstep <= int(n_steps/3.):
            tau = 1
        else:
            tau = 0

        if eachstep == 0:

        	root = Node(selfPlay, cummulativeMove, n_steps, args.Q, evaluating_fn, calc_reward, None, args.env, parent=None)

        for _ in range(args.simBudget):
            root.root_noise()
            v_l = TreePolicy(root, VisitedState, flag, args.env)	# from v_0 to v_l
            Backup(v_l, v_l.value)	# back propagation
        piVec = cal_piVec(root.N_sa, tau)

        # temporarily store
        currentState = root.state
        if selfPlay == 1:
            temp_store.append([currentState, piVec])

        # update state -> go to next time step
		# 选择下一动作
        nextMove = np.random.choice(args.Q , 1, p = piVec)[0]

		#
        root = root.children[root.seenChildIndex.index(nextMove)]
        root.parent = None
        cummulativeMove = root.move

        eachstep += 1

    return cummulativeMove, temp_store






