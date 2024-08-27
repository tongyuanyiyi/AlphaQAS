from MCTS import MCTS_main
from ConvNets import DeepNN
from StateCNT import Dotdict, HarshState
import numpy as np
import pickle
from collections import deque
import time
import random
import multiprocessing
import pdb

import  gym
from gym import spaces
import qas_gym
import cirq

env_name = 'BasicTwoQubit-v0'
fidelity_threshold = 0.97
reward_penalty = 0.01
max_timesteps = 10
# Environment
env = gym.make(env_name,
               fidelity_threshold=fidelity_threshold,
               reward_penalty=reward_penalty,
               max_timesteps=max_timesteps)
observation = env.reset()
########## 量子线路架构搜索，以bell态为例
args = Dotdict({
        'N': 6, # length of input image to DNN, i.e., 两个量子比特在X，Y，Z测量基下的结果，长度为6，量子线路的状态
        'K': 1, # width of input image to DNN, i.e., K_prime in the paper
        'lpos': 1, # number of positions filled in each time step，每个动作，表示添加1个门
        'M': 3, # feature planes，特征提取
        'Q': 12, # 2 for binary sequence,用于产生动作的数量，一共有12个
        'alpha': 0.05, # exploration noise, i.e., \alpha in the paper
        'simBudget': 300, # MCTS simulation budget
        'eval_games': 20,
        'updateNNcycle': 100, # parameter G
        'zDNN': 3, # parameter z
        'numfilters1': 256,
        'numfilters2': 512,
        'l2_const': 1e-3,
        'batchSize': 64,
        'numEpisode': 5000,
        'isMultiCore': 1,
        'recordState': 1,
        'env':env
        })



overallSteps = 10
stepSize = args.lpos
n_steps = 10 #线路生成一共走多少步
memorySize = n_steps * args.updateNNcycle * args.zDNN

###### 测度函数
worstMetric = 0
bestMetric = 1




######## reward definition - complementary code
def get_cirq_circuit(circuit_gates, qubits, error_gates, error_observables, maybe_add_noise=False):
    circuit = cirq.Circuit(cirq.I(qubit) for qubit in qubits)
    for gate in circuit_gates:
        circuit.append(gate)
        if maybe_add_noise and (error_gates is not None):
            noise_gate = cirq.depolarize(
                error_gates).on_each(*gate._qubits)
            circuit.append(noise_gate)
    if maybe_add_noise and (error_observables is not None):
        noise_observable = cirq.bit_flip(
            error_observables).on_each(*qubits)
        circuit.append(noise_observable)
    return circuit
def calc_reward(circuit_gates, qubits, error_gates, error_observables, maybe_add_noise=False):
    circuit = get_cirq_circuit(circuit_gates, qubits, error_gates, error_observables, maybe_add_noise)
    pred = cirq.Simulator().simulate(circuit).final_state_vector
    # bell态
    target = np.zeros(2**2, dtype=complex)
    target[0] = 1. / np.sqrt(2) + 0.j
    target[-1] = 1. / np.sqrt(2) + 0.j
    inner = np.inner(np.conj(pred),target)
    fidelity = np.conj(inner) * inner
    fidelity =  fidelity.real
    worstMetric = 0
    bestMetric = 1
    #保真度越大越好
    if fidelity >= worstMetric:
        reward = (worstMetric + bestMetric - 2 * fidelity) / (worstMetric - bestMetric) # -1 to 1
    else:
        reward = -1
    return circuit, reward, fidelity


######## DNN player
def DNN_play(n_games, evaluating_fn):
    rewardArray = []
    fidelityArray = []
    # 创建QAS环境
    # Parameters
    env_name = 'BasicTwoQubit-v0'
    fidelity_threshold = 0.97
    reward_penalty = 0.01
    max_timesteps = 10
    # Environment
    env = gym.make(env_name,
                   fidelity_threshold=fidelity_threshold,
                   reward_penalty=reward_penalty,
                   max_timesteps=max_timesteps)
    n_actions = len(env.action_gates)

    for nn in range(n_games):
        circuit_gates = []
        # print('第{}次DNN'.format(nn))
        #初始化
        observation= env.reset()
        for eachstep in range(n_steps):
            # 根据动作序列得到状态
            Prior_sa, value = evaluating_fn(observation.reshape([1, len(observation)]), 0)
            nextMove = np.random.choice(n_actions , 1, p=Prior_sa[0])[0]
            circuit_gates.append(env.action_gates[nextMove])
        qubits = cirq.LineQubit.range(2)
        error_gates = 0.001
        error_observables = 0.001
        circuit, reward, fidelity = calc_reward(circuit_gates, qubits, error_gates, error_observables, maybe_add_noise=False)
        print("DNN")
        print(circuit)
        print(fidelity)
        rewardArray.append(reward)
        fidelityArray.append(fidelity)

    # print("DNN play reward= ", rewardArray)
    # print("mean reward= ", np.mean(rewardArray))
    print("DNN play fidelity= ", fidelityArray)
    print("mean fidelity= ", np.mean(fidelityArray))
    print("max fidelity = " , np.max(fidelityArray))
    # return np.mean(corrArray), np.max(corrArray)
    return np.mean(fidelityArray), np.max(fidelityArray)

######## AlphaSeq player (Noiseless games)
def evaluate_DNN(n_games, tau, evaluating_fn):
    # play 50 games, calculate the mean corr
    # 创建QAS环境
    # Parameters
    env_name = 'BasicTwoQubit-v0'
    fidelity_threshold = 0.97
    reward_penalty = 0.01
    max_timesteps = 10
    # Environment
    env = gym.make(env_name,
                   fidelity_threshold=fidelity_threshold,
                   reward_penalty=reward_penalty,
                   max_timesteps=max_timesteps)
    rewardArray = []
    fidelityArray = []
    for _ in range(n_games):
        observation = env.reset()
        currentMove, _ = MCTS_main(args, VisitedState, stepSize, n_steps, DNN.evaluate_node, calc_reward, selfPlay = 0)
        circuit_gates=[]
        for mov in currentMove:
            circuit_gates.append(env.action_gates[int(mov)])
        # record every reward
        qubits = cirq.LineQubit.range(2)
        error_gates = 0.001
        error_observables = 0.001
        circuit, reward, fidelity = calc_reward(circuit_gates, qubits, error_gates, error_observables,
                                                maybe_add_noise=False)
        rewardArray.append(reward)
        fidelityArray.append(fidelity)

        # print("seq found = ", currentMove)
        # print("reward = ", reward)
        # print("corr = ", corr)

    # print("MCTS + DNN play reward = ", rewardArray)
    # print("mean reward= ", np.mean(rewardArray))
    print("MCTS + DNN play fidelity = ", fidelityArray)
    print("mean fidelity = ", np.mean(fidelityArray))
    print("max fidelity = " , np.max(fidelityArray))

    return np.mean(fidelityArray), np.max(fidelityArray)

######## Update DNN
def updateDNN(memoryBuffer, lr):
	######## without replacement
    # np.random.shuffle(memoryBuffer)
    # numBatch = int(len(memoryBuffer)/args.batchSize)
    # for ii in range(numBatch):
    #     mini_batch = []
    #     for jj in range(args.batchSize):
    #         mini_batch.append(memoryBuffer[ii*args.batchSize+jj])
    #     DNN.update_DNN(mini_batch, lr)
	######## with replacement
    # train DNN
    np.random.shuffle(memoryBuffer)
    numBatch = int(len(memoryBuffer)/args.batchSize) * 6
    for ii in range(numBatch):
        mini_batch = random.sample(memoryBuffer, args.batchSize)
        DNN.update_DNN(mini_batch, lr)


def main():
    # initialize memory buffer
    memoryBuffer = deque(maxlen = memorySize)

    # load/save latest structure
    # DNN.loadParams('./240812_bestParams/bestParams1000/net_params.ckpt')
    DNN.saveParams('./240822_bestParams/net_params.ckpt')

    # performance of current DNN
    DNNplayer, DNNmax = DNN_play(n_games = 20, evaluating_fn = DNN.evaluate_node)
    meanCorr , MCTSmax= evaluate_DNN(n_games = args.eval_games, tau = 0, evaluating_fn = DNN.evaluate_node)
    
    if args.recordState == 1:
        f = open('./240822_bestParams/Record.txt', 'w')
        f.write(str(0)+" "+str(DNNplayer)+" "+str(meanCorr)+" "+str(DNNmax)+" "+str(MCTSmax) + " " + str(0) + " ")
        f.write(str(0)+" ") # overall visited states
        f.write(str(0)+" ") # visited states in the last G episodes
        f.write(str(0)+" ") # mean entropy in the last G episodes
        f.write(str(0)+" ") # cross entropy in the last G episodes
        f.write(str(0)+";\n") # number of states being evaluated in the latest G episodes
        f.close()

    global worstMetric
    worstMetric = meanCorr

    print("-------------------------------------")
    print("worstMetric ="+str(worstMetric))
    print("bestMetric = "+str(bestMetric))

    overall_startTime = time.time()

    episode = 0
    while episode < args.numEpisode:
        print("----------------------------------------------  Episode %s:"%(episode))
        epi_time_start = time.time()
        env_name = 'BasicTwoQubit-v0'
        fidelity_threshold = 0.97
        reward_penalty = 0.01
        max_timesteps = 10
        # Environment
        env = gym.make(env_name,
                       fidelity_threshold=fidelity_threshold,
                       reward_penalty=reward_penalty,
                       max_timesteps=max_timesteps)

        # ---------------------- Part I: game-play with MCTS to gain experiences ----------------------
        cummulativeMove, temp_store = MCTS_main(args, VisitedState, stepSize, n_steps, DNN.evaluate_node, calc_reward, selfPlay = 1)


        circuit_gates = []
        # in each episode, we find a sequence - calculate reward
        for mov in cummulativeMove:
            circuit_gates.append(env.action_gates[int(mov)])
        qubits = cirq.LineQubit.range(2)
        error_gates = 0.001
        error_observables = 0.001
        circuit, reward, fidelity = calc_reward(circuit_gates, qubits, error_gates, error_observables,
                                                maybe_add_noise=False)
        # Store n_steps experience
        # print("actions found = ", circuit_gates)
        print("circuit =")
        print(circuit)
        # print("reward = ", reward)
        print("fidelity = ", fidelity)


        for state in temp_store:
            state.append(reward)

        memoryBuffer.extend(temp_store)

        if args.recordState == 1:
            print("...... The overall visited state so far =", VisitedState.printCnt())

        # --------------------------- Part II: DNN update ----------------------------
        lr = 0.0001

        # train NN
        if episode > 0 and episode % args.updateNNcycle == 0:
            # train new params
            updateDNN(memoryBuffer, lr)
            print("Deep Neural Network Updated, now evaluate the new DNN ...")
            print("learning rate = ",lr)

            # measure the performance of updated DNN
            print("---------------------------------------------------")

            DNNplayer, DNNmax = DNN_play(n_games=20, evaluating_fn=DNN.evaluate_node)
            print("DNN玩家20次游戏平均测评函数 = ", DNNplayer)
            print("DNN玩家20次游戏最大测评函数 = ", DNNmax)
            updatedCorr, MCTSmax = evaluate_DNN(n_games=args.eval_games, tau=0, evaluating_fn=DNN.evaluate_node)
            print("DNN-MCTS50次游戏玩家平均测评函数 = ", updatedCorr)
            print("DNN-MCTS50次游戏玩家最大测评函数 = ", MCTSmax)

            # ================================================================ store
            if args.recordState == 1:
                f = open('./240822_bestParams/Record.txt', 'a')
                f.write(str(episode)+" "+str(DNNplayer)+" "+str(updatedCorr)+" "+str(DNNmax)+" "+str(MCTSmax)+ " " + str(0)+ " ")
                f.write(str(VisitedState.printCnt())+" ") # overall visited states
                f.write(str(VisitedState.printCnt1())+" ") # visited states in the last G episodes
                VisitedState.renew()
                entropy, crossentropy, numStates = DNN.output_entropy()
                f.write(str(entropy)+" ") # mean entropy in the last G episodes
                f.write(str(crossentropy)+" ") # cross entropy in the last G episodes
                f.write(str(numStates)+";\n") # number of states being evaluated in the latest G episodes
                DNN.refresh_entropy()
                f.close()

            if args.recordState == 1:
                filename = "./240822_bestParams/bestParams" + str(episode) + "/net_params.ckpt"
                DNN.saveParams(filename)
                filename1 = "./240822_bestParams/States" + str(episode) + ".txt"
                f = open(filename1, 'w')
                f.write(str(VisitedState.visitedState))
                f.close()

        episode += 1
        print(time.time()-epi_time_start)

    # store experience
    # pickle.dump(DNN.memoryBuffer, open('./240731_bestParams/latestMemory', 'wb'))
    # loadedMemory = pickle.load(open('./240731_bestParams/latestMemory', 'rb'))

    print("-------------------------------------")
    print("-------------------------------------")
    print("After evaluation, the mean reward we get is %s"%(meanCorr))

    # seconds consumed from beginning
    print(time.time()-overall_startTime)

    DNN.plot_cost()
    pdb.set_trace()


if __name__ == "__main__":
    DNN = DeepNN(args, stepSize)
    VisitedState = HarshState(overallSteps)
    ##### load state counts
    # f = open('States3700.txt','r')
    # aa = f.read()
    # bb = eval(aa)
    # f.close()
    # VisitedState.visitedState = bb
    main()  
