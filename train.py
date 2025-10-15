from game import get_actual_action
from SPaM_Agent import SPaM_Agent
from FP_Agent import FP_Agent
from WoLF_PHC_Agent import WoLF_PHC_Agent
import numpy as np
import random
def single_experiment(agent1, agent2, payoff_matrix, actions, total_steps=5000, noise=0.05):
    """
         单轮实验：两个智能体博弈total_steps轮
    :param noise: 扰动概率（默认0.05 -> 95%执行意图动作）
    """
    agent1_total_pay = 0.0
    agent2_total_pay = 0.0
    agent1_avg_pays = []
    agent2_avg_pays = []
    
    for step in range(1, total_steps + 1):
        agent1_intend_act = agent1.choose_action()
        agent2_intend_act = agent2.choose_action()
        
        agent1_act = get_actual_action(agent1_intend_act, actions, noise=noise)
        agent2_act = get_actual_action(agent2_intend_act, actions, noise=noise)
        
        joint_act = (agent1_act, agent2_act)
        agent1_pay = payoff_matrix[joint_act][0]
        agent2_pay = payoff_matrix[joint_act][1]
        
        agent1_total_pay += agent1_pay
        agent2_total_pay += agent2_pay
        agent1_avg = agent1_total_pay / step
        agent2_avg = agent2_total_pay / step
        agent1_avg_pays.append(agent1_avg)
        agent2_avg_pays.append(agent2_avg)
        
        # 更新智能体
        if isinstance(agent1, SPaM_Agent):
            agent1.update(agent1_act, agent2_act, agent1_pay, agent2_pay)
        elif isinstance(agent1, FP_Agent):
            agent1.update(agent1_act, agent2_act)
        elif isinstance(agent1, WoLF_PHC_Agent):
            agent1.update(agent1_act, agent2_act)
        
        if isinstance(agent2, SPaM_Agent):
            agent2.update(agent2_act, agent1_act, agent2_pay, agent1_pay)
        elif isinstance(agent2, FP_Agent):
            agent2.update(agent2_act, agent1_act)
        elif isinstance(agent2, WoLF_PHC_Agent):
            agent2.update(agent2_act, agent1_act)
    
    return agent1_avg_pays, agent2_avg_pays


def repeat_experiments(agent1_class, agent2_class, payoff_matrix, actions, 
                       num_repeats=50, total_steps=5000, is_row_player1=True, noise=0.05, seed=None):
    """
         重复num_repeats次实验，返回平均收益曲线的均值和标准差
    :param noise: 扰动概率
    :param seed: 随机种子（若不为 None，将设置随机数生成器以保证可复现）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    agent1_all_pays = []
    agent2_all_pays = []
    
    for _ in range(num_repeats):
        agent1 = agent1_class(payoff_matrix, actions, is_row_player=is_row_player1)
        agent2 = agent2_class(payoff_matrix, actions, is_row_player=False)
        
        agent1_pays, agent2_pays = single_experiment(agent1, agent2, payoff_matrix, actions, total_steps, noise)
        
        agent1_all_pays.append(agent1_pays)
        agent2_all_pays.append(agent2_pays)
    
    agent1_mean = np.mean(agent1_all_pays, axis=0)
    agent1_std = np.std(agent1_all_pays, axis=0)
    agent2_mean = np.mean(agent2_all_pays, axis=0)
    agent2_std = np.std(agent2_all_pays, axis=0)
    
    return (agent1_mean, agent1_std), (agent2_mean, agent2_std)
