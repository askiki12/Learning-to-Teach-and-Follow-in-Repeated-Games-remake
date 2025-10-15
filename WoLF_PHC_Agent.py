from game import calculate_minimax
import numpy as np
import random
class WoLF_PHC_Agent:
    def __init__(self, payoff_matrix, actions, is_row_player=True):
        self.payoff_matrix = payoff_matrix
        self.actions = actions
        self.is_row_player = is_row_player
        self.num_actions = len(actions)
        # 策略：每个动作的选择概率
        self.policy = {act: 1.0 / self.num_actions for act in self.actions}
        # 平均策略（用于判断“赢/输”）
        self.avg_policy = {act: 1.0 / self.num_actions for act in self.actions}
        # 学习率（赢时小，输时大）
        self.alpha_win = 0.01  # 赢时学习率
        self.alpha_lose = 0.05  # 输时学习率
        # 价值函数V（每个动作的价值）
        self.V = {act: 0.0 for act in self.actions}
        # 迭代次数（用于更新平均策略）
        self.t = 0
    
    def choose_action(self):
        """
                  按策略概率选择动作
        """
        acts = list(self.policy.keys())
        probs = list(self.policy.values())
        return np.random.choice(acts, p=probs)
    
    def _get_reward(self, self_act, opp_act):
        """
                  获取自身收益
        """
        if self.is_row_player:
            return self.payoff_matrix[(self_act, opp_act)][0]
        else:
            return self.payoff_matrix[(opp_act, self_act)][1]
    
    def update(self, self_act, opp_act):
        """
                  更新策略、平均策略、价值函数（按WoLF-PHC规则）
        """
        self.t += 1
        reward = self._get_reward(self_act, opp_act)
        
        # 1. 更新价值函数V（TD(0)更新）
        gamma = 0.95  # 折扣因子（论文常用值）
        current_V = self.V[self_act]
        # 下一轮的期望价值（基于当前策略）
        next_expected_V = sum([self.policy[act] * self.V[act] for act in self.actions])
        self.V[self_act] = current_V + 0.1 * (reward + gamma * next_expected_V - current_V)  # TD学习率0.1
        
        # 2. 判断“赢/输”：当前策略的价值 > 平均策略的价值 → 赢
        current_policy_V = sum([self.policy[act] * self.V[act] for act in self.actions])
        avg_policy_V = sum([self.avg_policy[act] * self.V[act] for act in self.actions])
        is_win = current_policy_V > avg_policy_V
        alpha = self.alpha_win if is_win else self.alpha_lose
        
        # 3. 更新策略（梯度上升，最大化价值）
        # 找到当前价值最大的动作（最优动作）
        max_V = max(self.V.values())
        best_acts = [act for act, v in self.V.items() if v == max_V]
        # 策略更新：最优动作概率增加，其他减少
        for act in self.actions:
            if act in best_acts:
                self.policy[act] += alpha * (1 - self.policy[act])
            else:
                self.policy[act] -= alpha * self.policy[act]
        # 归一化策略概率（避免超界）
        total = sum(self.policy.values())
        self.policy = {act: p / total for act, p in self.policy.items()}
        
        # 4. 更新平均策略
        for act in self.actions:
            self.avg_policy[act] = (self.t - 1) / self.t * self.avg_policy[act] + 1 / self.t * self.policy[act]
