from game import calculate_minimax
import numpy as np
import random
class FP_Agent:
    def __init__(self, payoff_matrix, actions, is_row_player=True):
        self.payoff_matrix = payoff_matrix
        self.actions = actions
        self.is_row_player = is_row_player
        # 历史联合动作记录（用于统计对手动作频率）
        self.joint_history = []  # 存储(自身动作, 对手动作)
    
    def choose_action(self):
        """
                  改进版FP动作选择：向前看1步，计算期望收益
        """
        if len(self.joint_history) == 0:
            # 第一轮无历史，随机选动作
            return random.choice(self.actions)
        
        # 第一步：统计对手动作频率（基于历史联合动作）
        opp_actions = [h[1] for h in self.joint_history]
        opp_act_count = {act: opp_actions.count(act) for act in self.actions}
        total = len(opp_actions)
        opp_prob = {act: count / total for act, count in opp_act_count.items()}
        
        # 第二步：计算每个自身动作的“向前看1步期望收益”（简化：当前步期望收益）
        act_expected_pay = {}
        for self_act in self.actions:
            expected_pay = 0.0
            for opp_act, prob in opp_prob.items():
                # 自身选self_act、对手选opp_act的收益
                if self.is_row_player:
                    pay = self.payoff_matrix[(self_act, opp_act)][0]
                else:
                    pay = self.payoff_matrix[(opp_act, self_act)][1]
                expected_pay += prob * pay
            act_expected_pay[self_act] = expected_pay
        
        # 第三步：选期望收益最大的动作
        max_pay = max(act_expected_pay.values())
        best_acts = [act for act, pay in act_expected_pay.items() if pay == max_pay]
        return random.choice(best_acts)
    
    def update(self, self_act, opp_act):
        """
                  更新历史联合动作
        """
        self.joint_history.append((self_act, opp_act))
