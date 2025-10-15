# game.py
import numpy as np
import random
# 1. 囚徒困境（Prisoner's Dilemma）：动作C（合作）、D（背叛）
pd_payoff = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1)
}

# 2. 小鸡游戏（Chicken）：动作C（合作）、D（背叛）
chicken_payoff = {
    ('C', 'C'): (4, 4),
    ('C', 'D'): (2, 5),
    ('D', 'C'): (5, 2),
    ('D', 'D'): (0, 0)
}

# 3. Tricky游戏：动作a、b（行玩家选a/b，列玩家选a/b）
tricky_payoff = {
    ('a', 'a'): (0, 3),
    ('a', 'b'): (3, 2),  # 目标解：行a、列b
    ('b', 'a'): (1, 0),
    ('b', 'b'): (2, 1)
}

# 统一动作列表（每个游戏的动作集合）
game_actions = {
    'pd': ['C', 'D'],
    'chicken': ['C', 'D'],
    'tricky': ['a', 'b']
}

def calculate_minimax(payoff_matrix, is_row_player=True):
    """
    计算玩家的极小极大值
    :param payoff_matrix: 收益矩阵（字典，键为(行动作,列动作)，值为(行收益,列收益)）
    :param is_row_player: True=行玩家，False=列玩家
    :return: 极小极大值m
    """
    # 更稳健的动作集合采集：分别收集行和列动作
    row_actions = sorted(list(set([k[0] for k in payoff_matrix.keys()])))
    col_actions = sorted(list(set([k[1] for k in payoff_matrix.keys()])))
    if is_row_player:
        # 行玩家：选动作，最大化“列玩家选最差动作时的收益”
        min_payoffs = []
        for row_act in row_actions:
            # 行动作固定时，列玩家选让行收益最小的动作
            row_pays = [payoff_matrix[(row_act, col_act)][0] for col_act in col_actions]
            min_payoffs.append(min(row_pays))
        return max(min_payoffs)  # 行玩家选“最小收益最大”的动作
    else:
        # 列玩家：选动作，最小化“行玩家选最差动作时的收益”
        max_payoffs = []
        for col_act in col_actions:
            # 列动作固定时，行玩家选让列收益最大的动作
            col_pays = [payoff_matrix[(row_act, col_act)][1] for row_act in row_actions]
            max_payoffs.append(max(col_pays))
        return min(max_payoffs)  # 列玩家选“最大收益最小”的动作
        
        
def get_actual_action(intended_action, actions, noise=0.05):
    """
    按 (1-noise) 概率执行意图动作，noise 概率执行随机的其他动作
    支持动作数量 > 2 的情形（扰动时从其他动作随机选择）
    :param intended_action: 意图动作
    :param actions: 所有可能动作（如['C','D']）
    :param noise: 扰动概率，默认 0.05
    :return: 实际执行的动作
    """
    if random.random() < (1 - noise):
        return intended_action
    else:
        others = [a for a in actions if a != intended_action]
        if len(others) == 0:
            return intended_action
        return random.choice(others)

