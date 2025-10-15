from game import calculate_minimax
import numpy as np
import random

class SPaM_Agent:
    def __init__(self, payoff_matrix, actions, is_row_player=True):
        self.payoff_matrix = payoff_matrix  # 收益矩阵
        self.actions = actions  # 所有动作（如['C','D']）
        self.is_row_player = is_row_player  # 是否为行玩家
        self.eta = 0.1  # 论文参数：探索基础概率
        self.rho = 0.8  # 论文参数：纯收益优先的探索占比
        self.epsilon = 1e-3  # 愧疚值最小量（避免为0，在 case4 使用）
        
        # 1. 初始化目标解c（最大化双方正优势乘积的联合动作）
        self.target_solution = self._calculate_target_solution()
        # 2. 初始化参数
        self.G_self = 0.0  # 自身愧疚值
        self.G_opponent = 0.0  # 对手愧疚值
        self.T = {a: 0.0 for a in self.actions}  # 教导者效用（每个动作的T值）
        self.F = {a: 0.0 for a in self.actions}  # 追随者效用（每个动作的F值）
        # 3. 历史记录（用于更新F和期望收益）
        # 存储 dict: {'self_act','opp_act','self_pay','opp_pay','G_self','G_opp'}
        self.history = []
    
    def _calculate_target_solution(self):
        """
        计算目标解c：最大化双方正优势乘积（优势=收益-极小极大值）
        :return: 目标联合动作（如('C','C')）
        """
        # 第一步：计算双方的极小极大值
        m_self = calculate_minimax(self.payoff_matrix, self.is_row_player)
        m_opponent = calculate_minimax(self.payoff_matrix, not self.is_row_player)
        
        # 第二步：遍历所有联合动作，计算“正优势乘积”
        max_product = -float('inf')
        best_joint_act = None
        # 为健壮性，明确定义行/列动作集合
        row_actions = sorted(list(set([k[0] for k in self.payoff_matrix.keys()])))
        col_actions = sorted(list(set([k[1] for k in self.payoff_matrix.keys()])))
        for row_act in row_actions:
            for col_act in col_actions:
                joint_act = (row_act, col_act)
                # 自身收益和对手收益（根据is_row_player决定顺序）
                if self.is_row_player:
                    pay_self = self.payoff_matrix[joint_act][0]
                    pay_opp = self.payoff_matrix[joint_act][1]
                else:
                    pay_self = self.payoff_matrix[joint_act][1]
                    pay_opp = self.payoff_matrix[joint_act][0]
                # 优势（仅保留正数，否则乘积为0）
                adv_self = max(pay_self - m_self, 0)
                adv_opp = max(pay_opp - m_opponent, 0)
                product = adv_self * adv_opp
                # 找乘积最大的联合动作
                if product > max_product:
                    max_product = product
                    best_joint_act = joint_act
        # 若所有乘积都 <= -inf（理论上不会），退回任意一对
        if best_joint_act is None:
            # fallback: 第一个键
            best_joint_act = next(iter(self.payoff_matrix.keys()))
        return best_joint_act  # 目标解（如囚徒困境中的('C','C')）
    
    def _update_guilt(self, self_act, opp_act, self_pay, opp_pay):
        """
        按论文Figure 1更新自身和对手的愧疚值G（实现六种情况的分支）
        注意：本函数使用的是本轮的动作和收益，并更新 self.G_self 和 self.G_opponent。
        """
        # 目标动作（自身的目标动作c_self，对手的目标动作c_opp）
        c_self = self.target_solution[0] if self.is_row_player else self.target_solution[1]
        c_opp = self.target_solution[1] if self.is_row_player else self.target_solution[0]
        # 目标解的平均收益 r(c)（此处简化为目标动作的收益）
        r_self_c = self.payoff_matrix[self.target_solution][0] if self.is_row_player else self.payoff_matrix[self.target_solution][1]
        r_opp_c = self.payoff_matrix[self.target_solution][1] if self.is_row_player else self.payoff_matrix[self.target_solution][0]
        
        # -------------------------- 更新自身愧疚值G_self --------------------------
        # 根据论文六种情形分支实现（保持语意尽量贴近论文）
        if self.G_self > 0:  # 自身当前guilty (case1/2/4)
            if self_act == c_self:
                if opp_act == c_opp:
                    # case1：guilty + 自身执行c + 对手执行c -> 愧疚清零
                    self.G_self = 0.0
                else:
                    # case2：guilty + 自身执行c + 对手不执行c
                    # 更新 G = G + self_pay - r(c)，若结果 <= 0 则赦免为 0（允许赦免）
                    self.G_self = max(self.G_self + self_pay - r_self_c, 0.0)
            else:
                # case4：guilty + 自身偏离 -> 保持至少 epsilon
                self.G_self = max(self.epsilon, self.G_self + self_pay - r_self_c)
        else:  # 自身当前无罪（G_self <= 0）
            if self_act == c_self:
                # case3: 无罪 + 自身执行c -> 保持无罪
                self.G_self = 0.0
            else:
                if self.G_opponent > 0:
                    # case5：无罪 + 自身偏离 + 对手guilty -> 自身仍无罪
                    self.G_self = 0.0
                else:
                    # case6：无罪 + 自身偏离 + 对手无罪 -> G = max(self_pay - r(c) + ε, ε)
                    self.G_self = max(self_pay - r_self_c + self.epsilon, self.epsilon)
        
        # -------------------------- 更新对手愧疚值G_opponent --------------------------
        if self.G_opponent > 0:  # 对手当前 guilty (case1/2/4 mirrored)
            if opp_act == c_opp:
                if self_act == c_self:
                    # case1 for opponent: 对手guilty + 对手执行c + 自身执行c -> 对手愧疚清零
                    self.G_opponent = 0.0
                else:
                    # case2 mirrored：对手guilty + 对手执行c + 自身不执行c
                    self.G_opponent = max(self.G_opponent + opp_pay - r_opp_c, 0.0)
            else:
                # case4 mirrored：对手guilty + 对手偏离
                self.G_opponent = max(self.epsilon, self.G_opponent + opp_pay - r_opp_c)
        else:
            if opp_act == c_opp:
                # case3 mirrored：对手无罪 + 对手执行c -> 保持无罪
                self.G_opponent = 0.0
            else:
                if self.G_self > 0:
                    # case5 mirrored：对手无罪 + 对手偏离 + 自身guilty -> 对手仍无罪
                    self.G_opponent = 0.0
                else:
                    # case6 mirrored：对手无罪 + 偏离 + 自身无罪 -> 对手获得 guilt
                    self.G_opponent = max(opp_pay - r_opp_c + self.epsilon, self.epsilon)
    
    def _update_utilities(self):
        """
        更新教导者效用T和追随者效用F
        F: 用历史中某动作的平均自身收益来估计（跟随者效用）
        T: 按论文公式1/2构造（当对手无罪时用 ±1；当对手有罪时使用 r_opp(c) - E[U_opp|G_opp>0] - E_p）
        """
        # 1. 更新追随者效用F（用历史：F(a) = 选a的平均自身收益）
        if len(self.history) == 0:
            # 如果无历史，保持初始 F=0（或可保持现有值）
            # 直接返回，避免除以0
            pass
        else:
            for act in self.actions:
                act_history = [h for h in self.history if h['self_act'] == act]
                if len(act_history) == 0:
                    self.F[act] = 0.0
                else:
                    avg_pay = sum([h['self_pay'] for h in act_history]) / len(act_history)
                    self.F[act] = avg_pay
        
        # 2. 更新教导者效用T（公式1/2）
        c_self = self.target_solution[0] if self.is_row_player else self.target_solution[1]
        c_opp = self.target_solution[1] if self.is_row_player else self.target_solution[0]
        # 对手的极小极大值 m_opp
        m_opp = calculate_minimax(self.payoff_matrix, not self.is_row_player)
        # 目标动作的对手收益 r_opp(c)
        r_opp_c = self.payoff_matrix[self.target_solution][1] if self.is_row_player else self.payoff_matrix[self.target_solution][0]
        
        if self.G_opponent <= 0:
            # 公式1：对手无罪 -> T=1 当且仅当动作为目标动作，否则 T=-1
            for act in self.actions:
                self.T[act] = 1.0 if act == c_self else -1.0
        else:
            # 公式2：对手 guilty -> T = r_opp(c) - E[U_opp(s, act | G_opp > 0)] - E_p
            # 计算 E_p = min(G_opponent, r_opp(c) - m_opp)
            E_p = min(self.G_opponent, max(r_opp_c - m_opp, 0.0))
            for act in self.actions:
                # 筛选历史：当时自身选 act 且当时对手是 guilty（历史条目里的 G_opp 字段）
                guilty_act_history = [h for h in self.history if h['self_act'] == act and h['G_opp'] > 0]
                if len(guilty_act_history) == 0:
                    E_U_opp = 0.0
                else:
                    # 注意：E[U_opp] 是对手的收益（opp_pay）
                    E_U_opp = sum([h['opp_pay'] for h in guilty_act_history]) / len(guilty_act_history)
                # T 的计算（可能为负）
                self.T[act] = r_opp_c - E_U_opp - E_p
    
    def choose_action(self):
        """
        按论文Table 1选择动作并返回意图动作
        """
        # 第一步：确定可选动作集S
        S = []
        max_T = max(self.T.values()) if len(self.T) > 0 else 0.0
        for act in self.actions:
            if self.T[act] >= 0 or self.T[act] == max_T:
                S.append(act)
        if len(S) == 0:
            S = [max(self.T.items(), key=lambda x: x[1])[0]]
        
        # 第二步：按概率选择动作
        rand = random.random()
        if rand < (1 - self.eta):
            # 1 - eta 概率：在 S 中选使 F 最大的动作
            max_F_in_S = max([self.F[act] for act in S])
            best_acts = [act for act in S if self.F[act] == max_F_in_S]
            return random.choice(best_acts)
        elif rand < (1 - self.eta) + self.rho * self.eta:
            # rho*eta 概率：选全局 F 最大的动作（无论是否在 S 中）
            max_F_global = max(self.F.values()) if len(self.F) > 0 else 0.0
            best_acts = [act for act in self.actions if self.F[act] == max_F_global]
            return random.choice(best_acts)
        else:
            # 其余小概率：随机探索
            return random.choice(self.actions)
    
    def update(self, self_act, opp_act, self_pay, opp_pay):
        """
        每轮迭代后更新历史、愧疚值、效用函数
        必须按照论文 Table2 的顺序：
         1) 观察 -> 2) 更新 guilt -> 3) 记录带 guilt 标记的历史 -> 4) 更新 T & F
        """
        # 1. 先更新愧疚值（基于本轮动作与收益）
        self._update_guilt(self_act, opp_act, self_pay, opp_pay)
        
        # 2. 记录历史 —— 包含“对手当时是否 guilty”（使用更新后的值）
        entry = {
            'self_act': self_act,
            'opp_act': opp_act,
            'self_pay': self_pay,
            'opp_pay': opp_pay,
            'G_self': self.G_self,
            'G_opp': self.G_opponent
        }
        self.history.append(entry)
        
        # 3. 更新效用函数（使用带有 guilt 标记的 history）
        self._update_utilities()

