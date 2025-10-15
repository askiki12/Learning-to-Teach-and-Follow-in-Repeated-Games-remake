import numpy as np
import random
from game import pd_payoff, chicken_payoff, tricky_payoff, game_actions, get_actual_action
from train import single_experiment, repeat_experiments
from SPaM_Agent import SPaM_Agent
from FP_Agent import FP_Agent
from WoLF_PHC_Agent import WoLF_PHC_Agent
import matplotlib.pyplot as plt
import os

# 创建保存图片的目录
os.makedirs('plots', exist_ok=True)

# 固定随机种子以便复现
SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

# 实验参数
total_steps = 500
num_repeats = 10
noise = 0.05

# 定义三种Learner算法类（统一对比SPaM/FP/WoLF-PHC）
LEARNER_CLASSES = {
    'SPaM': SPaM_Agent,
    'FP': FP_Agent,
    'WoLF-PHC': WoLF_PHC_Agent
}

# 定义Learner样式（固定，确保所有图风格统一）
LEARNER_STYLES = {
    'SPaM': {'color': 'blue', 'linestyle': '-', 'linewidth': 2, 'label': 'SPaM (Learner)'},
    'FP': {'color': 'green', 'linestyle': '--', 'linewidth': 2, 'label': 'FP (Learner)'},
    'WoLF-PHC': {'color': 'red', 'linestyle': ':', 'linewidth': 2, 'label': 'WoLF-PHC (Learner)'}
}


def get_learner_data(learner_class, opponent_class, payoff_matrix, actions, is_learner_row=True, seed=SEED):
    """
    工具函数：获取单一Learner对抗指定对手的数据（mean, std）
    :param is_learner_row: True=Learner是行玩家，False=Learner是列玩家
    :return: (mean, std)
    """
    if is_learner_row:
        # Learner是行玩家（agent1），对手是列玩家（agent2）
        exp_result = repeat_experiments(
            agent1_class=learner_class, agent2_class=opponent_class,
            payoff_matrix=payoff_matrix, actions=actions,
            num_repeats=num_repeats, total_steps=total_steps,
            noise=noise, seed=seed
        )
        return exp_result[0]  # agent1（Learner行）的(mean, std)
    else:
        # Learner是列玩家（agent2），对手是行玩家（agent1）
        exp_result = repeat_experiments(
            agent1_class=opponent_class, agent2_class=learner_class,
            payoff_matrix=payoff_matrix, actions=actions,
            num_repeats=num_repeats, total_steps=total_steps,
            noise=noise, seed=seed
        )
        return exp_result[1]  # agent2（Learner列）的(mean, std)


def plot_three_learners(results_dict, title, save_path=None):
    """
    绘制单张图（3条线：三种Learner对抗同一对手）
    :param results_dict: 键=Learner名称，值=(mean, std)
    """
    plt.figure(figsize=(10, 6))
    # 获取迭代步数（所有曲线长度一致）
    first_mean = next(iter(results_dict.values()))[0]
    steps = range(1, len(first_mean) + 1)

    # 绘制每条Learner的曲线+标准差阴影
    for learner_name, (mean, std) in results_dict.items():
        style = LEARNER_STYLES[learner_name]
        plt.plot(steps, mean,
                 color=style['color'],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 label=style['label'])
        # 绘制标准差阴影（透明度0.2，与曲线同色）
        plt.fill_between(steps, mean - std, mean + std,
                         color=style['color'], alpha=0.2)

    # 图表标签（匹配论文格式）
    plt.xlabel('Number of iterations', fontsize=12)
    plt.ylabel('Average payoff (Learner)', fontsize=12)  # 明确纵轴是Learner的收益
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 避免标签被截断
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_scene_experiments(scene_name, payoff_matrix, actions, need_row_col_split=False):
    """
    运行单个场景的实验（生成3张图，若需分行列则生成6张）
    :param need_row_col_split: True=需分行/列Learner（如Tricky游戏），False=角色对称（如囚徒困境）
    """
    print(f"=== 开始 {scene_name} 实验 ===")
    # 定义需要对抗的3种对手（SPaM/FP/WoLF-PHC）
    OPPONENTS = {
        'SPaM_Opp': SPaM_Agent,
        'FP_Opp': FP_Agent,
        'WoLF-PHC_Opp': WoLF_PHC_Agent
    }
    # 实验种子偏移（避免不同实验重复随机序列）
    seed_offset = 0

    # 1. 角色无需拆分的情况（囚徒困境/小鸡游戏）
    if not need_row_col_split:
        for opp_name, opp_class in OPPONENTS.items():
            # 收集三种Learner对抗当前对手的数据（Learner均为行玩家，角色对称）
            learner_data = {}
            for learner_name, learner_class in LEARNER_CLASSES.items():
                learner_data[learner_name] = get_learner_data(
                    learner_class=learner_class,
                    opponent_class=opp_class,
                    payoff_matrix=payoff_matrix,
                    actions=actions,
                    is_learner_row=True,
                    seed=SEED + seed_offset
                )
                seed_offset += 1  # 每次实验种子+1，保证独立性

            # 绘制单张图（3条线：三种Learner对抗当前对手）
            plot_title = f"{scene_name}: 3 Learners vs {opp_name.replace('_Opp', '')}"
            save_path = f"plots/{scene_name.lower().replace(' ', '_')}_learners_vs_{opp_name.lower().replace('_opp', '')}.png"
            plot_three_learners(learner_data, plot_title, save_path)
            print(f"→ 已生成：{save_path}")

    # 2. 角色需拆分的情况（Tricky游戏：Learner行玩家 + Learner列玩家）
    else:
        for opp_name, opp_class in OPPONENTS.items():
            # 2.1 Learner作为行玩家，对抗当前对手（列玩家）
            row_learner_data = {}
            for learner_name, learner_class in LEARNER_CLASSES.items():
                row_learner_data[learner_name] = get_learner_data(
                    learner_class=learner_class,
                    opponent_class=opp_class,
                    payoff_matrix=payoff_matrix,
                    actions=actions,
                    is_learner_row=True,
                    seed=SEED + seed_offset
                )
                seed_offset += 1
            # 绘制Learner行玩家的图
            row_title = f"{scene_name}: 3 Learners (Row) vs {opp_name.replace('_Opp', '')}"
            row_save_path = f"plots/{scene_name.lower().replace(' ', '_')}_row_learners_vs_{opp_name.lower().replace('_opp', '')}.png"
            plot_three_learners(row_learner_data, row_title, row_save_path)
            print(f"→ 已生成：{row_save_path}")

            # 2.2 Learner作为列玩家，对抗当前对手（行玩家）
            col_learner_data = {}
            for learner_name, learner_class in LEARNER_CLASSES.items():
                col_learner_data[learner_name] = get_learner_data(
                    learner_class=learner_class,
                    opponent_class=opp_class,
                    payoff_matrix=payoff_matrix,
                    actions=actions,
                    is_learner_row=False,
                    seed=SEED + seed_offset
                )
                seed_offset += 1
            # 绘制Learner列玩家的图
            col_title = f"{scene_name}: 3 Learners (Col) vs {opp_name.replace('_Opp', '')}"
            col_save_path = f"plots/{scene_name.lower().replace(' ', '_')}_col_learners_vs_{opp_name.lower().replace('_opp', '')}.png"
            plot_three_learners(col_learner_data, col_title, col_save_path)
            print(f"→ 已生成：{col_save_path}")

    print(f"=== {scene_name} 实验完成 ===\n")


# -------------------------- 主函数：运行所有场景实验 --------------------------
if __name__ == "__main__":
    # 1. 囚徒困境（角色对称，生成3张图）
    run_scene_experiments(
        scene_name="Prisoner's Dilemma",
        payoff_matrix=pd_payoff,
        actions=game_actions['pd'],
        need_row_col_split=False
    )

    # 2. 小鸡游戏（角色对称，生成3张图）
    run_scene_experiments(
        scene_name="Chicken",
        payoff_matrix=chicken_payoff,
        actions=game_actions['chicken'],
        need_row_col_split=False
    )

    # 3. Tricky游戏（角色不对称，生成6张图：行Learner3张 + 列Learner3张）
    run_scene_experiments(
        scene_name="Tricky Game",
        payoff_matrix=tricky_payoff,
        actions=game_actions['tricky'],
        need_row_col_split=True
    )

    print("所有实验全部完成！")
