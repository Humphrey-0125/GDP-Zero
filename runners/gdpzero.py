import numpy as np
import logging
import pickle
import argparse
import numpy as np  # 重复导入，但无伤大雅

from tqdm.auto import tqdm
from core.gen_models import (
    LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner
)
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
    # ===== 1) 载入博弈本体（ontology）：系统/用户的对话行为集合、角色名等 =====
    game_ontology = PersuasionGame.get_game_ontology()
    sys_da = game_ontology['system']['dialog_acts']   # 系统端（劝说者）的可用对话行为集合
    user_da = game_ontology['user']['dialog_acts']    # 用户端（被劝说者）的可用对话行为集合
    system_name = PersuasionGame.SYS                  # "系统"的标识字符串
    user_name = PersuasionGame.USR                    # "用户"的标识字符串

    # 构造一个示例对话，作为 few-shot / in-context 学习用的对话先验
    exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)
    
    # ===== 2) 选择LLM后端与对应的Agent/Planner封装 =====
    # 不同模型对应不同的对话包装类（纯文本 vs Chat API），以及不同的系统规划器
    if cmd_args.llm in ['code-davinci-002']:
        backbone_model = OpenAIModel(cmd_args.llm)   # 旧版Completion风格
        SysModel = PersuaderModel                    # 劝说者（系统端）- 文本模型
        UsrModel = PersuadeeModel                    # 被劝说者（用户端）- 文本模型
        SysPlanner = P4GSystemPlanner                # 规划器（基于P4G任务）
    elif cmd_args.llm in ['gpt-3.5-turbo']:
        backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)  # Chat风格
        SysModel = PersuaderChatModel
        UsrModel = PersuadeeChatModel
        SysPlanner = P4GChatSystemPlanner
    elif cmd_args.llm == 'chatgpt':
        backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)  # Azure Chat
        SysModel = PersuaderChatModel
        UsrModel = PersuadeeChatModel
        SysPlanner = P4GChatSystemPlanner
    
    # ===== 3) 构造系统（劝说者）与用户（被劝说者）代理 =====
    # 设置推理超参：温度、采样等。注意do_sample=True是为了“开放环”MCTS需要的多样化生成
    system = SysModel(
        sys_da,
        backbone_model, 
        conv_examples=[exp_1],          # few-shot演示（In-Context）
        inference_args={
            "temperature": 0.7,         # 系统略偏保守（较低温度）
            "do_sample": True,          # 开放环MCTS依赖采样来产生多样化realizations
            "return_full_text": False,
        }
    )
    user = UsrModel(
        user_da,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,         # 用户端更高温度 → 回答更发散
            "repetition_penalty": 1.0,
            "do_sample": True,          # 同上
            "return_full_text": False,
        },
        backbone_model=backbone_model, 
        conv_examples=[exp_1]
    )

    # 系统规划器：根据对话行为空间/历史，借助LLM生成候选并用于MCTS估值
    planner = SysPlanner(
        dialog_acts=system.dialog_acts,
        max_hist_num_turns=system.max_hist_num_turns,
        user_dialog_acts=user.dialog_acts,
        user_max_hist_num_turns=user.max_hist_num_turns,
        generation_model=backbone_model,
        conv_examples=[exp_1]
    )

    # 博弈环境：把system与user代理装配到同一游戏（劝说对话）中
    game = PersuasionGame(system, user)

    print(f"System dialog acts: {system.dialog_acts}")
    print(f"User dialog acts: {user.dialog_acts}")

    # ===== 4) 加载评测用的“回合式标注对话”数据（来自P4G）=====
    # 文件结构：all_dialogs[did]["dialog"] & ["label"] 等
    with open("data/p4g/300_dialog_turn_based.pkl", "rb") as f:
        all_dialogs = pickle.load(f)

    # ===== 5) MCTS配置参数 =====
    num_dialogs = cmd_args.num_dialogs
    args = dotdict({
        "cpuct": 1.0,                          # MCTS探索-利用平衡系数
        "num_MCTS_sims": cmd_args.num_mcts_sims,     # 每个状态进行多少次模拟
        "Q_0": cmd_args.Q_0,                    # 未访问动作的初始Q值（影响探索）
        "max_realizations": cmd_args.max_realizations,  # 每状态rollout实现的数量k
    })

    # 评测输出：用于后续静态评估（test.py）
    # 每条元素包含：对话id、上下文、人工下一句/DA、模型生成下一句/DA、以及调试信息
    output = []  

    # 一些对话包含敏感内容，OpenAI接口会触发过滤/报错，这里直接跳过
    bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']
    num_done = 0
    pbar = tqdm(total=num_dialogs, desc="evaluating")

    # ===== 6) 逐个对话执行：滚动构建状态 → 运行MCTS → 取策略与实现 → 保存结果 =====
    for did in all_dialogs.keys():
        if did in bad_dialogs:
            print("skipping dialog id: ", did)
            continue
        if num_done == num_dialogs:
            break

        print("evaluating dialog id: ", did)
        context = ""                 # 为评测judge整理的可读上下文（文本串）
        dialog = all_dialogs[did]
        
        state = game.init_dialog()   # 初始化游戏状态（空历史）

        # 遍历该对话的每一回合（t指当前turn）
        for t, turn in enumerate(dialog["dialog"]):
            if len(turn["ee"]) == 0:    # 若该turn用户端为空，说明对话已结束
                break
            if t == len(dialog["dialog"]) - 1:
                # 最后一个turn没有“下一轮系统响应”可对比，跳过
                break

            # 用户发言与标签（原始数据里ee为用户回复列表）
            usr_utt = " ".join(turn["ee"]).strip()
            usr_da = dialog["label"][t]["ee"][-1]  # 取该轮用户端最后一个标注DA

            # 将原始标签映射到本文设定的统一DA集合（消歧/归并）
            if usr_da == "disagree-donation":
                usr_da = PersuasionGame.U_NoDonation
            elif usr_da == "negative-reaction-to-donation":
                usr_da = PersuasionGame.U_NegativeReaction
            elif usr_da == "positive-reaction-to-donation":
                usr_da = PersuasionGame.U_PositiveReaction
            elif usr_da == "agree-donation":
                usr_da = PersuasionGame.U_Donate
            else:
                usr_da = PersuasionGame.U_Neutral

            # 若用户已“同意捐赠”，博弈结束（无需再规划下一轮）
            if usr_da == PersuasionGame.U_Donate:
                break

            # 系统上一轮发言与DA（er为系统回复列表）
            sys_utt = " ".join(turn["er"]).strip()
            sys_da = set(dialog["label"][t]["er"])
            # 与系统可用DA集合求交，若无交集则标为"other"
            intersected_das = sys_da.intersection(system.dialog_acts)
            if len(intersected_das) == 0:
                sys_da = "other"
            else:
                sys_da = list(intersected_das)[-1]
            
            # 把“上一轮系统-本轮用户”的对话对加入到状态轨迹中（供MCTS使用）
            state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
            state.add_single(PersuasionGame.USR, usr_da, usr_utt)
            print("sys_utt: ", sys_utt)
            print("usr_utt: ", usr_utt)
            print("--------------------------------")

            # 更新可读上下文（交给judge比较“下一轮系统回复”的优劣）
            context = f"""
            {context}
            Persuader: {sys_utt}
            Persuadee: {usr_utt}
            """
            context = context.replace('\t', '').strip()

            # ===== 6.1) 运行Open-Loop MCTS进行策略搜索 =====
            # 对于OpenAI Completion类模型，清空缓存以避免旧上下文污染结果
            if isinstance(backbone_model, OpenAIModel):
                backbone_model._cached_generate.cache_clear()

            # 构造一个新的MCTS搜索器（给定当前对话状态）
            dialog_planner = OpenLoopMCTS(game, planner, args)
            print("searching")
            for i in tqdm(range(args.num_MCTS_sims)):
                dialog_planner.search(state)  # 单次模拟：选择-扩展-仿真-回传

            # 从MCTS得到当前状态下各对话行为的概率分布（策略）
            mcts_policy = dialog_planner.get_action_prob(state)
            mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]  # 选概率最大的DA

            # 从搜索树中取出“最优DA对应的最佳生成实句”（来自rollout的realization）
            mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

            # 取“下一轮”人工系统回复及其DA（作为对照/评测用）
            human_resp = " ".join(dialog["dialog"][t+1]["er"]).strip()
            next_sys_das = set(dialog["label"][t+1]["er"])
            next_intersected_das = next_sys_das.intersection(system.dialog_acts)
            if len(next_intersected_das) == 0:
                next_sys_da = "other"
            else:
                next_sys_da = list(next_intersected_das)[-1]

            # 调试信息：保存整棵搜索树的统计量（访问次数、Q、先验P、value、以及各realization的信息）
            debug_data = {
                "probs": mcts_policy,
                "da": mcts_policy_next_da,
                "search_tree": {
                    "Ns": dialog_planner.Ns,                 # 节点访问次数
                    "Nsa": dialog_planner.Nsa,               # (s,a)访问次数
                    "Q": dialog_planner.Q,                   # 动作价值
                    "P": dialog_planner.P,                   # 先验策略
                    "Vs": dialog_planner.Vs,                 # 节点估值
                    "realizations": dialog_planner.realizations,           # 各动作采样生成的候选文本
                    "realizations_Vs": dialog_planner.realizations_Vs,     # 候选的估值
                    "realizations_Ns": dialog_planner.realizations_Ns,     # 候选被使用的次数
                },
            }

            # 组织一条对比样本：上下文 + 人类回复 + 模型回复 + 各自DA + 调试信息
            cmp_data = {
                'did': did,
                'context': context,
                'ori_resp': human_resp,
                'ori_da': next_sys_da,
                'new_resp': mcts_pred_rep,
                'new_da': mcts_policy_next_da,
                "debug": debug_data,
            }
            output.append(cmp_data)

            # 若开启debug，则打印该回合的核心对比
            if cmd_args.debug:
                print(context)
                print("human resp: ", human_resp)
                print("human da: ", next_sys_da)
                print("mcts resp: ", mcts_pred_rep)
                print("mcts da: ", mcts_policy_next_da)

        # 每完成一个对话就把累计结果序列化到输出文件（断点续跑友好）
        with open(cmd_args.output, "wb") as f:
            pickle.dump(output, f)

        num_done += 1
        pbar.update(1)
    return


if __name__ == "__main__":
    # ===== 7) 命令行参数定义 =====
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="outputs/gdpzero.pkl",
                        help='output file')
    parser.add_argument('--llm', type=str, default="code-davinci-002",
                        choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo"],
                        help='OpenAI model name')
    parser.add_argument('--gen_sentences', type=int, default=-1,
                        help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
    parser.add_argument('--num_mcts_sims', type=int, default=20,
                        help='number of mcts simulations')
    parser.add_argument('--max_realizations', type=int, default=3,
                        help='number of realizations per mcts state')
    parser.add_argument('--Q_0', type=float, default=0.0,
                        help='initial Q value for unitialized states. to control exploration')
    parser.add_argument('--num_dialogs', type=int, default=20,
                        help='number of dialogs to test MCTS on')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.parse_args()
    cmd_args = parser.parse_args()
    print("saving to", cmd_args.output)

    main(cmd_args)
