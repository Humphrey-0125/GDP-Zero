import numpy as np
import logging
import pickle
import argparse
import numpy as np

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
from core.players import (
	PersuadeeModel, PersuaderModel, P4GSystemPlanner,
	PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner,
	# --- [新增] 引入我们刚才在 players.py 里写的 CB 类 ---
    CBBuyerChatModel, CBSellerChatModel, CBSystemPlanner
)
from core.game import PersuasionGame
from core.mcts import OpenLoopMCTS
from core.helpers import DialogSession
from utils.utils import dotdict
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):

	# =========================================================================
    # 1. 游戏本体 (Ontology) 与动作空间定义
    # =========================================================================
    # 根据传入的 dataset 参数，决定使用哪套动作空间（System DA）。
	if cmd_args.dataset == 'cb':
        # [修正] 这里的列表必须和 players.py 中的 da_prompts_mapping 的 Key 完全一致
		sys_da = [
            "greetings", 
            "ask_question", 
            "answer_question", 
            "propose_first_price", 
            "propose_counter_price", 
            "use_comparatives", 
            "confirm_information", 
            "affirm_confirmation", 
            "deny_confirmation", 
            "agree_proposal", 
            "disagree_proposal"
        ]
		user_da = sys_da
		system_name = "Buyer"
		user_name = "Seller"
	elif cmd_args.dataset == 'p4g':
		# 如果是 P4G，从 PersuasionGame 类中直接获取定义好的本体
		game_ontology = PersuasionGame.get_game_ontology()
		sys_da = game_ontology['system']['dialog_acts']
		user_da = game_ontology['user']['dialog_acts']
		system_name = PersuasionGame.SYS
		user_name = PersuasionGame.USR


	
	# =========================================================================
    # 2. 准备 Few-shot Examples
    # =========================================================================
    # P4G 依赖这个示例来规范输出格式；CB 使用 Zero-shot (Instruct Prompt)，所以传空列表

	# 加载 P4G 的少样本示例 (One-shot example)，用于 In-context Learning
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	if cmd_args.dataset == 'p4g':
		conv_examples = [exp_1]
	else:
		conv_examples = []


	# =========================================================================
    # 3. 选择模型类 (Model Class Selection)
    # =========================================================================
	if cmd_args.llm in ['code-davinci-002']:
		backbone_model = OpenAIModel(cmd_args.llm)
		SysModel = PersuaderModel
		UsrModel = PersuadeeModel
		SysPlanner = P4GSystemPlanner
	elif cmd_args.llm in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		if cmd_args.dataset == 'cb':
			SysModel = CBBuyerChatModel
			UsrModel = CBSellerChatModel
			SysPlanner = CBSystemPlanner
		else:
			SysModel = PersuaderChatModel
			UsrModel = PersuadeeChatModel
			SysPlanner = P4GChatSystemPlanner
	elif cmd_args.llm == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(cmd_args.llm, cmd_args.gen_sentences)
		SysModel = PersuaderChatModel
		UsrModel = PersuadeeChatModel
		SysPlanner = P4GChatSystemPlanner
	


	# =========================================================================
    # 4. 实例化智能体 (Agent Instantiation)
    # =========================================================================
    # 实例化 System (GDP-Zero 控制的一方：P4G中是劝导者，CB中是买家)
	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=conv_examples,   # p4g这个地方会用到，而cb时，只会是个空列表
		inference_args={
			"temperature": 0.7,
			"do_sample": True,  # for MCTS open loop
			"return_full_text": False,
		}
	)
	# 实例化 User (Simulator)
    # [关键修复] 全部使用 keyword arguments，避免 dict 被当成 positional arg 传给 max_hist_num_turns
	user = UsrModel(
        dialog_acts=user_da,
        backbone_model=backbone_model,
        max_hist_num_turns=5,  # 显式指定，确保它是 int
        conv_examples=conv_examples,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "return_full_text": False,
        }
    )

	# 实例化规划器 (Planner) - 这是 MCTS 搜索时用到的辅助类 !!!!!!!!!!!!!!!1
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=conv_examples    
	)
	game = PersuasionGame(system, user)

	print(f"System dialog acts: {system.dialog_acts}")
	print(f"User dialog acts: {user.dialog_acts}")

	# =========================================================================
    # 5. 加载数据 (Data Loading)
    # =========================================================================
	if cmd_args.dataset == 'p4g':
		pkl_path = "data/p4g/300_dialog_turn_based.pkl"
	if cmd_args.dataset == 'cb':
		pkl_path = "data/cb/cb_test.pkl" # 之前转换好的文件
    
	with open(pkl_path, "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = cmd_args.num_dialogs
	args = dotdict({
		"cpuct": 1.0,
		"num_MCTS_sims": cmd_args.num_mcts_sims,
		"Q_0": cmd_args.Q_0,
		"max_realizations": cmd_args.max_realizations,
	})

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models. See raw_prompting.py file for more details
	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']  # throws exception due to ChatGPT API filtering
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")


	# =========================================================================
    # 6. 主循环：遍历对话 (Main Loop)
    # =========================================================================
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			print("skipping dialog id: ", did)
			continue
		if num_done == num_dialogs:
			break

		print("evaluating dialog id: ", did)
		context = ""
		dialog = all_dialogs[did]

		# --- [CB 专属逻辑] 注入商品信息 ---
        # 这一步至关重要！因为 CB 每个对话卖的东西不一样。
        # 必须把 title/price/description 塞进模型，否则模型不知道自己在卖啥。
		if cmd_args.dataset == 'cb':
			item_info = dialog.get('item_info', {})
			# 调用我们在 players.py 里新加的方法
			system.set_item_info(item_info)
			user.set_item_info(item_info)
			planner.set_item_info(item_info)
		
		state = game.init_dialog()

		# --- [回合重放] 遍历历史对话 ---
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:  # ended
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()
			# usr_da = dialog["label"][t]["ee"][-1]

			# --- [动作映射] 将数据集标签映射到 MCTS 动作空间 ---
			if cmd_args.dataset == 'p4g':
				usr_da = dialog["label"][t]["ee"][-1]
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

				# game ended
				if usr_da == PersuasionGame.U_Donate:
					break
			else: # CB 数据集
				# CB test集没有DA标注，我们不需要用来做MCTS的终止条件，直接继续
				usr_da = "inform" # 使用 dummy DA

			# 获取系统说的话
			sys_utt = " ".join(turn["er"]).strip()

			if cmd_args.dataset == 'p4g':
				sys_da = set(dialog["label"][t]["er"])
				intersected_das = sys_da.intersection(system.dialog_acts)
				if len(intersected_das) == 0:
					sys_da = "other"
				else:
					sys_da = list(intersected_das)[-1]
			else:
				sys_da = "inform" # CB dummy DA
			
			# 更新状态 (State)：将历史回合加入
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# 更新用于 Prompt 的上下文文本 (Context String)
            # 根据数据集切换角色名 (Persuader vs Buyer)
			p_role = "Persuader" if cmd_args.dataset == 'p4g' else "Buyer"
			u_role = "Persuadee" if cmd_args.dataset == 'p4g' else "Seller"

			# update context for evaluation
			context = f"""
			{context}
			{p_role}: {sys_utt}
			{u_role}: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# =================================================================
            # 7. MCTS 搜索核心 (The Core)
            # =================================================================
            # 清除缓存，防止 OpenAI API 复用旧结果
			if isinstance(backbone_model, OpenAIModel):
				backbone_model._cached_generate.cache_clear()

			# 初始化 Open-Loop MCTS 规划器
			dialog_planner = OpenLoopMCTS(game, planner, args)
			print("searching")
			for i in tqdm(range(args.num_MCTS_sims)):
				dialog_planner.search(state)

			# --- [获取结果] ---
            # 1. 获取访问次数最多的动作 (Policy)
			mcts_policy = dialog_planner.get_action_prob(state)
			mcts_policy_next_da = system.dialog_acts[np.argmax(mcts_policy)]

			# 2. 获取该动作下最好的文本回复 (Best Realization)
			mcts_pred_rep = dialog_planner.get_best_realization(state, np.argmax(mcts_policy))

			# 3. 获取人类的真实下一句回复 (Ground Truth) 用于对比
			human_resp = " ".join(dialog["dialog"][t+1]["er"]).strip()

			# 获取人类的真实动作 (仅 P4G 有)  # 为什么这块CB不需要呢？？？？？？
			if cmd_args.dataset == 'p4g':
				next_sys_das = set(dialog["label"][t+1]["er"])
				next_intersected_das = next_sys_das.intersection(system.dialog_acts)
				if len(next_intersected_das) == 0:
					next_sys_da = "other"
				else:
					next_sys_da = list(next_intersected_das)[-1]
			else:
				next_sys_da = "unknown"

			# =================================================================
            # 8. 保存结果 (Update Output)
            # =================================================================
			debug_data = {
				"probs": mcts_policy,
				"da": mcts_policy_next_da,
				"search_tree": {
					"Ns": dialog_planner.Ns,
					"Nsa": dialog_planner.Nsa,
					"Q": dialog_planner.Q,
					"P": dialog_planner.P,
					"Vs": dialog_planner.Vs,
					"realizations": dialog_planner.realizations,
					"realizations_Vs": dialog_planner.realizations_Vs,
					"realizations_Ns": dialog_planner.realizations_Ns,
				},
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': mcts_pred_rep,
				'new_da': mcts_policy_next_da,
				"debug": debug_data,
			}
			# CB 额外保存 item_info 方便 debug
			if cmd_args.dataset == 'cb':
				cmp_data['item_info'] = dialog.get('item_info', {})
			output.append(cmp_data)

			if cmd_args.debug:
				print(context)
				print("human resp: ", human_resp)
				print("human da: ", next_sys_da)
				print("mcts resp: ", mcts_pred_rep)
				print("mcts da: ", mcts_policy_next_da)
		with open(cmd_args.output, "wb") as f:
			pickle.dump(output, f)
		num_done += 1
		pbar.update(1)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default="p4g", choices=["p4g", "cb"], help='Dataset to use')
	parser.add_argument('--output', type=str, default="outputs/gdpzero.pkl", help='output file')
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "chatgpt", "gpt-3.5-turbo"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='number of sentences to generate from the llm. Longer ones will be truncated by nltk.')
	parser.add_argument('--num_mcts_sims', type=int, default=20, help='number of mcts simulations')
	parser.add_argument('--max_realizations', type=int, default=3, help='number of realizations per mcts state')
	parser.add_argument('--Q_0', type=float, default=0.0, help='initial Q value for unitialized states. to control exploration')
	parser.add_argument('--num_dialogs', type=int, default=20, help='number of dialogs to test MCTS on')
	parser.add_argument('--debug', action='store_true', help='debug mode')
	parser.parse_args()
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)

	main(cmd_args)