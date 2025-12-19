import numpy as np
import logging
import pickle
import argparse

from tqdm.auto import tqdm
from core.gen_models import (
	LocalModel, OpenAIModel, OpenAIChatModel, AzureOpenAIChatModel
)
# [新增] 引入我们之前写好的 CB 类
from core.players import (
    PersuadeeModel, PersuaderModel, P4GSystemPlanner,
    PersuaderChatModel, PersuadeeChatModel, P4GChatSystemPlanner,
    CBBuyerChatModel, CBSellerChatModel, CBSystemPlanner
)
from core.game import PersuasionGame
from core.helpers import DialogSession
from utils.prompt_examples import EXP_DIALOG


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(cmd_args):
	
	if cmd_args.dataset == 'cb':
		# CB 的 11 个动作 (TRIP Table 9)
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
	else:
		game_ontology = PersuasionGame.get_game_ontology()
		sys_da = game_ontology['system']['dialog_acts']
		user_da = game_ontology['user']['dialog_acts']
		system_name = PersuasionGame.SYS
		user_name = PersuasionGame.USR
		
	exp_1 = DialogSession(system_name, user_name).from_history(EXP_DIALOG)

	# ==========================================
    # 2. [修改] Examples 设置 (CB 为空)
    # ==========================================
	if cmd_args.dataset == 'p4g':
		conv_examples = [exp_1]
	else:
		conv_examples = []

	# ==========================================
    # 3. [修改] 模型类选择
    # ==========================================
	if cmd_args.llm == 'code-davinci-002':
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



	system = SysModel(
		sys_da,
		backbone_model, 
		conv_examples=[exp_1]
	)
# 实例化 User
    # [注意] 显式传参 max_hist_num_turns=5 以防报错
	user = UsrModel(
        user_da,
        inference_args={
            "max_new_tokens": 128,
            "temperature": 1.1,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "return_full_text": False,
        },
        backbone_model=backbone_model, 
        conv_examples=conv_examples,
        max_hist_num_turns=5 
    )
	planner = SysPlanner(
		dialog_acts=system.dialog_acts,
		max_hist_num_turns=system.max_hist_num_turns,
		user_dialog_acts=user.dialog_acts,
		user_max_hist_num_turns=user.max_hist_num_turns,
		generation_model=backbone_model,
		conv_examples=conv_examples
	)
	game = PersuasionGame(system, user)



# ==========================================
    # 4. [修改] 数据加载路径
    # ==========================================
	if cmd_args.dataset == 'p4g':
		pkl_path = "data/p4g/300_dialog_turn_based.pkl"
	else:
		pkl_path = "data/cb/cb_test.pkl"

	with open(pkl_path, "rb") as f:
		all_dialogs = pickle.load(f)

	num_dialogs = 20

	output = []  # for evaluation. [{did, context, ori_resp, new_resp, debug}, ...]
	# those dialogs has inappropriated content and will throw an error/be filtered with OPENAI models
	bad_dialogs = ['20180808-024552_152_live', '20180723-100140_767_live', '20180825-080802_964_live']
	num_done = 0
	pbar = tqdm(total=num_dialogs, desc="evaluating")
	for did in all_dialogs.keys():
		if did in bad_dialogs:
			print("skipping dialog id: ", did)
			continue
		if num_done == num_dialogs:
			break

		print("evaluating dialog id: ", did)
		context = ""
		no_error = True
		dialog = all_dialogs[did]

		# ==========================================
        # 5. [新增] CB 专属：注入商品信息
        # ==========================================
		if cmd_args.dataset == 'cb':
			item_info = dialog.get('item_info', {})
			system.set_item_info(item_info)
			user.set_item_info(item_info)
			planner.set_item_info(item_info)
		
		state = game.init_dialog()
		for t, turn in enumerate(dialog["dialog"]):
			if len(turn["ee"]) == 0:  # ended
				break
			# also skip last turn as there is no evaluation
			if t == len(dialog["dialog"]) - 1:
				break

			usr_utt = " ".join(turn["ee"]).strip()

			if cmd_args.dataset == 'p4g':
				usr_da = dialog["label"][t]["ee"][-1]

				# map to our dialog act
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

			elif cmd_args.dataset == 'cb':
				usr_da = "inform" # CB dummy DA

			# map sys as well
			sys_utt = " ".join(turn["er"]).strip()

			if cmd_args.dataset == 'p4g':
				sys_da = set(dialog["label"][t]["er"])
				intersected_das = sys_da.intersection(system.dialog_acts)
				if len(intersected_das) == 0:
					sys_da = "other"
				else:
					sys_da = list(intersected_das)[-1]
			elif cmd_args.dataset == 'cb':
				sys_da = "inform" # CB dummy DA
			
			state.add_single(PersuasionGame.SYS, sys_da, sys_utt)
			state.add_single(PersuasionGame.USR, usr_da, usr_utt)

			# update context for evaluation
            # [修改] 角色名称动态化
			p_role = "Persuader" if cmd_args.dataset == 'p4g' else "Buyer"
			u_role = "Persuadee" if cmd_args.dataset == 'p4g' else "Seller"

			# update context for evaluation
			context = f"""
			{context}
			{p_role}: {sys_utt}
			{u_role}: {usr_utt}
			"""
			context = context.replace('\t', '').strip()

			# mcts policy
			prior, v = planner.predict(state)
			greedy_policy = system.dialog_acts[np.argmax(prior)]
			try:
				next_best_state = game.get_next_state(state, np.argmax(prior))
			except Exception as e:
				bad_dialogs.append(did)
				no_error = False
				raise e
			greedy_pred_resp = next_best_state.history[-2][2]

			# next ground truth utterance
			human_resp = " ".join(dialog["dialog"][t + 1]["er"]).strip()

			if cmd_args.dataset == 'p4g':
				next_sys_das = set(dialog["label"][t+1]["er"])
				next_intersected_das = next_sys_das.intersection(system.dialog_acts)
				if len(next_intersected_das) == 0:
					next_sys_da = "other"
				else:
					next_sys_da = list(next_intersected_das)[-1]
			elif cmd_args.dataset == 'cb':
				next_sys_da = "unknown"


			# logging for debug
			debug_data = {
				"prior": prior,
				"da": greedy_policy,
				"v": v
			}

			# update data
			cmp_data = {
				'did': did,
				'context': context,
				'ori_resp': human_resp,
				'ori_da': next_sys_da,
				'new_resp': greedy_pred_resp,
				'new_da': greedy_policy,
				"debug": debug_data,
			}
			# [新增] CB 额外保存 item_info
			if cmd_args.dataset == 'cb':
				cmp_data['item_info'] = dialog.get('item_info', {})
			output.append(cmp_data)
		
		if no_error:
			with open(cmd_args.output, "wb") as f:
				pickle.dump(output, f)
			pbar.update(1)
			num_done += 1
	pbar.close()
	print(bad_dialogs)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# [新增] dataset 参数
	parser.add_argument('--dataset', type=str, default="p4g", choices=["p4g", "cb"], help='Dataset to use')
	parser.add_argument('--llm', type=str, default="code-davinci-002", choices=["code-davinci-002", "gpt-3.5-turbo", "chatgpt"], help='OpenAI model name')
	parser.add_argument('--gen_sentences', type=int, default=-1, help='max number of sentences to generate. -1 for no limit')
	parser.add_argument('--output', type=str, default="outputs/raw_prompt.pkl", help='output file')
	parser.parse_args()
	cmd_args = parser.parse_args()
	print("saving to", cmd_args.output)

	main(cmd_args)