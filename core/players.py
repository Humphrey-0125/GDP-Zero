import logging
import numpy as np

from typing import List, Tuple
from core.helpers import DialogSession
from core.gen_models import GenerationModel, DialogModel
from core.game import PersuasionGame
from abc import ABC, abstractmethod
from collections import Counter


logger = logging.getLogger(__name__)


class DialogPlanner(ABC):
	@abstractmethod
	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		pass

	@abstractmethod
	def predict(self, state) -> "Tuple[np.ndarray, float]":
		# returns a prob and value
		pass


class P4GSystemPlanner(DialogPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__()
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns  # used in prompting next da
		self.user_dialog_acts = user_dialog_acts
		self.user_max_hist_num_turns = user_max_hist_num_turns  # used in heuristic function
		self.conv_examples = conv_examples
		self.generation_model = generation_model
		self.smoothing = 1.0
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The Persuader can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()

		self.inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return

	def process_exp(self, keep_sys_da=True, keep_user_da=False):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_sys_da=keep_sys_da, keep_user_da=keep_user_da) + "\n"
		return prompt_exps.strip()

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{state.to_string_rep(keep_sys_da=True)}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		logger.debug(prompt)
		data = self.generation_model.generate(prompt, **self.inf_args)
		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)
		prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is a conversation between a Persuader and	a Persuadee about a charity called Save the Children. 
		{self.process_exp(keep_sys_da=False, keep_user_da=True)}
		The following is a new conversation between another Persuader and Persuadee.
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.user_max_hist_num_turns)}
		Persuader: Would you be interested in donating to Save the Children?
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()

		inf_args = {
			"max_new_tokens": 8,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.generate(prompt, **inf_args)
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"persuadee prompt: {prompt}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == PersuasionGame.U_NoDonation:
				score.append(-1.0)
			elif da == PersuasionGame.U_NegativeReaction:
				score.append(-0.5)
			elif da == PersuasionGame.U_Neutral:
				score.append(0.0)
			elif da == PersuasionGame.U_PositiveReaction:
				score.append(0.5)
			elif da == PersuasionGame.U_Donate:
				score.append(1.0)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)
	

class P4GChatSystemPlanner(P4GSystemPlanner):
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model:GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		super().__init__(
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns,
			generation_model, conv_examples
		)
		self.task_prompt = f"""
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children. You can choose amongst the following actions during a conversation:
		{" ".join([f"[{da}]" for da in dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and a Persuadee."
		self.prompt_examples = self.process_chat_exp(new_task_prompt=self.new_task_prompt)

		self.inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}
		return
	
	def process_chat_exp(self, 
			new_task_prompt,
			assistant_role=PersuasionGame.SYS,
			keep_sys_da=True, keep_user_da=False):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp, keep_sys_da, keep_user_da, assistant_role)
			prompt_exps.append({
				"role":"system", "content": new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self,
			exp:DialogSession, 
			keep_sys_da, keep_user_da,
			assistant_role=PersuasionGame.SYS,
			max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system/Persuader
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		# init with user
		# if assistant_role == PersuasionGame.SYS:
		#	 if keep_user_da:
		#		 prompt_messages.append({
		#			 "role": "user",
		#			 "content": f"{PersuasionGame.USR}: [{PersuasionGame.U_Neutral}] Hello.".strip()
		#		 })
		#	 else:
		#		 prompt_messages.append({
		#			 "role": "user",
		#			 "content": f"{PersuasionGame.USR}: Hello.".strip()
		#		 })
		# all the rest
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			# if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
			if role == PersuasionGame.SYS:
				if keep_sys_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == PersuasionGame.SYS:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			else:
				if keep_user_da:
					content = f"{role}: [{da}] {utt}".strip()
				else:
					content = f"{role}: {utt}".strip()
				if assistant_role == PersuasionGame.USR:
					prompt_role = "assistant"
				else:
					prompt_role = "user"
			
			prompt_messages.append({
				"role": prompt_role,
				"content": content
			})
		return prompt_messages

	def get_valid_moves(self, state):
		# 1 if the i-th dialog act is valid, 0 otherwise
		turn = len(state)
		if turn < 1:
			return np.array([1 if da == PersuasionGame.S_Greeting else 0 for da in self.dialog_acts])
		return np.array([1 for _ in self.dialog_acts])

	def get_utterance(self, state, action) -> str:
		return ""  # should not be called

	def _get_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def predict(self, state:DialogSession) -> "Tuple[np.ndarray, float]":
		# test k times and compute prob. See num_return_sequences in the API
		# the value would be our objective function
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self.__proccess_chat_exp(state, keep_sys_da=True, keep_user_da=False)
		# produce a response
		print("P4G预测过程中传入chat_generate的messages是：", messages)
		data = self.generation_model.chat_generate(messages, **self.inf_args)
		print("P4G预测过程中chat_generate的输出data是：", data)


		sampled_das = self._get_generated_da(data)
		logger.debug(f"sampled das: {sampled_das}")
		# convert to prob distribution
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing
		for da in sampled_das:
			prob[self.dialog_acts.index(da)] += 1
		prob /= prob.sum()
		v = self.heuristic(state)
		return prob, v

	def _get_user_generated_da(self, data) -> list:
		# convert generated responses to DA
		pred_da = []
		for resp in data:
			resp = resp['generated_text'].strip()
			start_idx = resp.find("[")
			end_idx = resp.find("]")
			if start_idx == -1 or end_idx == -1:
				continue
			found_da = resp[start_idx + 1: end_idx].strip()
			if found_da in self.user_dialog_acts:
				pred_da.append(found_da)
		return pred_da

	def heuristic(self, state:DialogSession) -> float:
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)
		user_task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.user_dialog_acts])}
		The following is a new conversation between a Persuader and a Persuadee (you).
		""".replace("\t", "").strip()
		user_new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you)."

		messages = [
			{'role': 'system', 'content': user_task_prompt},
			*self.process_chat_exp(new_task_prompt=user_new_task_prompt, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True),
			{'role': 'system', 'content': user_new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, assistant_role=PersuasionGame.USR, keep_sys_da=False, keep_user_da=True)
		messages.append({
			'role': 'user', 'content': f'{PersuasionGame.SYS}: Would you be interested in donating to Save the Children?'
		})

		inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}
		data = self.generation_model.chat_generate(messages, **inf_args)
		sampled_das = self._get_user_generated_da(data)

		logger.debug(f"persuadee prompt: {messages}")
		logger.debug(f"sampled das: {sampled_das}")

		# heuristic score
		score = []
		for da in sampled_das:
			if da == PersuasionGame.U_NoDonation:
				score.append(-1.0)
			elif da == PersuasionGame.U_NegativeReaction:
				score.append(-0.5)
			elif da == PersuasionGame.U_Neutral:
				score.append(0.0)
			elif da == PersuasionGame.U_PositiveReaction:
				score.append(0.5)
			elif da == PersuasionGame.U_Donate:
				score.append(1.0)
		v = 0.0 if len(score) == 0 else np.mean(score)
		logger.debug(f"sampled das to v: {v}")
		return float(v)


class PersuaderModel(DialogModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.max_hist_num_turns = max_hist_num_turns
		# prompts and DAs
		self.da_prompts_mapping = {
			PersuasionGame.S_Greeting:					 "The Persuader greets the Persuadee.",
			# start of persuasion strategies
			PersuasionGame.S_CredibilityAppeal:			 "The Persuader establishes credibility of Save the Children by citing its impact.",
			PersuasionGame.S_EmotionAppeal:				 "The Persuader uses an emotion appeal to convince the Persuadee.",
			PersuasionGame.S_LogicalAppeal:				 "The Persuader use of reasoning and evidence to convince the Persuadee.",
			PersuasionGame.S_TaskRelatedInquiry:		 "The Persuader asks about the Persuadee's knowledge or opinion related to Save the Children.",
			PersuasionGame.S_PropositionOfDonation:		 "The Persuader asks if the Persuadee would like to make a small donation.",
			# end of persuasion strategies
			PersuasionGame.S_Other:						 "The Persuader responds to the Persuadee without using any persuaive strategy.",
		}
		# only allow da that has the mapping
		self.dialog_acts = [da for da in dialog_acts if da in self.da_prompts_mapping]
		
		logger.debug(self.dialog_acts)
		self.task_prompt = f"""
		The following is background information about Save the Children. 
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level
			"return_full_text": False,
			**inference_args
		}
		return

	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_exp(exp) + "\n"
		return prompt_exps.strip()

	def __proccess_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		prompt_exp = ""
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			
			if role == PersuasionGame.SYS:
				prompt_exp += f"{self.da_prompts_mapping[da]}\n{role}: {utt}\n"
			else:
				prompt_exp += f"{role}: {utt}\n"
		return prompt_exp.strip()
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		print("PersuaderModel get_utterance")
		# planner gives an action, state is history, you need to produce a response accrd to the action
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		if len(state) == 0:
			prompt = f"""
			{self.task_prompt}
			{da_prompt}
			Persuader:
			"""
		else:
			prompt = f"""
			{self.task_prompt}
			{self.__proccess_exp(state, max_hist_num_turns=self.max_hist_num_turns)}
			{da_prompt}
			Persuader:
			"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		sys_resp = self.backbone_model._cleaned_resp(data, prompt)[0]  # TODO
		return sys_resp

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError
	

class PersuaderChatModel(PersuaderModel):
	def __init__(self,
			dialog_acts:List[str],
			backbone_model:GenerationModel,
			max_hist_num_turns: int = 5,
			conv_examples: List[DialogSession] = [],
			inference_args: dict = {}):
		super().__init__(
			dialog_acts=dialog_acts,
			backbone_model=backbone_model,
			max_hist_num_turns=max_hist_num_turns,
			conv_examples=conv_examples,
			inference_args=inference_args
		)
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.0,
			"repetition_penalty": 1.0,
			"do_sample": False,  # otherwise tree will never go to the next level, unless you do OpenLoop search
			"return_full_text": False,
			**inference_args
		}
		self.task_prompt = """
		Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		You are Persuader who is trying to persuade the Persuadee to donate to a charity called Save the Children.
		The following is an example conversation between a Persuader and a Persuadee about Save the Children.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between Persuader (you) and another Persuadee.\nThe Persuader greets the persuadee."
		self.prompt_examples = self.process_chat_exp()
		return

	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		
		next_sys_da = PersuasionGame.S_Greeting
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == PersuasionGame.SYS:
				prompt_messages.append({
					"role": "assistant",
					"content": f"{role}: {utt}".strip()
				})
			else:
				if i+1 < len(exp.history):
					next_sys_da = exp[i+1][1]
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}\n{self.da_prompts_mapping[next_sys_da]}".strip()
					})
				else:
					prompt_messages.append({
						"role": "user",
						"content": f"{role}: {utt}".strip()
					})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action:int) -> str:
		print("PersuaderChatModel get_utterance")
		return self.get_utterance_batched(state, action, batch=1)[0]
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int=3) -> List[str]:
		da = self.dialog_acts[action]
		da_prompt = self.da_prompts_mapping[da]
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		
		if len(state) == 0:
			messages.append({'role': 'user', 'content': f'{PersuasionGame.USR}: Hello.\n{da_prompt}'})
		else:
			assert(state[-1][0] == PersuasionGame.USR)
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
		gen_args = {
			**self.inference_args,
			"num_return_sequences": batch,  # this will be changed to n inside chat_generate
		}
		print("PersuaderChatModel messages:", messages)
		data = self.backbone_model.chat_generate(messages, **gen_args)
		sys_resps = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.SYS}:", user_role=f"{PersuasionGame.USR}:"
		)
		return sys_resps

	def get_utterance_w_da(self, state: DialogSession, action) -> Tuple[str, str]:
		raise NotImplementedError


class PersuadeeModel(DialogModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__()
		self.conv_examples = conv_examples
		self.backbone_model = backbone_model
		self.dialog_acts = dialog_acts
		self.max_hist_num_turns = max_hist_num_turns
		# prompts
		self.task_prompt = f"""
		The following is background information about task. 
		The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		The Persuadee can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and a Persuadee about a charity called Save the Children.
		{self.process_exp()}
		The following is a new conversation between another Persuader and Persuadee.
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()
		self.inference_args = inference_args
		return
	
	def process_exp(self):
		prompt_exps = ""
		for exp in self.conv_examples:
			prompt_exps += exp.to_string_rep(keep_user_da=True) + "\n"
		return prompt_exps.strip()
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		print("PersuadeeModel get_utterance")
		assert(state[-1][0] == PersuasionGame.SYS)
		prompt = f"""
		{self.task_prompt}
		{state.to_string_rep(keep_user_da=True, max_turn_to_display=self.max_hist_num_turns)}
		Persuadee:
		"""
		prompt = prompt.replace("\t", "").strip()
		# produce a response
		data = self.backbone_model.generate(prompt, **self.inference_args)
		user_resp = self.backbone_model._cleaned_resp(data, prompt)[0]
		return user_resp

	def get_utterance_w_da(self, state:DialogSession, action=None) -> "Tuple[str, str]":
		user_resp = self.get_utterance(state, action)
		# extract da
		start_idx = user_resp.find("[")
		end_idx = user_resp.find("]")
		if start_idx == -1 or end_idx == -1:
			da = PersuasionGame.U_Neutral
		else:
			da = user_resp[start_idx+1:end_idx]
			user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
			if da not in self.dialog_acts:
				da = PersuasionGame.U_Neutral
		return da, user_resp


class PersuadeeChatModel(PersuadeeModel):
	def __init__(self,
			dialog_acts: List[str],
			inference_args: dict,
			backbone_model:GenerationModel, 
			conv_examples: List[DialogSession] = [], 
			max_hist_num_turns=5):
		super().__init__(
			dialog_acts=dialog_acts,
			inference_args=inference_args,
			backbone_model=backbone_model,
			conv_examples=conv_examples,
			max_hist_num_turns=max_hist_num_turns
		)
		self.inference_args = inference_args
		self.task_prompt = f"""
		You are a persuadee. A Persuader is trying to persuade you to donate to a charity called Save the Children.
		You can choose amongst the following actions during a conversation to respond to the Persuader:
		{" ".join([f"[{da}]" for da in self.dialog_acts])}
		The following is an example conversation between a Persuader and some Persuadee.
		""".replace("\t", "").strip()
		self.new_task_prompt = "The following is a new conversation between a Persuader and a Persuadee (you). You may or may not want to donate to Save the Children."
		self.heuristic_args: dict = {
			"max_hist_num_turns": 2,
			"example_pred_turn": [[0, 2, 3, 4]]
		}
		self.prompt_examples = self.process_chat_exp()
		return
	
	def process_chat_exp(self):
		prompt_exps = []
		for exp in self.conv_examples:
			prompt_exps += self.__proccess_chat_exp(exp)
			prompt_exps.append({
				"role":"system", "content": self.new_task_prompt
			})
		return prompt_exps[:-1]

	def __proccess_chat_exp(self, exp:DialogSession, max_hist_num_turns: int = -1):
		if len(exp) == 0:
			return []
		# P4G dataset starts with the system
		assert(exp[0][0] == PersuasionGame.SYS)

		prompt_messages = []
		num_turns_to_truncate = 0
		if max_hist_num_turns > 0:
			num_turns_to_truncate = max(0, len(exp) // 2 - max_hist_num_turns)
		
		for i, (role, da, utt) in enumerate(exp):
			# truncate to reduce the size of the prompt
			if (i // 2) < num_turns_to_truncate:
				continue
			if role == PersuasionGame.SYS:
				prompt_messages.append({
					"role": "user",
					"content": f"{role}: {utt}".strip()
				})
			else:
				prompt_messages.append({
					"role": "assistant",  # assistant is the user simulator
					"content": f"{role}: [{da}] {utt}".strip()
				})
		return prompt_messages
	
	def get_utterance(self, state:DialogSession, action=None) -> str:
		print("PersuadeeChatModel get_utterance")
		assert(state[-1][0] == PersuasionGame.SYS)  # next turn is user's turn
		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.prompt_examples,
			{'role': 'system', 'content': self.new_task_prompt}
		]
		messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)

		# produce a response
		data = self.backbone_model.chat_generate(messages, **self.inference_args)
		user_resp = self.backbone_model._cleaned_chat_resp(
			data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
		)[0]
		return user_resp
	
	def get_utterance_from_batched_states(self, states:List[DialogSession], action=None) -> List[str]:
		assert(all([state[-1][0] == PersuasionGame.SYS for state in states]))
		all_prompts = []
		for state in states:
			messages = [
				{'role': 'system', 'content': self.task_prompt},
				*self.prompt_examples,
				{'role': 'system', 'content': self.new_task_prompt}
			]
			messages += self.__proccess_chat_exp(state, max_hist_num_turns=self.max_hist_num_turns)
			all_prompts.append(messages)
		# produce a response
		datas = self.backbone_model.chat_generate_batched(all_prompts, **self.inference_args)
		user_resps = []
		for data in datas:
			user_resp = self.backbone_model._cleaned_chat_resp(
				data, assistant_role=f"{PersuasionGame.USR}:", user_role=f"{PersuasionGame.SYS}:"
			)
			user_resps.append(user_resp[0])
		return user_resps
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		gen_user_resps = self.get_utterance_from_batched_states(states, action)
		das = []
		user_resps = []
		# extract da
		for user_resp in gen_user_resps:
			start_idx = user_resp.find("[")
			end_idx = user_resp.find("]")
			if start_idx == -1 or end_idx == -1:
				da = PersuasionGame.U_Neutral
			else:
				da = user_resp[start_idx+1:end_idx]
				user_resp = user_resp.replace(f"[{da}]", "", 1).strip()
				if da not in self.dialog_acts:
					da = PersuasionGame.U_Neutral
			das.append(da)
			user_resps.append(user_resp)
		return das, user_resps

	def __process_heuristics_chat_exp(self, dialog:DialogSession):
		if len(dialog) == 0:
			return []
		# assumes you start with the system
		# and ends with a user utterance to predict
		assert(dialog[0][0] == PersuasionGame.SYS)
		assert(dialog[-1][0] == PersuasionGame.USR)

		prompt_messages = []
		input_context = []
		answer_da = dialog[-1][1]
		for i, (role, da, utt) in enumerate(dialog):
			# if assistant is the Persuader, then current data is also Persuader -> then it is of role "system"
			# treat this as a task
			content = f"{role}: {utt}".strip()
			input_context.append(content)
		input_context.append(f"{dialog.USR} feeling:")

		prompt_q = "\n".join(input_context)
		prompt_messages.append({
			"role": 'user',
			"content": prompt_q
		})
		prompt_messages.append({
			"role": 'assistant',
			"content": f"{answer_da}"
		})
		return prompt_messages
	
	def __truncate_heuristics_dialog(self, dialog:DialogSession, pred_end_idx=-1):
		max_history_length = self.heuristic_args['max_hist_num_turns']
		if pred_end_idx == -1:
			pred_end_idx = len(dialog.history) - 1
		new_sys_start_idx = max(0, pred_end_idx - (max_history_length * 2 - 1))
		new_history = []
		for j, (role, da, utt) in enumerate(dialog):
			if j >= new_sys_start_idx:
				new_history.append((role, da, utt))
			if j == pred_end_idx:
				# user's utternace to predict
				break
		new_dialog_session = DialogSession(dialog.SYS, dialog.USR).from_history(new_history)
		return new_dialog_session
	
	def process_heurstics_chat_exp(self, new_task_prompt: str):
		prompt_exps = []
		for i, exp in enumerate(self.conv_examples):
			pred_end_turns: List[int] = self.heuristic_args['example_pred_turn'][i]
			# make a new dialogue session until that pred_idx with max max_history_length turns
			for pred_end_turn in pred_end_turns:
				pred_end_idx = pred_end_turn * 2 + 1
				new_dialog_session = self.__truncate_heuristics_dialog(exp, pred_end_idx)
				prompt_exps += self.__process_heuristics_chat_exp(new_dialog_session)
				prompt_exps.append({
					"role":"system", "content": new_task_prompt
				})
		return prompt_exps[:-1]

	def predict_da(self, state:DialogSession, never_end=True) -> str:
		# never_end=True  during real chat, let user choose to terminate, not this function
		# insert prop to donate, and compute the likelihood of user simulator agreeing to donate
		assert(state[-1][0] == PersuasionGame.USR)

		messages = [
			{'role': 'system', 'content': self.task_prompt},
			*self.process_heurstics_chat_exp(new_task_prompt=self.new_task_prompt),
			{'role': 'system', 'content': self.new_task_prompt}
		]
		new_dialog_session = self.__truncate_heuristics_dialog(state, -1)
		messages += self.__process_heuristics_chat_exp(new_dialog_session)[:-1]

		# majority vote, same as value function
		inf_args = {
			"max_new_tokens": 5,
			"temperature": 0.7,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 5,
		}
		datas = self.backbone_model.chat_generate(messages, **inf_args)
		# process into das
		sampled_das: list = []
		for resp in datas:
			user_da = resp['generated_text'].strip()
			if user_da not in self.dialog_acts:
				sampled_das.append(PersuasionGame.U_Neutral)
			if never_end:
				if user_da == PersuasionGame.U_Donate:
					sampled_das.append(PersuasionGame.U_PositiveReaction)
				elif user_da == PersuasionGame.U_NoDonation:
					sampled_das.append(PersuasionGame.U_NegativeReaction)
				else:
					sampled_das.append(user_da)
			else:
				sampled_das.append(user_da)
		logger.info(f"sampled das: {sampled_das}")
		# majority vote
		counted_das = Counter(sampled_das)
		user_da = counted_das.most_common(1)[0][0]
		return user_da


# ==========================================
#  Additions for CraigslistBargain (CB)
# ==========================================

class CBBuyerChatModel(PersuaderChatModel):
	def __init__(self,
			dialog_acts: List[str],
			backbone_model,
			max_hist_num_turns: int = 5,
			conv_examples: List = [],
			inference_args: dict = {}):
		
	   # [关键修复1] 使用关键字参数调用父类，防止位置参数错乱导致 dict 传给 int
		super().__init__(
			dialog_acts=dialog_acts, 
			backbone_model=backbone_model, 
			max_hist_num_turns=max_hist_num_turns, 
			conv_examples=conv_examples, 
			inference_args=inference_args
		)
		
		# =====================================================
		# [修正] 严格对应 TRIP 论文 Table 9 (Page 16)
		# =====================================================
		self.da_prompts_mapping = {
			# 1. Greetings
			"greetings": "Please say hello or chat randomly.",
			# 2. Ask a question
			"ask_question": "Please ask any question about product, year, price, usage, etc.",
			# 3. Answer a question
			"answer_question": "Please provide information about the product, year, usage, etc.",
			# 4. Propose the first price
			"propose_first_price": "Please initiate a price or a price range for the product.",
			# 5. Propose a counter price
			"propose_counter_price": "Please propose a new price or a new price range.",
			# 6. Use comparatives
			"use_comparatives": "Please propose a vague price by using comparatives with existing price.",
			# 7. Confirm information
			"confirm_information": "Please ask a question about the information to be confirmed.",
			# 8. Affirm confirmation
			"affirm_confirmation": "Please give an affirmative response to a confirm.",
			# 9. Deny confirmation
			"deny_confirmation": "Please give a negative response to a confirm.",
			# 10. Agree with the proposal
			"agree_proposal": "Please agree with the proposed price.",
			# 11. Disagree with a proposal
			"disagree_proposal": "Please disagree with the proposed price.",

			# --- [关键修复] 辅助 Keys ---
			# 这些 key 不会被 MCTS 搜索到（只要不放进 dialog_acts 列表），
			# 但必须存在，用于解析 gdpzero.py 中设置的 history dummy DA ("inform")
			"inform": "Please provide information or continue the conversation.",
			"quit": "Please end the conversation."
		}
		
		# 3. 过滤动作空间 (只保留 mapping 中存在的动作)
		self.dialog_acts = dialog_acts
		print(f"CB Buyer Acts: {self.dialog_acts}")
		# logger.debug(f"CB Buyer Acts: {self.dialog_acts}")

		# 4. 初始化 task_prompt
		# 注意：P4G 在这里直接写死了 Prompt，但 CB 的商品信息还没进来。
		# 所以我们先定义一个基础模板，或者留空，等待 set_item_info 被调用时再填充。
		self.base_instruction = "Now enter the role-playing mode. In the following conversation, you will play as a buyer in a price bargaining game."
		self.task_prompt = "" # 暂时为空，等待 injected item info

		# 设置推理参数 (参照 P4G)
		self.inference_args = {
			"max_new_tokens": 64, # CB 回复一般较短
			"temperature": 0.7,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"return_full_text": False,
			**inference_args
		}

	def set_item_info(self, item_info):
		"""
		这是 CB 特有的方法。
		因为每个 Dialog 的商品不一样，所以必须在 gdpzero.py 循环里调用这个方法，
		动态更新 self.task_prompt。
		"""
		title = item_info.get('title', 'item')
		price = item_info.get('price', 'unknown')
		desc = item_info.get('description', '')

		# 构造类似 P4G 的 task_prompt，但是带入了商品信息
		# 参考 TRIP 论文 Table 19 的格式
		self.task_prompt = f"""
		{self.base_instruction}
		You are the buyer who is trying to buy the {title} with the listing price of {price}.
		Product description: {desc}
		Please reply with only one short and succinct sentence.
		
		The following is the conversation history:
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()

	def _get_prompt(self, context, history):
		"""
		重写获取 Prompt 的逻辑。
		P4G 的父类通常会把 self.task_prompt 和 context 拼起来。
		"""
		# 确保 task_prompt 已经被 set_item_info 设置过了
		if not self.task_prompt:
			logger.warning("Warning: Item info not set for CBBuyerChatModel!")
		
		# 拼接 Prompt：任务描述 + 对话历史 + "Buyer:"
		# 注意：这里的 context 已经是处理过的对话历史字符串
		full_prompt = f"{self.task_prompt}\n{context}\nBuyer: "
		return full_prompt
	


class CBSellerChatModel(PersuadeeChatModel):
	def __init__(self,
			dialog_acts: List[str],
			backbone_model,
			max_hist_num_turns: int = 5,
			conv_examples: List = [],
			inference_args: dict = {}):
		
		# [核心修复] 必须显式指定父类参数名！
		# PersuadeeChatModel 继承自 PersuadeeModel
		# PersuadeeModel.__init__(self, dialog_acts, inference_args, backbone_model, conv_examples, max_hist_num_turns)
		# 注意：原版 PersuadeeModel 的参数顺序非常乱！必须用 keyword arguments！
		
		super().__init__(
			dialog_acts=dialog_acts, 
			backbone_model=backbone_model, 
			max_hist_num_turns=max_hist_num_turns, 
			conv_examples=conv_examples, 
			inference_args=inference_args
		)
		
		# Seller 的动作通常比较简单，或者直接用 generic mapping
		self.da_prompts_mapping = {
			"greeting": "The Seller greets the buyer.",
			"inform": "The Seller answers questions or provides info.",
			"ask_price": "The Seller asks for a price.",
			"agree_price": "The Seller agrees to the price.",
			"disagree_price": "The Seller rejects the price.",
			"counter_price": "The Seller offers a new price.",
			"quit": "The Seller ends the chat.",
			# 必须包含所有传入的 user_da，否则会被过滤掉
			"propose_price": "The Seller proposes a price.", 
			"ask_info": "The Seller asks for info." 
		}
		
		self.dialog_acts = [da for da in dialog_acts if da in self.da_prompts_mapping]
		self.task_prompt = "" 

		self.inference_args = {
			"max_new_tokens": 64,
			"temperature": 1.0,
			**inference_args
		}

	def set_item_info(self, item_info):
		title = item_info.get('title', 'item')
		price = item_info.get('price', 'unknown')
		
		# 参考 TRIP 论文 Table 13  # 但是这个地方没有加persona
		self.task_prompt = f"""
		Now enter the role-playing mode. In the following conversation, you will play as a seller in a price bargaining game.
		You are the seller who is selling the {title} for {price}.
		Your goal is to sell the item at a good price. Do not easily agree to low offers.
		Please reply with only one short and succinct sentence.
		
		The following is the conversation history:
		"""
		self.task_prompt = self.task_prompt.replace("\t", "").strip()

	def _get_prompt(self, context, history):
		return f"{self.task_prompt}\n{context}\nSeller: "


# GDP-Zero/core/players.py (添加到文件末尾)
## ===== 柏拉图 API 配置 =====
import os
import requests
from typing import Dict, List, Optional
PLATO_URL = "https://api.bltcy.ai/v1/chat/completions"
PLATO_MODEL = "gpt-3.5-turbo-0125"
PLATO_API_KEY = os.environ.get("PLATO_API_KEY", "sk-Teb2mDnGww8tGY96NfxNuIBERw6lzuP1F9zYZkvmPJ5XhsSn")
if not PLATO_API_KEY:
    raise RuntimeError("请先在环境变量中设置 PLATO_API_KEY，用于访问 https://api.plato.ai 的接口。")

def call_plato_api(messages: List[Dict[str, str]], model: str = PLATO_MODEL,
                              temperature: float = 0.5, max_tokens: int = 200) -> str:
    """
    调用 Plato API
    """
    print("call_plato_api")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {PLATO_API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(PLATO_URL, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"Plato API call failed: {e}")
        raise

class CBSystemPlanner(P4GChatSystemPlanner):
	"""
	CraigslistBargain (CB) 专用规划器。
	重写 predict 和 heuristic，适配 Zero-shot MCTS。
	"""
	def __init__(self, 
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns, 
			generation_model: GenerationModel, 
			conv_examples: List[DialogSession] = []) -> None:
		
		super().__init__(
			dialog_acts, max_hist_num_turns,
			user_dialog_acts, user_max_hist_num_turns,
			generation_model, conv_examples
		)
		
		# 初始化 item_info
		self.item_info = {'title': 'item', 'price': 'unknown'}
		self.dialog_acts = dialog_acts
		
		# [关键] 针对 Predict 的推理参数
		# 既然是 MCTS Prior，我们需要采样多次 (n=10~15) 来形成概率分布
		self.inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.0,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 15,
		}

	def set_item_info(self, item_info):
		self.item_info = item_info

	def _get_generated_da(self, data) -> list:
		"""
		解析器：从模型的输出中提取 DA。
		模型预期输出： "ask_price]" 或者 "ask_price" (因为我们在 prompt 里可能预置了 '[')
		"""
		pred_da = []
		for resp in data:
			text = resp['generated_text'].strip()
			
			# 简单的清洗：去掉可能存在的括号
			clean_text = text.replace("[", "").replace("]", "").strip()
			
			# 匹配最长的前缀策略 (防止 subtring 匹配错误)
			found = None
			for da in self.dialog_acts:
				if da == clean_text: # 优先精确匹配
					found = da
					break
			
			if not found:
				# 模糊匹配尝试
				for da in self.dialog_acts:
					if da in clean_text:
						found = da
						break
			
			if found:
				pred_da.append(found)
			else:
				# 如果没解析出来，记录一个 fallback (比如 inform) 或者忽略
				# print(f"  [Warn] Failed to parse act from: {text}")
				pass
				
		return pred_da

	def predict(self, state: DialogSession) -> "Tuple[np.ndarray, float]":
		"""
		[重写核心] MCTS 扩展节点时调用。
		目标：计算当前状态下，各个动作的先验概率 P(a|s)。
		方法：让 LLM 采样生成 10 次，统计各个动作出现的频率。
		"""
		title = self.item_info.get('title', 'item')
		price = self.item_info.get('price', 'unknown')
		
		# 1. 过滤掉不想让模型选的动作 (Masking)
		# 注意：这里仅仅是 Prompt 展示层面的过滤，实际上 calculate prob 时还是基于全集 self.dialog_acts
		display_acts = [da for da in self.dialog_acts if da not in ['inform', 'quit']]
		valid_acts_str = ", ".join([f"[{da}]" for da in display_acts])

		# 2. 构建纯粹的“策略选择” Prompt
		# 我们不让模型生成对话，只让它做选择题
		sys_prompt = f"""
		You are a negotiation expert assisting a Buyer.
		Item: {title}. Listing Price: {price}.
		
		Analyze the conversation history and select the SINGLE best strategic move from the list below.
		
		Valid Strategies: {valid_acts_str}
		
		Output ONLY the strategy tag enclosed in brackets (e.g., ask_question). Do not write any dialogue content.
		""".strip()

		# 处理历史
		hist_str = state.to_string_rep(keep_sys_da=True, keep_user_da=False)
		hist_str = hist_str.replace("Persuader", "Buyer").replace("Persuadee", "Seller")

		messages = [
			{'role': 'system', 'content': sys_prompt},
			{'role': 'user', 'content': f"Conversation History:\n{hist_str}\n\nDetermine the Next Buyer Strategy. Output format: strategy_name\nResponse: "}
		]

		# 3. 调用生成
		# 注意：因为我们在 prompt 最后加了 "["，模型可能会补全 "ask_price]"。
		# 或者是输出完整的 "[ask_price]"，这取决于模型对 "Response: [" 的理解。
		# 最稳妥的方式是不预填 "["，直接问 "Response:"，然后在 _get_generated_da 里处理
		
		# 修正 Prompt 策略：不玩预填的花活，直接明确指令
		messages[-1] = {'role': 'user', 'content': "Determine the Next Buyer Strategy. Output ONLY the bracketed tag.\nResponse:"}

		print("MCTS的predict过程中传入chat_generate的messages是：", messages)
		data = self.generation_model.chat_generate(messages, **self.inf_args)
		# data = call_plato_api(messages)
		print("MCTS的predict过程中chat_generate的输出data是：", data)
		
		# 4. 解析结果
		sampled_das = self._get_generated_da(data)
		
		# 调试日志
		# if len(sampled_das) == 0:
		#	 print(f"  [Predict Debug] Raw outputs: {[d['generated_text'] for d in data]}")

		# 5. 计算概率分布
		prob = np.zeros(len(self.dialog_acts))
		prob += self.smoothing # 平滑，防止 0 概率
		
		for da in sampled_das:
			if da in self.dialog_acts:
				prob[self.dialog_acts.index(da)] += 1.0
		
		# 归一化
		prob /= prob.sum()
		
		# 6. 计算价值 (Heuristic)
		v = self.heuristic(state)

		return prob, v

	def get_valid_moves(self, state):
		"""屏蔽 inform/quit 等动作"""
		mask = np.ones(len(self.dialog_acts))
		forbidden_acts = ["inform", "quit"] # 根据需要调整
		for i, da in enumerate(self.dialog_acts):
			if da in forbidden_acts:
				mask[i] = 0.0

		return mask

	def heuristic(self, state: DialogSession) -> float:
		# ... (保持你现在的 Heuristic 代码不变，记得用你刚才修正过的 inf_args) ...
		# ... 也就是 max_new_tokens=10, temperature=0.0 (或低温), repetition_penalty=1.0, stop=...
		# 这里为了完整性简写了
		
		title = self.item_info.get('title', 'item')
		price = self.item_info.get('price', 'unknown')
		
		user_task_prompt = f"""
		You are an expert negotiation judge. Item: {title}, Price: {price}.
		Evaluate likelihood of a deal: DEFINITELY_YES, LIKELY_YES, NEUTRAL, LIKELY_NO, DEFINITELY_NO.
		Output ONLY the label.
		"""
		
		hist_str = state.to_string_rep(keep_sys_da=True, keep_user_da=False)
		hist_str = hist_str.replace("Persuader", "Buyer").replace("Persuadee", "Seller")
		
		messages = [
			{'role': 'system', 'content': user_task_prompt},
			{'role': 'user', 'content': f"History:\n{hist_str}\n\nLikelihood Label:"}
		]

		inf_args = {
			"max_new_tokens": 12,
			"temperature": 1.1,
			"return_full_text": False,
			"do_sample": True,
			"num_return_sequences": 10,
		}

		try:
			print("Heuristic过程中传入chat_generate的messages是：", messages)
			data = self.generation_model.chat_generate(messages, **inf_args)
			print("Heuristic过程中chat_generate的输出data是：", data)
		except Exception as e:
			print(f"Heuristic过程中chat_generate调用失败: {e}")
			import traceback
			print("详细错误信息:")
			traceback.print_exc()
			return 0.0

		# ... (解析逻辑保持不变) ...
		score_sum = 0.0
		valid_count = 0
		score_map = {
			"DEFINITELY_YES": 1.0, 
			"LIKELY_YES": 0.5, 
			"NEUTRAL": 0.0,
			"LIKELY_NO": -0.5, 
			"DEFINITELY_NO": -1.0
		}
		
		for resp in data:
			text = resp['generated_text'].strip().upper()
			matched_score = None
			if text in score_map:
				matched_score = score_map[text]
			else:
				for label, score in score_map.items():
					if label in text:
						matched_score = score
						break
			
			if matched_score is not None:
				score_sum += matched_score
				valid_count += 1
			else:
				score_sum += 0.0 # 无法识别按中立处理
				valid_count += 1
				
		return float(score_sum / valid_count) if valid_count > 0 else 0.0