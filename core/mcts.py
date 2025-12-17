import numpy as np
import logging
import math

from core.helpers import DialogSession
from core.game import DialogGame
from core.players import DialogPlanner


logger = logging.getLogger(__name__)


class MCTS():
	def __init__(self, game:DialogGame, player:DialogPlanner, configs) -> None:
		self.game = game
		self.player = player
		self.configs = configs
		# U(s,a) = Q(s,a) + c * P(s,a) * (\sqrt{ \sum_{a'} N(s,a')}) / (1+N(s,a))
		self.Ns: dict = {}  # saves compute
		self.Nsa: dict = {}
		self.Q: dict = {}
		self.P: dict = {}
		# utility
		self.valid_moves: dict = {}
		self.terminals: dict = {}
		# debugging / more information
		self.Vs: dict = {}
		return

	def _to_string_rep(self, state:DialogSession):
		# for tree search, keep all dialog turns
		return state.to_string_rep(keep_sys_da=True, keep_user_da=True, max_turn_to_display=-1)

	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}

		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v

	def search(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		
		is_leaf_node = False
		v = 0.0
		if hashable_state not in self.terminals:
			# selected leaf node, expand
			self.terminals[hashable_state] = self.game.get_dialog_ended(state)
			v = self._init_node(state)
			is_leaf_node = True
		# if this leaf node is terminal, return the value
		if self.terminals[hashable_state] > 0:
			# terminal node
			logger.debug("ended")
			return self.terminals[hashable_state]
		# otherwise, return v
		if is_leaf_node:
			return v
		
		# existing, continue selection
		# go next state by picking best according to U(s,a)
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a
		# transition
		next_state = self.game.get_next_state(state, best_action)
		
		# 1. if not leaf, continue traversing, and state=s will get the value from the leaf node
		# 2. if leaf, we will expand it and return the value for backpropagation
		v = self.search(next_state)

		# update stats
		# add in new estimate and average
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1
		
		# now we are single player, hence just v instead of -v
		return v

	def get_action_prob(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.Ns:
			# selected leaf node, expand
			logging.warn("querying a state that has not been visited")
			self._init_node(state)
		# get the counts for all moves
		# convert to prob
		prob = np.zeros(self.player.get_valid_moves(state).shape)
		for a in self.valid_moves[hashable_state]:
			prob[a] = self.Nsa[hashable_state][a]
		prob /= prob.sum()
		return prob


class OpenLoopMCTS(MCTS):
	def __init__(self, game, player, configs) -> None:
		# 初始化：继承基础MCTS，并初始化“开放式”MCTS所需的数据结构
		super().__init__(game, player, configs)
		self.realizations: dict = {}        # 记录：每个状态对应的已采样到的真实对话轨迹列表 state -> [DialogSession, ...]
		self.realizations_Vs: dict = {}     # 记录：每个状态下，不同生成的系统回复utterance的平均价值 state -> {utt: V}
		self.realizations_Ns: dict = {}     # 记录：每个utterance被采样多少次 state -> {utt: N}
		self.max_realizations = configs.max_realizations  # 每个状态最多缓存多少条真实轨迹
		return

	def _to_string_rep(self, state:DialogSession):
		# 将状态转换为可哈希字符串，只保留系统端的DA序列，用于open-loop MCTS的状态表示
		das = []
		for (speaker, da, _) in state:
			if speaker == state.SYS:
				das.append(da)
		return "__".join(das)

	def _init_node(self, state:DialogSession):
		# 初始化一个新的叶子节点，包括P、Q、N等统计，并调用player.predict评估先验和节点价值
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		# 初始化统计量
		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		self.realizations[hashable_state] = [state.copy()]  # 保存此状态的第一条对话轨迹

		# 调用LLM预测得到：先验策略prior & 当前节点价值v
		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # 仅用于调试
		self.P[hashable_state] = prior * allowed_actions

		# 若无有效prior则均匀分配，否则归一化
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v

	def _sample_realization(self, hashable_state):
		# 从该状态下缓存的多个真实轨迹中随机采样一条，用于open-loop模拟
		rand_i = np.random.randint(len(self.realizations[hashable_state]))
		return self.realizations[hashable_state][rand_i]

	def _add_new_realizations(self, state):
		# 若此真实轨迹未被记录，则加入缓存；并保持缓存数量不超过上限
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = []
		if state in self.realizations[hashable_state]:
			return
		
		self.realizations[hashable_state].append(state.copy())
		if len(self.realizations[hashable_state]) > self.max_realizations:
			logger.warning(f"len(self.realizations[hashable_state])={len(self.realizations[hashable_state])}")
			self.realizations[hashable_state].pop(0)
		return

	def _get_next_state(self, state, best_action):
		# 根据“当前状态+动作”尝试从缓存中获取后继状态；若数量已满，则直接复用（不再调用LLM生成）
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# 使用缓存的真实轨迹，减少模型调用次数
			return self._sample_realization(prefetch_state)
		
		# 否则需要通过game环境生成新的后继状态（会触发LLM生成）
		next_state = self.game.get_next_state(state, best_action)
		return next_state
	
	def _update_realizations_Vs(self, state: DialogSession, v: float):
		# 回传更新：更新特定状态下某个系统utterance的平均价值
		hashable_state = self._to_string_rep(state)
		if hashable_state not in self.realizations_Vs:
			self.realizations_Vs[hashable_state] = {}
			self.realizations_Ns[hashable_state] = {}
		sys_utt = state.get_turn_utt(
			turn=-1,
			role=state.SYS,
		)
		if sys_utt not in self.realizations_Vs[hashable_state]:
			self.realizations_Vs[hashable_state][sys_utt] = 0
			self.realizations_Ns[hashable_state][sys_utt] = 0

		# 用增量平均更新utterance价值
		self.realizations_Ns[hashable_state][sys_utt] += 1
		self.realizations_Vs[hashable_state][sys_utt] += (v - self.realizations_Vs[hashable_state][sys_utt]) / self.realizations_Ns[hashable_state][sys_utt]
		return

	def search(self, state:DialogSession):
		# MCTS核心递归：选择→扩展→模拟→回传（open-loop版本）
		hashable_state = self._to_string_rep(state)
		
		# 若对话已结束，直接返回终局价值
		terminated_v = self.game.get_dialog_ended(state)
		if terminated_v == 1.0:
			logger.debug("ended")
			return terminated_v
		
		# 若为叶子节点（从未访问过），扩展并返回初始价值
		if hashable_state not in self.P:
			v = self._init_node(state)
			return v
		else:
			# 已访问过：则将此新轨迹加入对应状态的缓存中
			self._add_new_realizations(state)
		
		# 选择阶段：按PUCT公式选择UCT最大的动作
		best_uct = -float('inf')
		best_action = -1
		for a in self.valid_moves[hashable_state]:
			Ns = self.Ns[hashable_state]
			if Ns == 0:
				Ns = 1e-8
			uct = self.Q[hashable_state][a] + self.configs.cpuct * self.P[hashable_state][a] * math.sqrt(Ns) / (1 + self.Nsa[hashable_state][a])
			if uct > best_uct:
				best_uct = uct
				best_action = a

		# open-loop特性：先从缓存中随机抽出一个真实轨迹作为起点
		state = self._sample_realization(hashable_state)
		# 再基于动作获得下一个状态
		next_state = self._get_next_state(state, best_action)
		
		# 递归向下搜索
		v = self.search(next_state)

		# 回传更新：更新Q/Ns/Nsa统计
		self.Q[hashable_state][best_action] = (self.Nsa[hashable_state][best_action] * self.Q[hashable_state][best_action] + v) / (self.Nsa[hashable_state][best_action] + 1)
		self.Ns[hashable_state] += 1
		self.Nsa[hashable_state][best_action] += 1

		# 同时更新该后继状态下不同utterance的价值统计（方便最终输出最优生成文本）
		self._update_realizations_Vs(next_state, v)
		return v
	
	def get_best_realization(self, state:DialogSession, action: int):
		# 在给定状态+动作下，从已采样的utterance中挑选价值最高的一句自然语言回复
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[action]
		if prefetch_state not in self.realizations_Vs:
			raise Exception("querying a state that has no realizations sampled before")
		
		curr_best_v = -float('inf')
		curr_best_realization = None
		for sys_utt, v in self.realizations_Vs[prefetch_state].items():
			if v > curr_best_v:
				curr_best_v = v
				curr_best_realization = sys_utt
		return curr_best_realization

	

class OpenLoopMCTSParallel(OpenLoopMCTS):
	def __init__(self, game, player, configs) -> None:
		super().__init__(game, player, configs)

	def _populate_next_realizations(self, state, next_action, num_to_add):
		next_states = self.game.get_next_state_batched(state, next_action, batch=num_to_add)
		for next_state in next_states:
			self._add_new_realizations(next_state)
		return

	def _get_next_state(self, state, best_action):
		prefetch_state = self._to_string_rep(state) + "__" + self.player.dialog_acts[best_action]
		if prefetch_state in self.realizations and len(self.realizations[prefetch_state]) == self.max_realizations:
			# use the cached realization
			return self._sample_realization(prefetch_state)

		self._populate_next_realizations(state, best_action, self.max_realizations)
		return self._sample_realization(prefetch_state)
	
	def _init_node(self, state:DialogSession):
		hashable_state = self._to_string_rep(state)
		allowed_actions = self.player.get_valid_moves(state)
		self.valid_moves[hashable_state] = allowed_actions.nonzero()[0]

		self.Ns[hashable_state] = 0
		self.Nsa[hashable_state] = {action: 0 for action in self.valid_moves[hashable_state]}
		self.Q[hashable_state] = {action: self.configs.Q_0 for action in self.valid_moves[hashable_state]}
		# should have been initialized during _get_next_state, except for the root node
		if hashable_state not in self.realizations:
			self.realizations[hashable_state] = [state.copy()]

		# TODO: batch predict value function
		prior, v = self.player.predict(state)
		self.Vs[state.to_string_rep(keep_sys_da=True, keep_user_da=True)] = v  # for debugging
		self.P[hashable_state] = prior * allowed_actions
		# renormalize
		if np.sum(self.P[hashable_state]) == 0:
			self.P[hashable_state] = allowed_actions / np.sum(allowed_actions)
			logger.warning("This should never happen")
		else:
			self.P[hashable_state] /= np.sum(self.P[hashable_state])
		return v