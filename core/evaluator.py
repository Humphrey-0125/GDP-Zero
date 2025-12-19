import logging
import random

from abc import ABC, abstractmethod
from typing import List
from core.gen_models import GenerationModel


logger = logging.getLogger(__name__)


class RespRanker(ABC):
	@abstractmethod
	def evaluate(self, context, resp_a, resp_b):
		"""
		Compare two responses and return the preference.
		"""
		raise NotImplementedError


class P4GEvaluator(RespRanker):
	def __init__(self, gen_model: GenerationModel):
		super().__init__()
		self.gen_model = gen_model
		self.inference_args = {
			"max_tokens": 2,
			"temperature": 0.7,
			"echo": False,
			"n": 5,
			"stop": ""
		}
	
	def evaluate(self, context, resp_a, resp_b):
		do_swap = False
		if random.random() < 0.5:
			do_swap = True
			resp_a, resp_b = resp_b, resp_a
		# prompt = f"""
		# The following is background information about Save the Children.
		# Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
		# The following is a conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
		# {context}
		# Which of the following responses can better help the Persuader convince the Persuadee to donate to Save the Children?
		# A. Persuader: {resp_a}
		# B. Persuader: {resp_b}
		# C. Can't tell.
		# Your can choose from either A, B, or C.
		# Your choice:
		# """.replace('\t', '').strip()
		prompt = f"""
        The following is background information about Save the Children.
        Save the Children is head-quartered in London, and they work to help fight poverty around the world. Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        The following is a conversation between a Persuader and a Persuadee about a charity called Save the Children. The Persuader is trying to persuade the Persuadee to donate to Save the Children.
        {context}
        Which of the following responses can better help the Persuader convince the Persuadee to donate to Save the Children?
        A. Persuader: {resp_a}
        B. Persuader: {resp_b}
        C. Can't tell.
        Your can choose from either A, B, or C.
        Please output ONLY the single letter (A, B, or C) without explanation.
        Your choice:
        """.replace('\t', '').strip()
		logger.debug(f"prompt: {prompt}")
		resps = self.gen_model.generate(prompt, **self.inference_args)
		choices, rationales = self._process_resps(resps)
		preference = self._majority_vote(choices, do_swap)
		return preference, {'choices': choices, 'rationales': rationales, 'do_swap': do_swap}

	def _process_resps(self, resps:List[dict]):
		choices = []
		rationales = []
		for resp in resps:
			gen = resp['generated_text'].strip()
			
			if len(gen) == 0:
				print("Empty response")
				choice = 'c'
			else:
				choice = gen[0].lower()
			
			if choice not in ['a', 'b', 'c']:
				print(f"Invalid choice: {choice}")
				choice = 'c'
			choices.append(choice)
			# see if there is a rationale  # just dump the entire response
			rationale = gen
			rationales.append(rationale)
		return choices, rationales

	def _majority_vote(self, resps:List[str], do_swap=False):
		# if there is a majority vote between A=0 and B=1, return the majority vote
		# otherwise, return C=2
		a_cnt = 0
		b_cnt = 0
		for resp in resps:
			if resp == 'a':
				a_cnt += 1
			elif resp == 'b':
				b_cnt += 1
		if a_cnt > b_cnt:
			return 0 if not do_swap else 1
		elif b_cnt > a_cnt:
			return 1 if not do_swap else 0
		return 2
	


# 这个地方参数设置应该看一看！！！！！！！！！！！！！！！！！！！！！！！
class CBEvaluator(RespRanker):
    def __init__(self, gen_model: GenerationModel):
        super().__init__()
        self.gen_model = gen_model
        # 评估时的参数：温度设低一点，让它更理性
        self.inference_args = {
            "max_tokens": 5, 
            "temperature": 0.0, 
            "echo": False,
            "n": 1, 
            "stop": ["\n"] 
        }
    
    def evaluate(self, context, resp_a, resp_b, item_info=None):
        """
        对比两个砍价回复，判断哪个更好。
        新增 item_info 参数，用于在评估时告知裁判商品信息。
        """
        do_swap = False
        # 随机交换 A/B 位置，防止模型总是偏爱选项 A
        if random.random() < 0.5:
            do_swap = True
            resp_a, resp_b = resp_b, resp_a
            
        # 构造商品背景信息
        if item_info:
            title = item_info.get('title', 'item')
            price = item_info.get('price', 'unknown price')
            desc = item_info.get('description', '')
            bg_info = f"Item: {title}\nListing Price: {price}\nDescription: {desc}"
        else:
            bg_info = "Item info is missing."

        # 专门针对 CB 的评估 Prompt
        # 这里的核心是把“劝捐”改成“以更低价格买到商品”
        prompt = f"""
        You are an expert in negotiation and bargaining psychology.
        
        [Product Information]
        {bg_info}
        
        [Conversation Context]
        The following is a conversation between a Buyer and a Seller. The Buyer is trying to purchase the item at a lower price.
        {context}
        
        [Task]
        Which of the following responses is more effective for the Buyer to negotiate a better deal (lower price) while maintaining a good conversation flow?
        A. Buyer: {resp_a}
        B. Buyer: {resp_b}
        C. Can't tell / Both are equal.
        
        [Instruction]
        - Choose A if response A is more strategic, polite yet firm, or logically persuasive.
        - Choose B if response B is better.
        - Choose C if both are similar or neither makes sense.
        - Output ONLY the single letter (A, B, or C) without explanation.
        
        Answer:
        """.replace('\t', '').strip()
        
        # logger.debug(f"prompt: {prompt}")
        
        # 调用模型生成评价
        resps = self.gen_model.generate(prompt, **self.inference_args)
        choices, rationales = self._process_resps(resps)
        
        # 投票逻辑
        preference = self._majority_vote(choices, do_swap)
        
        return preference, {'choices': choices, 'rationales': rationales, 'do_swap': do_swap}

    def _process_resps(self, resps:List[dict]):
        choices = []
        rationales = []
        for resp in resps:
            gen = resp['generated_text'].strip()
            
            # 更鲁棒的解析逻辑
            choice = 'c'
            if len(gen) > 0:
                first_char = gen[0].lower()
                if first_char in ['a', 'b', 'c']:
                    choice = first_char
                # 处理模型可能输出 "Option A" 的情况
                elif 'a' in gen.lower()[:10]: choice = 'a'
                elif 'b' in gen.lower()[:10]: choice = 'b'
            
            choices.append(choice)
            rationales.append(gen)
        return choices, rationales

    def _majority_vote(self, resps:List[str], do_swap=False):
        # 简化版投票，通常 n=1 时直接返回
        if not resps: return 2
        
        c = resps[0]
        if c == 'a': return 0 if not do_swap else 1
        if c == 'b': return 1 if not do_swap else 0
        return 2