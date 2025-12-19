import argparse
import pickle
import logging
import os

from tqdm.auto import tqdm
from core.evaluator import P4GEvaluator, CBEvaluator
from core.gen_models import OpenAIModel, OpenAIChatModel, AzureOpenAIModel, AzureOpenAIChatModel


logger = logging.getLogger(__name__)


def main(args):
	if args.debug:
		logging.basicConfig(level=logging.DEBUG)
		logger.setLevel(logging.DEBUG)
	
	# 初始化裁判模型 (Judge Model)
	if args.judge in ['gpt-3.5-turbo']:
		backbone_model = OpenAIChatModel(args.judge)
	elif args.judge == 'chatgpt':
		backbone_model = AzureOpenAIChatModel(args.judge)
	else:
		raise ValueError(f"unknown judge: {args.judge}")
	
	
	# [修改点 2] 根据 dataset 参数选择评估器
	if args.dataset == 'cb':
		print("Using CBEvaluator for CraigslistBargain...")
		evaluator = CBEvaluator(backbone_model)
	else:
		print("Using P4GEvaluator for PersuasionForGood...")
		evaluator = P4GEvaluator(backbone_model)

	with open(args.f, 'rb') as f:
		data: list = pickle.load(f)
	h2h_data = []
	if args.h2h:
		with open(args.h2h, 'rb') as f:
			h2h_data: list = pickle.load(f)
		assert(len(data) == len(h2h_data))
		assert(args.output != '')  # specify output path when doing h2h comparisons

	result = []
	stats = {
		'win': 0,  # if b=new_resp is better than a=ori_resp
		'draw': 0,
		'lose': 0,
	}

	# 开始遍历评估
	for i, d in tqdm(enumerate(data[:]), total=len(data), desc="evaluating"):
		context = d['context']
		ori_resp = d['ori_resp']
		new_resp = d['new_resp']
		# [修改点 3] 获取 CB 专属的商品信息 (P4G 没有这个字段，get 会返回默认 {})
		item_info = d.get('item_info', {}) if args.dataset == 'cb' else None
		if len(h2h_data) > 0:
			ori_resp = h2h_data[i]['new_resp']
		
		# [修改点 4] 调用 evaluate 时传入 item_info
        # 注意：P4GEvaluator.evaluate 定义里只有 (context, resp_a, resp_b)
        # 而 CBEvaluator.evaluate 定义里有 (context, resp_a, resp_b, item_info=None)
        # 为了兼容，我们需要分情况调用，或者确认 P4GEvaluator 也能接收 **kwargs
        
		if args.dataset == 'cb':
			winner, info = evaluator.evaluate(context, ori_resp, new_resp, item_info=item_info)
		else:
			winner, info = evaluator.evaluate(context, ori_resp, new_resp)

		# update winners
		if winner == 0:
			stats['lose'] += 1
		elif winner == 1:
			stats['win'] += 1
		else:
			stats['draw'] += 1
		
		info['winner'] = winner
		result.append(info)

	# save
	if args.output != '':
		output_file = args.output
	else:
		output_folder = os.path.join(os.path.dirname(args.f), 'evaluation')
		output_filename = os.path.basename(args.f).replace('.pkl', '_evaluated.pkl')
		output_file = os.path.join(output_folder, output_filename)
	with open(output_file, 'wb') as f:
		pickle.dump(result, f)

	# statistics
	win_rate = stats['win'] / sum(stats.values())
	print(f"win rate: {win_rate*100.0:.2f}%")
	print("stats: ", stats)
	return


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default="p4g", choices=["p4g", "cb"], help='Dataset to use')
	parser.add_argument('-f', type=str, help='path to the data file for comparing against human in p4g. See P4GEvaluator documentation to see the format of the file.')
	parser.add_argument('--judge', type=str, default='gpt-3.5-turbo', help='which judge to use.', choices=['gpt-3.5-turbo', 'chatgpt'])
	parser.add_argument('--h2h', type=str, default='', help='path to the data file for head to head comparison. If empty compare against human in p4g.')
	parser.add_argument("--output", type=str, default='', help="output file")
	parser.add_argument("--debug", action='store_true', help="debug mode")
	args = parser.parse_args()

	main(args)