import torch
import os
import argparse
import config as cnf


def writeToFile(filename, line):
	with open(filename, 'a') as file:
		file.write(line)


def parameterNorm(parameters, p):
	vec = []
	for param in parameters:
		if p == 'grad':
			vec.append(param.grad.view(-1))
		elif p == 'weight':
			vec.append(param.view(-1))
	param_norm = torch.cat(vec).norm(2)
	return param_norm


def getArgumentParser():
	parser = argparse.ArgumentParser(description='Train network')
	parser.add_argument('-f', '--model-file', required=True, help='used to set different model file name other than default one')
	parser.add_argument('-m', '--multi-gpu', action='store_true', help='use multiple gpus?')
	parser.add_argument('--full-train-set', action='store_true', help='use entire training set or split into train val set')
	args = parser.parse_args()

	return args