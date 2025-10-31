import yaml
import argparse
import sys
from test_single import Tester
from trainer import Trainer
import tools


# sys.argv = ['main.py', '--hyper_parameter_path', './hyper_parameters.yaml']  # 在IDE中运行时需要，模拟命令行参数  python main.py --hyper_parameter_path ./hyper_parameters.yaml
parser = argparse.ArgumentParser()
parser.add_argument('--hyper_parameter_path', type=str, required=True, help="超参数文件路径")
config = parser.parse_args()

# 读取设置文件
with open(config.hyper_parameter_path, 'r') as f:
    hyper = yaml.safe_load(f)
tools.set_seed(hyper['TRAIN']['SEED'])

SCRATCH = True  # 是否为首次训练
if SCRATCH:  # 首次训练
    trainer = Trainer(hyper)
    trainer.save_hyper(hyper_file_path=config.hyper_parameter_path)
    MODEL_PATH = trainer.train()
    auto_test = Tester(log_path=MODEL_PATH, config_file_name=config.hyper_parameter_path)  # 自动完成测试
    auto_test.test()
else:  # 接续训练
    trainer = Trainer(hyper)
    trainer.load(check_path=None)  # 加载路径
    MODEL_PATH = trainer.train()
    auto_test = Tester(log_path=MODEL_PATH, config_file_name=config.hyper_parameter_path)  # 自动完成测试
    auto_test.test()

