from nni.experiment import Experiment
experiment = Experiment('local')

search_space = {
    'learning_rate': {'_type': 'uniform', '_value': [0.00001, 0.01]},
    'momentum': {'_type': 'uniform', '_value': [0.8, 0.99]},
    # 'e_layers': {'_type': 'quniform', '_value': [1, 10, 1]},
    # 'n_heads': {'_type': 'quniform', '_value': [1, 10, 1]},
    # 'x': {'_type': 'quniform', '_value': [1, 10, 1]},
    # 'd_ff': {'_type': 'quniform', '_value': [1, 1000, 1]},
    # 'dropout': {'_type': 'uniform', '_value': [0.01, 0.999]},
    # 'fc_dropout': {'_type': 'uniform', '_value': [0.01, 0.999]},
    # 'head_dropout': {'_type': 'uniform', '_value': [0, 0.999]},
    'batch_size': {'_type': 'quniform', '_value': [2, 1000, 2]},
    # 'frequency': {'_type': 'uniform', '_value': [0.01, 0.5]},
    # 'sampling_rate': {'_type': 'quniform', '_value': [0,200,1]},
}
# 'd_model': {'_type': 'quniform', '_value': [1, 100, 1]},

experiment.config.trial_command = 'bash scripts/PatchTST/ettm1.sh'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1

experiment.run(8099)

