"""
@File       : config.py
@Author     : Cao Zhijie
@Email      : 
@Date       : 2020/5/26
@Desc       :
"""
import os
import yaml

def parse_configs(config_file_path,train_or_test='train'):
    with open(config_file_path) as f:
        config_file = yaml.load(f)
    if train_or_test=='train':
        common_config = config_file['common']
        train_config= config_file['train']
        config=dict( common_config, **train_config )
        # define some folder and file path
        config['train_list_path'] = 'data/list/{}_train.json'.format(config['dataset_name'])
        config['test_list_path'] = 'data/list/{}_test.json'.format(config['dataset_name'])
        config['model_save_path'] = os.path.join('output', '{}_{}_{}'.format(config['dataset_name'],config['model_name'],config['model_save_name']))
        config['pretrain_path'] = os.path.join(config['model_save_path'], 'checkpoint.pth.tar')
        config['log_folder_path']=os.path.join(config['model_save_path'],'log')
        if not os.path.exists(config['log_folder_path']):
            os.makedirs(config['log_folder_path'])
    else:
        common_config = config_file['common']
        test_config = config_file['test']
        config = dict(common_config, **test_config)
        config['test_list_path'] = 'data/list/{}_{}.json'.format(config['dataset_name'],config['train_or_test'])
        config['model_save_path'] = os.path.join('output',
                                                 '{}_{}_{}'.format(config['dataset_name'], config['model_name'],
                                                                   config['model_save_name']))
        config['log_folder_path'] = os.path.join(config['model_save_path'], 'log')
        if not os.path.exists(config['log_folder_path']):
            os.makedirs(config['log_folder_path'])
        config['pretrain_path'] = os.path.join(config['model_save_path'], 'best.pth.tar')
    return config

def print_config(config):
    print("\n---------config--------")
    for key in config.keys():
        print("[{}]:{}".format(key,config[key]))
