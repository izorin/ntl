from os import path
import sys
import fire

sys.path.append('/Users/ivan_zorin/Documents/DEV/code/ntl/') 
# FIXME move this import to the top
from trainer import Trainer
from utils import load_config


def main(experiment_config, path_config):
    config = load_config(experiment_config, path_config)
    
    trainer = Trainer(config)
    trainer.train()
    
    # trainer.save()



if __name__ == '__main__':
    
    # fire.Fire(main)
    experiment_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/trainer_debug.yaml'
    path_config = '/Users/ivan_zorin/Documents/DEV/code/ntl/configs/local_pathes.yaml'
    main(experiment_config, path_config)
    
    
