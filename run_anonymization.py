from pathlib import Path
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--config', default='anon_config.yaml')
parser.add_argument('--lang', default='en', choices=['en', 'de', 'fr', 'it', 'es', 'pt', 'nl', 'pl', 'ru'])
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--force_compute', default=False, type=bool)
parser.add_argument('--anonymize_train_data', default=False, type=bool)
parser.add_argument('--full_train_data', default=False, type=bool)
args = parser.parse_args()

if 'CUDA_VISIBLE_DEVICES' not in os.environ:  # do not overwrite previously set devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

import torch

#from anonymization.pipelines.sttts_pipeline import STTTSPipeline
from anonymization.pipelines.sfm_pipeline import SFMPipeline
from utils import parse_yaml, get_datasets, setup_logger

PIPELINES = {
    #'sttts': STTTSPipeline,
    'sfm': SFMPipeline
}

if __name__ == '__main__':
    config_path = Path('configs', args.config)
    config = parse_yaml(config_path, overrides={'lang': args.lang})

    if args.anonymize_train_data:
        if args.full_train_data:  # anonymize all train data (for ASV training from scratch)
            datasets = {f'train-clean-360_part': Path(config['data_dir'], f'train-clean-360_part')}
        else: # anonymize only part of the train data (for ASV finetuning)
            datasets = {f'train-clean-360': Path(config['data_dir'], f'train-clean-360')}
        config['modules']['speaker_embeddings']['emb_level'] = 'utt'  # train data for eval models is anonymized on utt level
        config['modules']['speaker_embeddings']['extraction_results_path'] = (
            Path(str(config['modules']['speaker_embeddings']['extraction_results_path']).replace('spk-level',
                                                                                                 'utt-level')))
        config['modules']['speaker_embeddings']['anon_results_path'] = (
            Path(str(config['modules']['speaker_embeddings']['anon_results_path']).replace('spk-level', 'utt-level')))
    else:
        datasets = get_datasets(config)

    devices = []
    if torch.cuda.is_available():
        gpus = args.gpu_ids.split(',')
        devices = [torch.device(f'cuda:{i}') for i in range(len(gpus))]
    else:
        devices = [torch.device('cpu')]

    with torch.no_grad():
        logger = setup_logger(__name__)
        logger.info(f'Running pipeline: {config["pipeline"]}')
        pipeline = PIPELINES[config['pipeline']](config=config, force_compute=args.force_compute, devices=devices,
                                                 config_name=config_path.stem)
        pipeline.run_anonymization_pipeline(datasets)
