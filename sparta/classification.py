import logging
import json
from time import time

import click
import numpy as np

from tasks import load_classification_data, DATASETS
from sft import SFT_Config, SFT, PEFT_METHODS

logger = logging.getLogger(__name__)
log = logger

BASE_MODELS = ['google/gemma-2b',
               'mistralai/Mistral-7B-v0.3',
               'ibm-granite/granite-3.0-2b-base',
               'google/gemma-2-27b',
               'EleutherAI/pythia-160m']

INSTRUCT_MODELS = ['google/gemma-2b-it',
                   'mistralai/Mistral-7B-Instruct-v0.3']
MODELS = INSTRUCT_MODELS + BASE_MODELS
DEFAULT_SEED = 10605


def run(*,
        dataset_name: str,
        peft_method: str,
        model_name: str,
        train_input_maxlen: int,
        bs_train: int,
        bs_eval: int,
        num_epochs: int,
        lr: float,
        lr_min: float,
        warmup_steps: int,
        lr_decay_steps: int,
        sparsity: float,
        dropout: float,
        weight_decay: float,
        max_grad_norm: float,
        print_freq: int,
        save_dir: str,
        output_dir: str,
        seed: int,
        sft_config_fpath: str,
        ):

    # PT model
    if model_name in INSTRUCT_MODELS:
        instruction_format = True
        head_init = 'from_pretrained'
    elif model_name in BASE_MODELS:
        instruction_format = False
        head_init = 'random'
    else:
        raise ValueError(f"{model_name=} not supported: not in list {INSTRUCT_MODELS} nor in {BASE_MODELS}")

    print(f'>> Loading classification (task) dataset: {dataset_name}')
    ds_train, ds_val, ds_test, sep_token = load_classification_data(dataset_name, instruction_format)
    num_classes = ds_train.features['label'].num_classes

    # Hardcoded to shrink training data  -> Discuss this
    if ds_train.num_rows >= 100000:
        ds_train = ds_train.select(range(100000))

    print(f'>> Pre-Trained (PT) model: {model_name}')
    model_config = {'name_or_fpath': model_name,
                    'task': 'SEQ_CLS',
                    'sep_token': sep_token,
                    'num_classes': num_classes,
                    'head_init': head_init}

    if peft_method == 'lora':
        print('>> Creating LoRA model')
        peft_config = {'task_type': 'SEQ_CLS',
                       'inference_mode': False,
                       'r': 8,
                       'lora_alpha': 16,
                       'lora_dropout': 0.1}
        # peft_config['target_modules'] = ['q_proj', 'o_proj']

    elif peft_method == 'sparse':
        print('>> Creating Sparse model')
        peft_config = {'sparsity': sparsity}
        # comment below the target modules for sparsification
        peft_config['frozen_modules'] = ['embed_tokens',
                                         'self_attn.q',
                                         'self_attn.k',
                                         # 'self_attn.v',
                                         # 'self_attn.o',
                                         'mlp',
                                         'norm',
                                         ]

    else:  # 'head_only' or 'full_sft'
        peft_config = None

    sft_config = SFT_Config()
    sft_config['train_input_maxlen'] = train_input_maxlen
    sft_config['bs_train']       = bs_train
    sft_config['bs_eval']        = bs_eval
    sft_config['num_epochs']     = num_epochs
    sft_config['lr']             = lr  # too small do 1e-4 range
    sft_config['lr_min']         = lr_min  # for linear decay
    sft_config['warmup_steps']   = warmup_steps
    sft_config['lr_decay_steps'] = lr_decay_steps
    sft_config['dropout']        = dropout
    sft_config['weight_decay']   = weight_decay
    sft_config['max_grad_norm']  = float('inf') if max_grad_norm is None else max_grad_norm
    sft_config['print_freq']     = print_freq
    sft_config['output_dir']     = output_dir
    sft_config['save_dir']       = save_dir

    # overrides
    if dataset_name == 'imdb':
        sft_config['train_input_maxlen'] = 384
        print(f"OVERRIDE: imdb {sft_config['train_input_maxlen']=}")

    # overrides from config file
    if sft_config_fpath:
        sft_config.update_from_yaml(sft_config_fpath)
        print(f"OVERRIDE: SFTConfig from {sft_config_fpath}")

    sft = SFT(sft_config, model_config, ds_train, ds_val, ds_test, peft_method, peft_config)

    print('Model:\n', sft.model)
    if hasattr(sft.model, 'hf_device_map'):
        print('device_map:', sft.model.hf_device_map)

    if peft_method == 'lora' or peft_method == 'sparse':
        sft.model.print_trainable_parameters()

    print('>> Created tokenized dataloaders from datasets')
    if sft.config['train_input_maxlen']:
        print(f'Num. of training examples after filtering: {len(sft.train_dataloader.dataset)}')
    print(f"Max input token length when training: {max([input_ids.shape[0] for input_ids in sft.train_dataloader.dataset['input_ids']])}")

    test_dataloader = sft.create_dataloader(ds_test, sft.config['bs_eval'])

    print('>> Evaluate PT model')
    sft.evaluate(test_dataloader, verbose=True)

    print('>> SFT training: fine-tunes PT model')
    tic = time()
    sft.train()
    print(f'Training time: {(time()-tic)/60:.1f} minutes')

    print('>> Evaluate SFT model')
    sft.evaluate(test_dataloader, verbose=True)

    if sft.val_dataloader:
        print('>> Best model (by early stopping):')
        print('Best validation loss: {0:.3f}'.format(min(sft.stats['val_loss'])))
        ind = np.argmin(np.array(sft.stats['val_loss']))
        print('Best validation loss epoch: {0:3d}'.format(sft.stats['val_loss_epoch'][ind]))
        print('Best validation loss batch: {0:3d}'.format(sft.stats['val_loss_batch'][ind]))
        print('Test loss according to min validation loss: {0:.3f}'.format(sft.stats['test_loss'][ind]))
        print('Test acc according to min validation loss: {0:.3f}'.format(sft.stats['test_acc'][ind]*100))
        print('Test mcc according to min validation loss: {0:.3f}'.format(sft.stats['test_mcc'][ind]))
        ind = np.argmax(np.array(sft.stats['val_acc']))
        print('Best validation accuracy: {0:.1f}%'.format(max(sft.stats['val_acc'])*100))
        print('Best validation accuracy epoch: {0:3d}'.format(sft.stats['val_acc_epoch'][ind]))
        print('Test loss according to max validation acc: {0:.3f}'.format(sft.stats['test_loss'][ind]))
        print('Test acc according to max validation acc: {0:.3f}'.format(sft.stats['test_acc'][ind]*100))
        print('Test mcc according to max validation acc: {0:.3f}'.format(sft.stats['test_mcc'][ind]))
        ind = np.argmax(np.array(sft.stats['val_mcc']))
        print('Best validation MCC: {0:.1f}%'.format(max(sft.stats['val_mcc'])))
        print('Best validation MCC epoch: {0:3d}'.format(sft.stats['val_mcc_epoch'][ind]))
        print('Test loss according to max validation mcc: {0:.3f}'.format(sft.stats['test_loss'][ind]))
        print('Test acc according to max validation mcc: {0:.3f}'.format(sft.stats['test_acc'][ind]*100))
        print('Test mcc according to max validation mcc: {0:.3f}'.format(sft.stats['test_mcc'][ind]))

    print("SFT Config:")
    print(sft_config)
    last_lr = sft.lr_scheduler.get_last_lr()[0]
    print('>> Last learning rate: {0:.1e}'.format(last_lr))

    if peft_method == 'sparse':
        s = '-' + str(peft_config['sparsity']).split('.')[-1]
    else:
        s = ''
    fname_prefix = f"{peft_method}{s}-{model_name.split('/')[-1]}-{dataset_name}-"

    sft.plot_stats(fname_prefix)

    # sft.config['save_dir'] += dataset_name # + '-it'
    # Use:
    #    merged = False # for merging sparse models
    #    merged = True (default) # for generation
    # sft.save_model(merged=False)

@click.command()
@click.option("--dataset-name", type=click.Choice(DATASETS), help="dataset to use")
@click.option("--peft-method", type=click.Choice(PEFT_METHODS), help="peft method to use")
@click.option("--model-name", type=click.Choice(MODELS), help="model to use")
@click.option("--train-input-maxlen", type=int, default=512, help="max len (tokens) for training input sample, if longer, filtered out.")
@click.option("--bs-train", type=int, default=24, help="Batch size to use for training.")
@click.option("--bs-eval", type=int, default=48, help="Batch size to use for evaluation.")
@click.option("--num-epochs", type=int, default=1, help="Number of epochs to train for.")
@click.option("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
@click.option("--lr-min", type=float, default=5e-5, help="Learning rate decay from lr to lr_min.")
@click.option("--warmup-steps", type=int, default=100, help="Number of steps to warm up to learning rate")
@click.option("--lr-decay-steps", type=int, default=5000, help="number of training iteration for LR decay to reach lr_min.")
@click.option("--sparsity", type=float, default=0.99, help="sparsity rate if self.config['peft_method'] == sparse.")
@click.option("--dropout", type=float, default=0.1, help="dropout rate.")
@click.option("--weight-decay", type=float, default=0.0, help="weight decay for the AdamW optimizer.")
@click.option("--max-grad-norm", type=float, default=None, help="gradient clipping.")
@click.option("--print-freq", type=int, default=100, help="frequency of print of values on output files.")
@click.option("--save-dir", type=str, default="./save", help="location directory to save files")
@click.option("--output-dir", type=str, default="./output/", help="director for saving output (plots)")
@click.option("--sft_config_fpath", type=str, default='', help="config file in yaml format")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
def main(**kwargs):
    print(json.dumps(kwargs, indent=4))
    run(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
