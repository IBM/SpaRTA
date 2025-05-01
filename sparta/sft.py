import os
import yaml
import torch
import matplotlib.pyplot as plt
from utils import mcc, f1_score


PEFT_METHODS = ['lora', 'sparse', 'head_only', 'full_sft']


class SFT_Config(dict):
    def __init__(self):
        super().__init__()

        self['train_input_maxlen'] = 512     # 512 (filters out everything larger than this number)
        self['bs_train']           = 24      # batch size of train dataloader
        self['bs_eval']            = 48      # batch size of val/test dataloader
        self['num_epochs']         = 1
        self['lr']                 = 5e-5    # learning rate (max)
        self['lr_min']             = 5e-5    # lr_scheduler: decays lr linearly from lr to lr_min
        self['warmup_steps']       = 100     #               after warmup
        self['lr_decay_steps']     = 5000    #               in a number of training iterations
        self['early_stopping']     = False
        self['dropout']            = 0.0     # dropout rate
        self['weight_decay']       = 0.0     # (decoupled) weight decay regularization for the AdamW optimizer
        self['max_grad_norm']      = None    # gradient clipping
        self['print_freq']         = 100
        self['output_dir']         = './output'
        self['save_dir']           = './save'

    def update_from_yaml(self, fpath):
        with open(fpath, 'r') as f:
            updates = yaml.safe_load(f)
            self.update(updates)


class SFT:
    def __init__(self, config, model_config, train_dataset, val_dataset=None, test_dataset=None, peft_method=None, peft_config=None):

        self.config = config
        self.config['peft_method'] = 'full_sft' if peft_method is None else peft_method.lower()
        if self.config['peft_method'] not in PEFT_METHODS:
            raise ValueError(f"peft_method = '{peft_method}' not supported")

        if model_config['task'] == 'SEQ_CLS':
            from models import load_classification_model
            tokenizer, model = load_classification_model(model_config['name_or_fpath'],
                                                         model_config['num_classes'],
                                                         head_init=model_config['head_init'],
                                                         sep_token=model_config['sep_token'],
                                                         attention_dropout=self.config['dropout'])
            self.task = 'SEQ_CLS'
        else:
            raise NotImplementedError(f"Task: {model_config['task']} not implemented")

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

        if self.config['peft_method'] == 'sparse':

            from sparsify import SparsifyModel
            model = SparsifyModel(model, **peft_config)

        elif self.config['peft_method'] == 'lora':

            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(**peft_config)
            model = get_peft_model(model, lora_config)

        else:
            pass

        self.model = model

        if not hasattr(self.model, 'hf_device_map'):
            self.model.to('cuda')

        if self.config['peft_method'] == 'head_only':
            params_to_optimize = [p for name, p in self.model.named_parameters() if 'score' in name]
        else:
            params_to_optimize = self.model.parameters()

        self.optimizer = self.configure_optimizer(params_to_optimize)

        self.lr_scheduler = self.configure_lr_scheduler(self.optimizer)

        self.train_dataloader = self.create_dataloader(train_dataset,
                                                       self.config['bs_train'],
                                                       shuffle=True,
                                                       max_token_len=self.config['train_input_maxlen'])

        self.val_dataloader = self.create_dataloader(val_dataset,
                                                     self.config['bs_eval'])

        self.test_dataloader = self.create_dataloader(test_dataset,
                                                      self.config['bs_eval'])

        self.stats = {'train_loss': [], 'val_loss': [], 'grad_norm': [], 'lr': [], 'val_loss_epoch': [], 'val_loss_batch': []}
        if self.task == 'SEQ_CLS':
            self.stats['val_acc'] = []
            self.stats['val_acc_epoch'] = []
            self.stats['val_mcc'] = []
            self.stats['val_mcc_epoch'] = []
            self.stats['test_loss'] = []
            self.stats['test_acc'] = []
            self.stats['test_mcc'] = []

    def configure_optimizer(self, params_to_optimize):
        return torch.optim.AdamW(params_to_optimize,
                                 lr=self.config['lr'], betas=(0.9, 0.99),
                                 weight_decay=self.config['weight_decay'],
                                 fused=True)

    def configure_lr_scheduler(self, optimizer):
        lr_max, lr_min = self.config['lr'], self.config['lr_min']
        warmup_steps = self.config['warmup_steps']
        decay_steps = self.config['lr_decay_steps']
        assert lr_max >= lr_min and decay_steps > 0
        min_factor = lr_min/lr_max

        def lr_lambda(i):
            """linear lr decay with warmup"""
            if lr_max == lr_min:
                return 1.
            elif i <= warmup_steps:
                return max(min_factor, i/warmup_steps)
            else:
                return max(min_factor, 1 - (i - warmup_steps)/decay_steps)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def create_dataloader(self, dataset, batch_size, shuffle=False, max_token_len=None):

        if dataset is None:
            return None

        dataset = dataset.map(lambda example: {'input_ids': self.tokenizer.encode(example['text'])},
                              batched=False)
        dataset = dataset.remove_columns(['text'])
        dataset = dataset.rename_column('label', 'labels')
        dataset.set_format('torch')
        if max_token_len:
            dataset = dataset.filter(lambda example: len(example['input_ids']) <= max_token_len,
                                     batched=False)
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           collate_fn=lambda examples: self.tokenizer.pad(examples, return_tensors='pt')
                                           )

    def train(self, num_epochs=None):

        num_epochs = self.config['num_epochs'] if num_epochs is None else num_epochs

        self.model.train()

        for epoch in range(num_epochs):
            for i, batch in enumerate(self.train_dataloader):

                batch = {k: v.to(self.model.device) for k, v in batch.items()}  # input_ids, attention_mask, labels

                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                self.stats['train_loss'].append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()

                if type(self.model).__name__ == 'SparsifyModel':
                    self.model.unmerge_deltas()

                # clip params grad (if too large)
                if self.config['max_grad_norm']:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.optimizer.param_groups[0]['params'],
                        self.config['max_grad_norm'], norm_type=2).item()
                    self.stats['grad_norm'].append(grad_norm)

                self.optimizer.step()
                self.stats['lr'].append(self.lr_scheduler.get_last_lr()[0])
                self.lr_scheduler.step()

                stop = self.log_training_progress(epoch, i, freq=self.config['print_freq'])

                if stop:
                    print('\n*** stopping training early ***\n')
                    break
            else:
                continue  # Discuss this
            break

    def log_training_progress(self, epoch, i, freq=10):

        if (epoch*len(self.train_dataloader)+i) % freq == 0:
            if (epoch == 0 and i == 0):
                if self.val_dataloader:
                    epoch_str = 'Epoch'
                    batch_str = 'Batch'
                    loss_str = 'Loss'
                    val_loss_str = 'Val Loss'
                    test_loss_str = 'Test Loss'
                    val_acc_str = 'Val Acc'
                    test_acc_str = 'Test Acc'
                    val_mcc_str = 'Val MCC'
                    test_mcc_str = 'Test MCC'
                    print(f'{epoch_str:5s} {batch_str:10s} {loss_str:7s} {val_loss_str:8s} {test_loss_str:9s} {val_acc_str:7s} {test_acc_str:8s} {val_mcc_str:7s} {test_mcc_str:8s}')
                else:
                    print('Epoch \t Batch \t Loss')

            last_losses = self.stats['train_loss'][-self.train_dataloader.batch_size:]
            print(f'{epoch:3d}\t\t{i:3d}|{len(self.train_dataloader):d}\t\t{sum(last_losses)/len(last_losses):.4f}')

            if self.val_dataloader:
                results = self.evaluate(self.val_dataloader)
                val_loss = results['loss']
                self.stats['val_loss'].append(results['loss'])
                self.stats['val_loss_epoch'].append(epoch)
                self.stats['val_loss_batch'].append(i)
                if self.task == 'SEQ_CLS':
                    val_acc = results['accuracy']
                    val_mcc = results['mcc']
                    self.stats['val_acc'].append(results['accuracy'])
                    self.stats['val_acc_epoch'].append(epoch)
                    self.stats['val_mcc'].append(results['mcc'])
                    self.stats['val_mcc_epoch'].append(epoch)

                    results = self.evaluate(self.test_dataloader)
                    test_loss = results['loss']
                    test_acc = results['accuracy']
                    test_mcc = results['mcc']
                    self.stats['test_loss'].append(test_loss)
                    self.stats['test_acc'].append(test_acc)
                    self.stats['test_mcc'].append(test_mcc)

                    print(f'{epoch:5d} {i:4d}|{len(self.train_dataloader):5d} {sum(last_losses)/len(last_losses):.4f}   {val_loss:.4f}   {test_loss:.4f}   {val_acc:.4f}  {test_acc:.4f}  {val_mcc:.4f}  {test_mcc:.4f}')
                else:
                    print(f'{epoch:3d}\t{i:3d}|{len(self.train_dataloader):d}\t{sum(last_losses)/len(last_losses):.4f}')

                if self.config['early_stopping']:
                    n = len(self.stats['val_loss'])
                    if n > 2 and epoch > 0:
                        prev_best = min(self.stats['val_loss'][:-1])
                        new = self.stats['val_loss'][-1]
                        if new < prev_best + 0.002:
                            pass  # self.save_model(merged=False, save_tokenizer=False, sft_info=False)
                        else:
                            patience = n - 1 - self.stats['val_loss'].index(prev_best)
                            if patience >= 1:  # 1
                                return True  # stop training

                self.model.train()
            else:
                print(f'{epoch:3d}\t{i:3d}|{len(self.train_dataloader):d}\t{sum(last_losses)/len(last_losses):.4f}')

        return False

    @torch.no_grad()
    def evaluate(self, eval_dataloader, verbose=False):
        self.model.eval()
        eval_loss = 0.0
        if self.task == 'SEQ_CLS':
            accuracy = 0.0
            cm = torch.zeros((self.model.num_labels, self.model.num_labels), dtype=torch.int32)

        for batch in eval_dataloader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch, use_cache=False)
            eval_loss += outputs.loss.item()
            if self.task == 'SEQ_CLS':
                predictions = torch.argmax(outputs.logits, dim=-1)
                accuracy += (predictions == batch['labels']).sum().item()

                for label, pred in zip(batch['labels'].tolist(), predictions.tolist()):
                    cm[label, pred] += 1

        eval_loss /= len(eval_dataloader)
        if self.task == 'SEQ_CLS':
            accuracy /= len(eval_dataloader.dataset)
        if verbose:
            print(f'Evaluation Loss: {eval_loss:.3f}')
            if self.task == 'SEQ_CLS':
                print(f'Accuracy: {accuracy*100:.1f}%')
                print(f'Confusion Matrix:\n {cm/cm.sum()}')
                if cm.shape == (2, 2):
                    print(f'MCC: {mcc(cm):.3f}')
                    print(f'F1-score: {f1_score(cm):.3f}')
        else:
            results = {'loss': eval_loss}
            if self.task == 'SEQ_CLS':
                results['accuracy'] = accuracy
                if cm.shape == (2, 2):
                    results['mcc'] = 100*mcc(cm)
                else:
                    results['mcc'] = -1
            return results

    def save_model(self, merged=True, save_tokenizer=True, sft_info=True):

        save_dir = self.config['save_dir']

        if type(self.model).__name__ == 'SparsifyModel':
            self.model.save(save_dir, merged)
        else:
            self.model.save_pretrained(save_dir)

        if save_tokenizer:
            self.tokenizer.save_pretrained(save_dir)

        if sft_info:
            with open(os.path.join(save_dir, 'sft_info.txt'), 'w') as f:
                f.write('config:\n')
                for k, v in self.config.items():
                    f.write(' %s: %s\n' % (k, v))
                # f.write('train_loss: '+ str(self.stats['train_loss'][-1]) +'\n')

        print('\nModel saved in: %s' % save_dir)

    def plot_stats(self, fname_prefix=''):

        fpath_prefix = os.path.join(self.config['output_dir'], fname_prefix)

        num_classes = self.model.num_labels
        loss_baseline = -torch.tensor(1/num_classes).log().item()

        plot(self.stats['train_loss'], w=self.config['print_freq'],
             baseline=loss_baseline,
             title='Training Loss',
             save_file=fpath_prefix + 'train_loss.pdf')

        plot(self.stats['lr'],
             title='Learning Rate',
             save_file=fpath_prefix + 'learning_rate.pdf')

        if self.val_dataloader:
            plot(self.stats['val_loss'],  # w = 1,
                 baseline=loss_baseline,
                 title='Validation Loss',
                 save_file=fpath_prefix + 'val_loss.pdf')

            if self.task == 'SEQ_CLS':
                plot(self.stats['val_acc'],  # w = 1,
                     baseline=1.0,
                     title='Validation Accuracy',
                     save_file=fpath_prefix + 'val_acc.pdf')

        if self.config['max_grad_norm']:
            max_grad_norm = self.config['max_grad_norm']
            plot(self.stats['grad_norm'],  # w = 1,
                 baseline=max_grad_norm if max_grad_norm < float('inf') else 0,
                 title='Gradient Norm',
                 save_file=fpath_prefix + 'grad_norm.pdf')


def plot(ts, w=None, baseline=None, title=None, save_file=None):
    fig, ax = plt.subplots()
    if w:  # rolling avg window
        y = [sum(ts[i:i+w])/w for i in range(len(ts) - w + 1)]
    else:
        y = ts
    ax.plot(y)
    if baseline:
        ax.hlines(baseline, xmin=0, xmax=len(y), color='r')
    ax.set_ylim(bottom=0)
    ax.set_title(title)
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
