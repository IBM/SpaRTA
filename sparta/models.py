import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors import safe_open


@torch.no_grad()
def load_classification_model(model_name, num_classes, head_init='random', sep_token=None, **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        device_map = 'auto'  # loads model into multiple gpus
    else:
        device_map = None  # loads model to cpu and SFT will move it to cuda

    model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                               num_labels=num_classes,
                                                               torch_dtype=torch.bfloat16,
                                                               device_map=device_map,
                                                               **kwargs)
    # classification head init params
    if head_init == 'random':
        model.score.weight.data *= 3. / math.sqrt(model.score.in_features)  # * 0.1 (google/gemma-2-27b)

    elif head_init == 'from_pretrained':
        # reuse vocab head to init classification head (only for instruct-tuned PT models)
        vocab = tokenizer.get_vocab()  # vocab[token] <=> tokenizer.convert_tokens_to_ids(token)
        if num_classes == 2:
            classification_tokens = ["▁No", "▁Yes"]
        elif num_classes == 5:
            classification_tokens = ["1", "2", "3", "4", "5"]
        class_ids = [vocab[token] for token in classification_tokens]

        head_fpath, param_name = find_head(model)
        with safe_open(head_fpath, framework='pt') as f:
            model.score.weight.data = f.get_slice(param_name)[class_ids].to(model.score.weight.data.device)
    else:
        raise ValueError(f"head_init = '{head_init}', " +
                         "but only options are 'random' or 'from_pretrained'")

    if hasattr(model, 'hf_device_map'):
        # move head outout (logits) to model.device (cuda:0) where labels are
        def set_output_device_hook(module, input, output):
            return output.to(model.device)
        model.score.register_forward_hook(set_output_device_hook)

    # pad_token config
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # extend vocab to add SEP_TOKEN
    # tokenizer.add_special_tokens({"additional_special_tokens": [SEP_TOKEN]},
    #                             replace_additional_special_tokens=False)
    # model.resize_token_embeddings(len(tokenizer))

    if sep_token:
        assert tokenizer.sep_token is None
        num_added_tokens = tokenizer.add_special_tokens({'sep_token': sep_token})
        assert num_added_tokens == 1
        print(f">> Added sentence separator token '{tokenizer.sep_token}' with id {tokenizer.sep_token_id}")
        resize_token_embeddings(model, num_added_tokens)

    return tokenizer, model


def find_head(model):

    HEAD_PARAM_NAME_MAP = {
        'google/gemma-2b-it': 'model.embed_tokens.weight',
        'mistralai/Mistral-7B-Instruct-v0.3': 'lm_head.weight'
        }

    param_name = HEAD_PARAM_NAME_MAP[model.config._name_or_path]

    if os.path.isdir(model.config._name_or_path):
        model_dir = model.config._name_or_path
    else:
        cache_dir = os.getenv('HF_HOME', '~/.cache/huggingface/')
        model_name = model.config._name_or_path.replace('/', '--')
        cache_dir = os.path.join(cache_dir, f"models--{model_name}")
        if not os.path.isdir(cache_dir):
            raise ValueError(f"PT model '{model.config._name_or_path}' not in HF cache")
        with open(os.path.join(cache_dir, 'refs', 'main')) as f:
            revision = f.read()
        model_dir = os.path.join(cache_dir, 'snapshots', revision)

    try:  # params sharded in multiple files
        index_file = os.path.join(model_dir, 'model.safetensors.index.json')
        with open(index_file, 'r') as f:
            fname = json.load(f)['weight_map'][param_name]
    except FileNotFoundError:
        fname = 'model.safetensors'  # params stored in a single file
    except KeyError:
        raise ValueError(f"param '{param_name}' not in {index_file}")

    head_fpath = os.path.join(model_dir, fname)

    return head_fpath, param_name


@torch.no_grad()
def resize_token_embeddings(model, num_added_tokens=1, init_centered=False):
    embeddings = model.get_input_embeddings()
    new_embeddings = torch.randn(num_added_tokens, embeddings.embedding_dim,
                                 dtype=embeddings.weight.dtype,
                                 device=embeddings.weight.device)
    if not init_centered:
        new_embeddings = new_embeddings * model.config.initializer_range
    else:
        new_embeddings = new_embeddings * 1e-4 + torch.mean(embeddings.weight.data, axis=0)
    embeddings.weight.data = torch.cat((embeddings.weight.data, new_embeddings))
    embeddings.num_embeddings += num_added_tokens
    model.config.vocab_size += num_added_tokens
