import re
import os
from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainerState


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_and_model_for_hf_trainer(trainer: Trainer):
    """
    save the state and model for trainer
    :param trainer: transformers.Trainer to be saved
    :return:
    """
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    # save model at output_dir
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def load_state_and_model_for_hf_trainer(model: nn.Module, load_model_dir: str, map_location: str = None):
    """
    load the state and model for trainer
    :param model: nn.Module, the model to be loaded
    :param load_model_dir: str, the path where the state and model to be loaded
    :param map_location: str, how to remap the storage locations
    :return:
    """
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(os.path.join(load_model_dir, "pytorch_model.bin"), map_location=map_location))
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(os.path.join(load_model_dir, "trainer_state.json"))
    return model, trainer_state


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    assert tokenizer.vocab_size == 32000
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
