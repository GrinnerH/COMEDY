# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.modeling_utils import load_sharded_checkpoint 

from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers import AutoModelForSequenceClassification
# from transformers.models.gptj.configuration_gptj import GPTJConfig
# from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

from .reward_model import RewardModel


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config, trust_remote_code=True)
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=model_config,
            trust_remote_code=True)

    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # TODO why?
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_critic_model(model_name_or_path,
                        tokenizer,
                        ds_config,
                        num_padding_at_beginning=0,
                        rlhf_training=False,
                        is_infer=False):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule
    critic_model = create_hf_model(AutoModelForSequenceClassification, model_name_or_path, tokenizer,
                                   ds_config, rlhf_training)
    
    critic_model = RewardModel(
            critic_model,
            tokenizer,
            num_padding_at_beginning=num_padding_at_beginning)

    if rlhf_training:
        if os.path.exists(os.path.join(model_name_or_path, 'pytorch_model.bin')):
            model_ckpt_path = os.path.join(model_name_or_path, 'pytorch_model.bin')
            critic_model.load_state_dict(
                torch.load(model_ckpt_path, map_location='cpu'), strict=False
            )
        elif os.path.exists(os.path.join(model_name_or_path, 'pytorch_model.bin.index.json')):
            load_sharded_checkpoint(critic_model, model_name_or_path, strict=False)
        else:
            raise Exception(f"Cannot find model checkpoint at {model_ckpt_path}")

    return critic_model
