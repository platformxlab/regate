import json
from math import ceil
from typing import Any

from trace_util.llm_ops_generator.configs.models.ModelConfig import ModelConfig
import trace_util.llm_ops_generator.query_results_helper_lib as results_lib

from trace_util.llm_ops_generator.configs.models.DiTConfig import DiTConfig
from trace_util.llm_ops_generator.configs.models.DLRMConfig import DLRMConfig
from trace_util.llm_ops_generator.configs.models.GLIGENConfig import GLIGENConfig
from trace_util.llm_ops_generator.configs.models.LLMConfig import LLMConfig


BYTES_FP32 = 4
BYTES_FP16 = 2


def get_llm_training_mem_requirement(
    config: str | dict[str, Any] | LLMConfig,
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
    optimizer_bytes_per_element: int = BYTES_FP32,
) -> int:
    '''
    Calculate the memory requirement for training a LLM model with DP/TP/PP.
    @config: path to the config file or the config dict.
    @weight_bytes_per_element: bytes per element for weights. Defaults to FP16.
    @activation_bytes_per_element: bytes per element for activations. Defaults to FP16.
    @optimizer_bytes_per_element: bytes per element for optimizer states. Defaults to FP32.

    @return: memory requirement in bytes per chip.
    '''
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = LLMConfig.model_validate(config)

    dp: int = config.data_parallelism_degree
    tp: int = config.tensor_parallelism_degree
    pp: int = config.pipeline_parallelism_degree

    global_batch_size = config.global_batch_size
    batch_size = ceil(global_batch_size / dp)

    num_heads: int = config.num_heads
    d_head: int = config.d_head
    d_model: int = config.d_model
    d_ff: int = config.d_ff
    num_layers: int = config.num_layers

    num_heads = ceil(num_heads / tp)
    d_ff = ceil(d_ff / tp)
    num_layers_per_chip = ceil(num_layers / pp)
    input_seqlen: int = config.input_seqlen

    # attn
    w_attn = 0
    a_attn = 0

    # input
    a_attn += batch_size * input_seqlen * ceil(d_model / tp)
    # layer norm
    w_attn += 2 * batch_size * input_seqlen * d_model
    a_attn += 2 * batch_size * input_seqlen * d_model
    # Wq, Wk, Wv -> einsum
    w_attn += 3 * d_model * num_heads * d_head
    a_attn += max(3 * d_model * num_heads * d_head, 3 * batch_size * input_seqlen * num_heads * d_head)
    # Q*K
    w_attn += 0
    a_attn += batch_size * input_seqlen * input_seqlen * num_heads
    # softmax
    w_attn += 0
    a_attn += batch_size * input_seqlen * input_seqlen * num_heads
    # (Q*K)*V
    w_attn += 0
    a_attn += batch_size * input_seqlen * num_heads * d_head
    # output einsum
    w_attn += num_heads * d_head * d_model
    a_attn += max(num_heads * d_head * d_model, batch_size * input_seqlen * d_model)
    # output layernorm
    w_attn += 2 * batch_size * input_seqlen * d_model
    a_attn += 2 * batch_size * input_seqlen * d_model

    # ffn
    w_ff = 0
    a_ff = 0
    # up + gate
    w_ff += 2 * (d_model * d_ff)
    a_ff += max(2 * (d_model * d_ff), 2 * batch_size * input_seqlen * d_ff)
    # elementwise mul
    w_ff += 0
    a_ff += batch_size * input_seqlen * d_ff
    # down
    w_ff += d_ff * d_model
    a_ff += max(d_ff * d_model, batch_size * input_seqlen * d_model)

    w_ff *= num_layers_per_chip
    a_ff *= num_layers_per_chip

    w = w_attn + w_ff
    a = a_attn + a_ff  # activations and grads

    # https://arxiv.org/pdf/2108.05818
    ## master param + momentum + variance in fp32
    opt = 3 * w

    mem = w * weight_bytes_per_element + a * activation_bytes_per_element + opt * optimizer_bytes_per_element
    return mem


def get_llm_inference_mem_requirement(
    config: str | dict[str, Any] | LLMConfig,
    prefill_or_decode: str = "decode",
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
) -> int:
    '''
    Calculate the memory requirement for serving a LLM model with DP/TP/PP. \\
    @config: path to the config file or the config dict. \\
    @prefill_or_decode: "prefill" or "decode". This is used to determine the KV cache size.
        If "prefill", only the input sequence length is considered.
        If "decode", both input and output sequence lengths are considered. \\
    @weight_bytes_per_element: bytes per element for weights. Defaults to FP16. \\
    @activation_bytes_per_element: bytes per element for activations. Defaults to FP16. \\

    @return: memory requirement in bytes per chip.
    '''
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = LLMConfig.model_validate(config)

    # dp: int = config.data_parallelism_degree * config.data_parallel_degree_dcn
    tp: int = config.tensor_parallelism_degree * config.tensor_parallel_degree_dcn
    pp: int = config.pipeline_parallelism_degree * config.pipeline_parallel_degree_dcn

    # global_batch_size = config.global_batch_size
    # batch_size = ceil(global_batch_size / dp)
    # per dp-ici replica batch size
    batch_size = ceil(config.microbatch_size_ici / config.data_parallelism_degree)

    num_heads: int = config.num_heads
    num_kv_heads: int = config.num_kv_heads
    if num_kv_heads == -1:
        num_kv_heads = num_heads
    d_head: int = config.d_head
    d_model: int = config.d_model
    d_ff: int = config.d_ff
    num_layers: int = config.num_layers

    num_heads = ceil(num_heads / tp)
    num_kv_heads = ceil(num_kv_heads / tp)
    d_ff = ceil(d_ff / tp)
    num_layers_per_chip = ceil(num_layers / pp)
    input_seqlen: int = config.input_seqlen
    output_seqlen: int = config.output_seqlen
    seqlen = input_seqlen + output_seqlen if prefill_or_decode == "decode" else input_seqlen

    ### Attention Layer ###

    # For GQA, if TP <= # of KV heads, then the KV cache is shared.
    # Otherwise, the KV cache needs to be replicated across TP chips.
    if tp <= num_kv_heads:
        w_attn_kv = d_model * num_kv_heads * d_head * 2
    else:
        w_attn_kv = d_model * num_heads * d_head * 2
    w_attn_q = d_model * num_heads * d_head
    w_attn_qkv = w_attn_kv + w_attn_q
    w_attn_output = num_heads * d_head * d_model
    w_attn = w_attn_qkv + w_attn_output  # attention weights

    a_attn_q = batch_size * seqlen * num_heads * d_head
    if tp <= num_kv_heads:
        a_attn_kv = batch_size * seqlen * num_kv_heads * d_head * 2
    else:
        a_attn_kv = batch_size * seqlen * num_heads * d_head * 2
    a_attn_qkv = a_attn_kv + a_attn_q  # KV cache size + Q activation size
    a_attn = a_attn_qkv  # KV cache needs separate storage, Q activation can be used in place

    ### FFN Layer ###

    if config.ffn_type == "default":
        w_ff = 2 * d_model * d_ff
        a_ff = batch_size * seqlen * max(d_ff, d_model)
    elif config.ffn_type == "llama":
        w_ff = 3 * d_model * d_ff
        a_ff = 2 * batch_size * seqlen * max(d_ff, d_model)
    else:
        raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

    # a_ff = 2 * batch_size * seqlen * max(d_ff, d_model)

    total_weights = (w_attn + w_ff) * num_layers_per_chip
    total_act = (a_attn + a_ff) * num_layers_per_chip

    mem = total_weights * weight_bytes_per_element + total_act * activation_bytes_per_element
    return mem


def get_dlrm_inference_mem_requirement(
    config: str | dict[str, Any] | DLRMConfig,
    weight_bytes_per_element: int = BYTES_FP32,
    activation_bytes_per_element: int = BYTES_FP32,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = DLRMConfig.model_validate(config)
    memory_capacity_per_chip = config.hbm_size_GB
    # TODO: for now, assume at least 8 chips for DLRM
    return 8 * memory_capacity_per_chip * 1024**3 - 1024  # offset a little bit for safety


def get_dit_inference_mem_requirement(
    config: str | dict[str, Any] | DiTConfig,
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = DiTConfig.model_validate(config)

    def get_llm_config_from_dit_config(config: DiTConfig) -> LLMConfig:
        '''
        Convert DiT config to LLM config.
        '''
        llm_config = LLMConfig(**config.__dict__)
        llm_config.model_type = "llm"
        llm_config.input_seqlen = ceil(
            (config.image_width * config.image_width)
            / (config.patch_size * config.patch_size)
        )
        llm_config.output_seqlen = 1
        return llm_config

    llm_config = get_llm_config_from_dit_config(config)
    return get_llm_inference_mem_requirement(
        llm_config,
        prefill_or_decode="prefill",
        weight_bytes_per_element=weight_bytes_per_element,
        activation_bytes_per_element=activation_bytes_per_element,
    )


def get_gligen_inference_mem_requirement(
    config: str | dict[str, Any] | GLIGENConfig,
    weight_bytes_per_element: int = BYTES_FP32,
    activation_bytes_per_element: int = BYTES_FP32,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = GLIGENConfig.model_validate(config)
    memory_capacity_per_chip = config.hbm_size_GB
    # TODO: for now, assume at least 1 chip for GLIGEN
    return memory_capacity_per_chip * 1024**3 - 1024  # offset a little bit for safety


def get_mem_requirement(
    config: str | dict[str, Any] | ModelConfig,
    model: str,
    workload: str,
    weight_bytes_per_element: int = 4,
    activation_bytes_per_element: int = 4,
) -> int:
    if results_lib.is_model_llm(model):
        assert isinstance(config, (str, dict, LLMConfig)), \
            f"Expected config to be a path, dict, or LLMConfig instance, got {type(config)}"
        if workload == "training":
            return get_llm_training_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        elif workload == "inference":
            return get_llm_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        else:
            raise ValueError(f"Unknown workload: {workload}")
    elif results_lib.is_model_dlrm(model):
        assert isinstance(config, (str, dict, DLRMConfig)), \
            f"Expected config to be a path, dict, or DLRMConfig instance, got {type(config)}"
        return get_dlrm_inference_mem_requirement(
            config,
            weight_bytes_per_element=weight_bytes_per_element,
            activation_bytes_per_element=activation_bytes_per_element,
        )
    elif results_lib.is_model_sd(model):
        if "dit" in model.lower():
            assert isinstance(config, (str, dict, DiTConfig)), \
                f"Expected config to be a path, dict, or DiTConfig instance, got {type(config)}"
            return get_dit_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        elif "gligen" in model.lower():
            assert isinstance(config, (str, dict, GLIGENConfig)), \
                f"Expected config to be a path, dict, or GLIGENConfig instance, got {type(config)}"
            return get_gligen_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        raise ValueError(f"Unknown model: {model}")
