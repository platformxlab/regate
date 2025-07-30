from pydantic import TypeAdapter

from trace_util.llm_ops_generator.configs.models.ModelConfig import ModelConfig


# @dataclass(kw_only=True)
class LLMConfig(ModelConfig):
    input_seqlen: int
    output_seqlen: int
    d_model: int
    num_heads: int
    num_kv_heads: int = -1
    '''
    -1 or the same as num_heads: MHA, will be treated as num_kv_heads == num_heads.
    1: MQA.
    '''
    d_head: int
    d_ff: int
    num_layers: int
    ffn_type: str

    decode_width: int = 1
    use_flash_attention: bool = True

    model_type: str = "llm"

    def __init__(self, **kwargs):
        if "num_kv_heads" not in kwargs:
            # If num_kv_heads is not provided, default to num_heads.
            kwargs["num_kv_heads"] = kwargs["num_heads"]
        super().__init__(**kwargs)


class MoELLMConfig(LLMConfig):
    '''
    MoE LLM model configuration.
    '''

    router_type: str = "topk"
    num_shared_experts: int = 1
    '''Number of common experts that are shared across all tokens.'''
    num_routed_experts: int = 256
    '''Total number of routed experts.'''
    num_activated_routed_experts_per_token: int = 8
    '''Total number of activated routed experts per token, excluding shared experts.'''
    num_limited_groups: int = 4
    '''Max number of expert groups to route to.'''
    moe_d_ff: int = -1
    '''Dimension of the MoE feed-forward network. Defaults to d_ff.'''

    expert_parallelism_degree: int = 1
    num_expert_parallel_axes: int = 1
    expert_parallel_degree_dcn: int = 1

    @property
    def expert_tensor_parallelism_degree(self) -> int:
        '''
        Returns the expert tensor parallelism degree.
        This is computed as dp*tp // ep.
        '''
        return self.data_parallelism_degree * self.tensor_parallelism_degree // self.expert_parallelism_degree

    @property
    def num_expert_tensor_parallel_axes(self) -> int:
        '''
        Returns the number of expert tensor parallel axes.
        This is computed as dp + tp - ep.
        '''
        return (self.num_data_parallel_axes + self.num_tensor_parallel_axes) - self.num_expert_parallel_axes

    @property
    def num_experts_per_token(self) -> int:
        '''
        Returns the total number of experts per token, including shared experts and routed experts.
        '''
        return self.num_shared_experts + self.num_activated_routed_experts_per_token

    def __init__(self, **kwargs):
        if "moe_d_ff" not in kwargs:
            # If moe_d_ff is not provided, default to d_ff.
            kwargs["moe_d_ff"] = kwargs["d_ff"]
        super().__init__(**kwargs)


class DeepSeekConfig(MoELLMConfig):
    '''
    DeepSeek model configuration.
    '''

    num_dense_layers: int = 1
    '''Number of dense layers in the model. Will be the first layer(s) in the model.'''

    # MLA configs
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    num_dense_layers: int = 1
    '''Number of dense layers in the model. Will be the first layer(s) in the model.'''

    @property
    def qk_head_dim(self) -> int:
        '''
        Returns the total dimension of the query-key head.
        '''
        return self.qk_rope_head_dim + self.qk_nope_head_dim

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
