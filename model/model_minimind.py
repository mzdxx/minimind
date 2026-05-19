import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Config
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"  # 类变量

    def __init__(
        self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs
    ):  # 把不关心的参数丢给kwargs，透传给父类
        super().__init__(
            **kwargs
        )  # 把kwarges字典解包成关键字参数，传给父类的构造器，先调用父类，确保父类初始化完成，再用子类显式参数覆盖，保证子类默认值生效
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句子开头和结束标记？？？
        self.eos_token_id = kwargs.get("eos_token_id", 2)
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否使用Flash Attention
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.intermediate_size = kwargs.get(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )  # FFN中间层维度，，math.ceil()一个向上取整函数
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # 旋转位置编码的基频
        self.tie_word_embeddings = kwargs.get(
            "tie_word_embeddings", True
        )  # 绑定词嵌入，也就是词嵌入和输出层共享权重
        self.inference_rope_scaling = kwargs.get(
            "inference_rope_scaling", False
        )  # 推理时是否启用长度外推
        self.rope_scaling = (
            {  # 外推配置参数
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )
        ### MoE specific configs (ignored if use_moe = False)混合专家模型的参数
        self.num_experts = kwargs.get("num_experts", 4)  # 专家总数
        self.num_experts_per_tok = kwargs.get(
            "num_experts_per_tok", 1
        )  # 每个token激活几个专家
        self.moe_intermediate_size = kwargs.get(
            "moe_intermediate_size", self.intermediate_size
        )  # 每个专家的FFN中间维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind Model
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class RMSNorm(torch.nn.Module):  # 均方根归一化，y=x/RMS(x) *weight
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  #

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # x/RMS(x)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(
    dim: int,
    end: int = int(32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: dict = None,
):
    freqs, attn_factor = (
        1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)),
        1.0,
    )
    if (
        rope_scaling is not None
    ):  # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (
                2 * math.log(rope_base)
            )
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(
                math.ceil(inv_dim(beta_slow)), dim // 2 - 1
            )
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.001),
                0,
                1,
            )
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat(
            (-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1
        )

    q_embed = (
        (q * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    ).to(q.dtype)
    k_embed = (
        (k * cos.unsqueeze(unsqueeze_dim))
        + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    ).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):  # Grouped Query Attention分组查询注意力，8个Q头共享4个KV头
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = (
            config.num_attention_heads
            if config.num_key_value_heads is None
            else config.num_key_value_heads
        )
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个KV头要重复的次数
        self.head_dim = config.head_dim
        self.is_causal = True  # 因果注意力
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )  # # 768 → 8*96 = 768
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )  # 768 → 4*96 = 384
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # 均方根归一化
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and config.flash_attn
        )  # hasattr中object是要检查的目标对象，可以是任意Python对象，name:str是要检查的属性或方法名字，必须是字符串格式

    def forward(
        self,
        x,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)  # Q和K做均方根归一化
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)  # 位置编码
        if past_key_value is not None:  # KV Cache 拼接之前的KV
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 在seq_len维度拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )  # 复制KV
        # 维度变成(bsz, num_heads, seq_len, head_dim)

        # Flash Attention条件判断，环境支持，序列长度大于1，非因果或没有KVCache，没有自定义mask或全为1
        if (
            self.flash
            and (seq_len > 1)
            and (not self.is_causal or past_key_value is None)
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):  # 训练的时候应该使用到FlashAttention,推理时，只有第一步prompt阶段可能用到，
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = (
                self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
            )
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )  # LLaMA系列通常使用SwiGLU激活函数，门控权重从hidden_size到intermediate_size,上投影hidden_size到intermediate_size,逐元素乘法,最后下投影回到hidden_size


class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForward(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = topk_idx == i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(
                    0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype)
                )
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (
                (load * scores.mean(0)).sum()
                * self.config.num_experts
                * self.config.router_aux_loss_coef
            )
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):  # 标准的 Transformer 块（Decoder Block）
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)  # 注意力层
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )  # 均方根归一化层
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = (
            FeedForward(config) if not config.use_moe else MOEFeedForward(config)
        )  # 前馈网络，FFN和MoE

    def forward(
        self,
        hidden_states,
        position_embeddings,
        past_key_value=None,
        use_cache=False,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # 前置归一化，放在子层(注意力/FFN)之前
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )
        hidden_states += residual  # x = x+f(norm(x))
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return (
            hidden_states,
            present_key_value,
        )  # 其实都是前置归一化，这里包含注意力和线性层


# 模型的核心主体，负责把token id转换成隐藏状态，并返回层缓存和辅助损失
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        # 调用父类nn.Module的初始化，这是必须的，它会注册一些内部状态
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        # 嵌入层，词表大小到每个token的隐藏层维度，本质上是一个查找表，内部就是一个大矩阵(vocab_size,hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # nn.ModuleList(modules)包装一个模块列表，让Pytorch知道这些模块是模型的一部分，从而能正确的注册参数等
        self.layers = nn.ModuleList(
            [MiniMindBlock(l, config) for l in range(self.num_hidden_layers)]
        )
        # 均方根归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 预计算旋转位置编码（RoPE）需要的 cos 和 sin 频率表
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # 注册缓冲张量buffer,不是模型参数，但是作为模型状态的一部分
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [None] * len(self.layers)

        # 正常past_key_values最外层是列表，列表里每一项是元组(key,value),key/value都是张量，(batch, seq_len, num_heads, head_dim)
        start_pos = (
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )

        # 嵌入+dropout得到初始隐藏状态
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            # 转移到当前 hidden_states 所在的设备。
            self.freqs_cos, self.freqs_sin = freqs_cos.to(
                hidden_states.device
            ), freqs_sin.to(hidden_states.device)

        # 切片取出对应的RoPE频率表，和本次的输入序列长度seq_len对应
        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        # 这里的layer 就是一个个MiniMindBlock实例
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            # present表示的是当前着一层刚刚计算出来的新key和value张量
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        return hidden_states, presents, aux_loss


# 多重继承，PreTrained是来自transform库的基类提供模型加载保存，设备管理等通用功能，GenerationMixin提供.generate()方法的默认实现，这里我们会重写
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):

    config_class = MiniMindConfig
    # 类属性，表示权重绑定，也就是这两块是同一块权重
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    # 构造函数，config的类型是MiniMindConfig，默认值是None
    def __init__(self, config: MiniMindConfig = None):
        # 如果config是其他假值，就创建默认的MiniMindConfig()
        self.config = config or MiniMindConfig()
        # 调用父类的初始化，把config传进去，完成模型的通用初始化，才能安全地添加自己的子模块
        super().__init__(self.config)

        self.model = MiniMindModel(self.config)  # 创建一个MiniMindModel实例
        # 语言模型头部，输入维度是模型隐藏层大小，输出维度是词表大小
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )
        # 开启权重共享，就是嵌入层和语言模型头的权重一致
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=0,
        labels=None,
        **kwargs
    ):  # inputs_ids 输入的token_id张量，形状通常是(batch_size,sequence_length)，logits_to_keep,只在最后几个位置计算logits
        # 输入形状[batch_size,seq_len],输出形状[batch_size,seq_len,hidden_size]
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            **kwargs,
        )
        # slice是Python的内置函数，用来动态构建切片，例如SFT只看回答的损失
        # 用来决定保留哪些位置的logits,正常训练时logits_to_keep = 0,默认保留全部位置
        # 如果推理时只想要最后一个token的logits,logits_to_keep = 1,因为生成下一个token确实只需要最后一个位置得到logits
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        # 将选中的隐藏状态通过线性层得到logits ,形状为(batch,selected_len,vocab_size)
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        # 训练loss
        if labels is not None:
            # [..., :-1, :]取倒数第二第个之前的所有位置，也就是去掉最后一个,[..., 1:]标签整体右移一位，作为预测目标，.contiguous()确保张量在在内存中连续存储，
            # 训练的时候错位一位
            x, y = (
                logits[..., :-1, :].contiguous(),
                labels[..., 1:].contiguous(),
            )
            # x.view(-1, x.size(-1))把(batch,seq_len-1,vocab_size)展平成(bat*(seq),vocab_size)
            # 如果label里某个位置是-100，这个位置不参与loss计算，这在SFT里会使用到
            loss = F.cross_entropy(
                x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100
            )  # 这里默认reduction: str = "mean"，也就是返回的是平均交叉熵

        # 它把多个返回值打包成一个带有名字书信的对象，方便调用方以属性方式访问如out.loss
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    # https://github.com/jingyaogong/minimind/discussions/611
    # 装饰器，功能类似于torch.no_grad(),但更加轻量，会禁用梯度计算和自动求导，生成时使用
    @torch.inference_mode()
    # 重写的.generate()方法
    def generate(
        self,
        inputs=None,
        attention_mask=None,
        max_new_tokens=8192,
        temperature=0.85,
        top_p=0.85,  # 核采样，只从累计概率前 p 的 token 里采样。
        top_k=50,  # 只从概率最高的 k 个 token 里采样。
        eos_token_id=2,  # 结束 token，生成到它就停止。
        streamer=None,
        use_cache=True,
        num_return_sequences=1,
        do_sample=True,
        repetition_penalty=1.0,  # 重复惩罚，降低已经出现 token 的概率。
        **kwargs
    ):
        # top_p核采样阈值，top_k只保留概率最高的k个token,eos_token_id,结束符的id,生成到这个id就停止，num_return_sequences返回多少个独立序列

        # 想对同一个 prompt 生成多条回答，就复制多份
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = (
            attention_mask.repeat(num_return_sequences, 1)
            if attention_mask is not None
            else None
        )

        # 初始化 KV cache 和 finished
        # 是一个元组，每层一个(key,value)张量
        # past_key_values最外层是一个列表，列表中的每个元素是元组(key,value)，代表每一层的KVcache
        past_key_values = kwargs.pop("past_key_values", None)

        # 布尔张量，形状（batchsize,1),标记每个序列是否生成结束符
        # finished表示batch里每一条序列是否已经生成结束
        finished = torch.zeros(
            input_ids.shape[0], dtype=torch.bool, device=input_ids.device
        )
        if streamer:
            streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):  # 逐个token循环生成
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[
                    :, past_len:
                ],  # input_ids[:, past_len:]第一次forward会处理完整prompt,第二次循环时，只是把新生成的token喂进去
                attention_mask,
                past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            # 更新attention_mask，每新生成一个token就要在mask后面补一个1
            attention_mask = (
                torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones(attention_mask.shape[0], 1),
                    ],
                    -1,
                )
                if attention_mask is not None
                else None
            )

            # 取最后一个位置的logits，形状变为(batch,voc_size)
            # 温度越低，分布越尖锐，越保守
            # 温度越高，分布越平，越随机
            logits = outputs.logits[:, -1, :] / temperature

            # 重复惩罚，如果某些token在前文出现过，就降低它们再次出现的概率
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):  # 每个序列独立处理去重
                    seen = torch.unique(input_ids[i])
                    score = logits[i, seen]
                    logits[i, seen] = torch.where(
                        score > 0,
                        score / repetition_penalty,
                        score * repetition_penalty,
                    )
            if top_k > 0:  # 只保留分数最高的top_k个token,其他token的logits全部设为-inf
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float(
                    "inf"
                )
                # torch.topk(logits, top_k)返回一个元组(values,indices),[0]取values形状(batch,top_k),[..., -1, None]切片取每行的最后一个值，并在最后加一个维度变成(batch,1),然后就是布尔比较和布尔索引

            if top_p < 1.0:  # 核采样过滤
                # 从概率最高的token开始累加，知道累计概率超过0.85，只在这些token里采样
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True
                )  # 返回一个元组，(排序后的值，原始位置的索引)降序
                mask = (
                    torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                )  # torch.cumsum()累积求和，比较操作得到[[False, False, True, True]]
                mask[..., 1:], mask[..., 0] = (
                    mask[..., :-1].clone(),
                    0,
                )  # mask[..., :-1].clone()除最后一个元素之外的所有值，mask[..., 0]第一个元素设为0
                logits[mask.scatter(1, sorted_indices, mask)] = -float("inf")

            # 按照概率分布随机采样一个token，每行采样一个token，形状(batch,1)
            next_token = (
                torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                if do_sample
                else torch.argmax(logits, dim=-1, keepdim=True)
            )

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token,
                )
            # next_token.new_full((next_token.shape[0], 1), eos_token_id)创建一个形状[B,1]的张量，里面全部填eos_token_id,finish=True,就填结束符，如果=False,就保留原本预测的next_token

            input_ids = torch.cat(
                [input_ids, next_token], dim=-1
            )  # 把新token 拼接到序列末尾
            past_key_values = (
                outputs.past_key_values if use_cache else None
            )  # 更新KV缓存
            if streamer:
                streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(
                    eos_token_id
                )  # 压缩最后一维，逐个元素判断是否等于结束符,然后按照位或，更新finish标志
                if finished.all():
                    break  # 判断所有句子是否都True
        if streamer:
            streamer.end()
        if kwargs.get("return_kv"):
            return {"generated_ids": input_ids, "past_kv": past_key_values}
        return input_ids
