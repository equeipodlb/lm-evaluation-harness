# hf_causal_perlayer.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch, hashlib
from collections import OrderedDict

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM

def _parse_model_args(s: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not s:
        return out
    for kv in s.split(","):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out

@register_model("hf-causal-perlayer")
class HFCausalPerLayer(HFLM):
    """
    HuggingFace Causal LM whose logits are replaced by a LogitLens-style
    readout from a chosen hidden layer (apply final LN, project with W_U).
    Supports caching hidden states so multiple layers reuse one forward.
    """

    AUTO_MODEL_CLASS = AutoModelForCausalLM

    def __init__(self, **kwargs: Any):
        parsed = {}
        if "model_args" in kwargs and kwargs["model_args"]:
            parsed = _parse_model_args(kwargs["model_args"])

        pretrained = kwargs.get("pretrained", parsed.get("pretrained"))
        if not pretrained:
            raise ValueError("You must supply `pretrained=...` (either directly or in model_args).")

        layer = kwargs.get("layer", parsed.get("layer", -1))
        try:
            layer = int(layer)
        except Exception:
            layer = -1

        # accept device/dtype/attn_implementation from either kwargs or model_args
        batch_size = kwargs.get("batch_size", 1)
        device     = kwargs.get("device", parsed.get("device", None))
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        dtype      = kwargs.get("dtype", parsed.get("dtype", None))
        # strings like "float16", "bfloat16", "float32" are okay for HFLM
        if dtype is None and torch.cuda.is_available():
            dtype = "float16"

        attn_impl  = kwargs.get("attn_implementation", parsed.get("attn_implementation", None))

        # caching knobs
        self.cache_hidden = bool(kwargs.get("cache_hidden", parsed.get("cache_hidden", True)))
        self.cache_to_cpu = bool(kwargs.get("cache_to_cpu", parsed.get("cache_to_cpu", True)))
        self._hs_cache_max = int(kwargs.get("hs_cache_max", parsed.get("hs_cache_max", 256)))
        self._hs_cache: OrderedDict[str, tuple[torch.Tensor,...]] = OrderedDict()

        # Let base class load
        super_kwargs = dict(pretrained=pretrained, batch_size=batch_size, device=device, dtype=dtype)
        if attn_impl is not None:
            super_kwargs["attn_implementation"] = attn_impl
        super().__init__(**super_kwargs)

        self.layer_idx = layer
        self._setup_readout()

    
    def set_layer(self, layer: int):
        self.layer_idx = int(layer)

    # ----- internal: cache key for a batch
    def _batch_key(self, inps: torch.Tensor, attn_mask: torch.Tensor | None):
        h = hashlib.sha1()
        h.update(inps.detach().cpu().numpy().tobytes())
        if attn_mask is not None:
            h.update(attn_mask.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def _setup_readout(self):
        m = self._model

        # final layer norm
        ln_f = None
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "final_layer_norm"):
            ln_f = m.gpt_neox.final_layer_norm
        elif hasattr(m, "transformer") and hasattr(m.transformer, "ln_f"):
            ln_f = m.transformer.ln_f
        elif hasattr(m, "model") and hasattr(m.model, "norm"):
            ln_f = m.model.norm
        elif hasattr(m, "ln_f"):
            ln_f = m.ln_f
        elif hasattr(m, "final_layer_norm"):
            ln_f = m.final_layer_norm

        # output projection (unembedding)
        lm_head = getattr(m, "lm_head", None) or getattr(m, "embed_out", None) or getattr(m, "score", None)
        if lm_head is None and hasattr(m, "get_output_embeddings"):
            lm_head = m.get_output_embeddings()

        if ln_f is None or lm_head is None:
            raise RuntimeError(
                f"Could not find final_layer_norm and/or output head on model {type(m).__name__}."
            )

        self._ln_f = ln_f
        self._W_U  = lm_head.weight
        self._b_U  = getattr(lm_head, "bias", None)

        # infer number of layers
        with torch.inference_mode():
            tmp = torch.ones((1, 3), dtype=torch.long, device=m.device)
            hs = m(tmp, output_hidden_states=True, use_cache=False).hidden_states
            self._L = len(hs) - 1
        if self.layer_idx < 0:
            self.layer_idx = self._L - 1

    @torch.inference_mode()
    def _get_hidden_states(self, inps: torch.Tensor, attention_mask=None):
        if not self.cache_hidden:
            return self._model(inps, attention_mask=attention_mask,
                               output_hidden_states=True, use_cache=False).hidden_states

        key = self._batch_key(inps, attention_mask)
        hs = self._hs_cache.get(key)
        if hs is None:
            out = self._model(inps, attention_mask=attention_mask,
                              output_hidden_states=True, use_cache=False)
            hs = tuple(t.detach().cpu() if self.cache_to_cpu else t.detach() for t in out.hidden_states)
            self._hs_cache[key] = hs
            # LRU evict
            if len(self._hs_cache) > self._hs_cache_max:
                self._hs_cache.popitem(last=False)
        return hs

    @torch.inference_mode()
    def _model_call(self, inps: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return (B, T, V) logits computed from the chosen layer’s hidden states.
        lm-eval will handle loss/metrics on top.
        """
        attention_mask = kwargs.get("attention_mask", None)
        hs = self._get_hidden_states(inps, attention_mask)
        h = hs[self.layer_idx + 1]
        if self.cache_to_cpu:
            h = h.to(self._W_U.device)
        if self._ln_f is not None:
            h = self._ln_f(h)
        logits = torch.matmul(h.to(self._W_U.dtype), self._W_U.T)
        if self._b_U is not None:
            logits = logits + self._b_U
        return logits.to(dtype=torch.float32)
