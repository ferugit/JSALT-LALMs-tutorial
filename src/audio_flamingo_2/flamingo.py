import torch
from einops import rearrange
from torch import nn
import torch.nn.utils as utils

from torch.distributed.fsdp.wrap import (
    enable_wrap,
    wrap,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

try:
    from .helpers import TransformerEncoder
    from .utils import apply_with_stopping_condition
except:
    from audio_flamingo_2.helpers import TransformerEncoder
    from audio_flamingo_2.utils import apply_with_stopping_condition


class Flamingo(nn.Module):
    def __init__(
        self,
        clap,
        unfreeze_clap,
        lang_encoder: nn.Module,
        eoc_token_id: int,
        media_token_id: int,
        sep_token_id: int,
        clap_embed_dim: int,
        audio_transformer_kwargs: dict,
        cross_attn_every_n_layers: int = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.eoc_token_id = eoc_token_id
        self.media_token_id = media_token_id
        self.sep_token_id = sep_token_id 

        # initialize embedding dimensions
        self.clap_embed_dim = clap_embed_dim

        # Initilaize CLAP
        self.clap = clap
        self.unfreeze_clap = unfreeze_clap
        self.clap.requires_grad_(unfreeze_clap)

        if hasattr(lang_encoder.config, "d_model"):
            self.lang_dim = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.lang_dim = lang_encoder.config.hidden_size

        # define just one type of audio transformer
        n_head = audio_transformer_kwargs["n_head"]
        n_layers = audio_transformer_kwargs["n_layers"]
        d_inner = audio_transformer_kwargs["d_inner"]
        max_num_media = audio_transformer_kwargs["max_num_media"]
        max_window_per_audio = audio_transformer_kwargs["max_window_per_audio"]
        
        common_encoder_embed_dim = clap_embed_dim

        # define the transformers
        assert common_encoder_embed_dim % n_head == 0
        self.audio_transformer_clap = TransformerEncoder(
            d_word_vec=common_encoder_embed_dim, 
            n_layers=n_layers, 
            n_head=n_head, 
            d_k=common_encoder_embed_dim // n_head, 
            d_v=common_encoder_embed_dim // n_head,
            d_model=common_encoder_embed_dim, 
            d_inner=d_inner, 
            dropout=0.0, 
            n_position=max_num_media, 
            scale_emb=True
        )
        
        self.lang_encoder = lang_encoder
        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_dim,
            audio_hidden_size=common_encoder_embed_dim,
            max_window_per_audio=max_window_per_audio,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            gradient_checkpointing=gradient_checkpointing,
        )

        self._use_gradient_checkpointing = gradient_checkpointing
        
        # enable gradient checkpoint for the audio transformers
        self.audio_transformer_clap._use_gradient_checkpointing = gradient_checkpointing

        # enable gradient checkpoint for encoders
        self.clap._use_gradient_checkpointing = gradient_checkpointing


    def forward(
        self,
        audio_x: torch.Tensor,
        audio_x_mask: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        clear_conditioned_layers: bool = True,
        past_key_values=None,
        use_cache: bool = False,
    ):
        assert (
            self.lang_encoder.initialized_flamingo
        ), "Flamingo layers are not initialized. Please call `init_flamingo` first."

        assert (
            self.lang_encoder._use_cached_audio_x or audio_x is not None
        ), "Must provide either audio_x or have precached media using cache_media()."

        if self.lang_encoder._use_cached_audio_x:
            assert (
                audio_x is None
            ), "Expect audio_x to be None when media has been cached using cache_media(). Try uncache_media() first."
            assert self.lang_encoder.is_conditioned()

        else:
            self._encode_audio_x(audio_x=audio_x, audio_x_mask=audio_x_mask)
            self._condition_media_locations(input_ids=lang_x)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def generate(
        self,
        audio_x: torch.Tensor,
        audio_x_mask: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs,
    ):
        num_beams = kwargs.pop("num_beams", 1)
        if num_beams > 1:
            audio_x = audio_x.repeat_interleave(num_beams, dim=0)

        self.lang_encoder._use_cached_audio_x = True
        self._encode_audio_x(audio_x=audio_x, audio_x_mask=audio_x_mask)

        eos_token_id = kwargs.pop("eos_token_id", self.eoc_token_id)
        output = self.lang_encoder.generate(
            input_ids=lang_x,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            num_beams=num_beams,
            **kwargs,
        )

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_audio_x = False
        return output

    def _encode_audio_x(self, audio_x: torch.Tensor, audio_x_mask: torch.Tensor):
        """
        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert audio_x.ndim == 3, "audio_x should be of shape (B, num_window, window_length)"

        #------------------------------------------------------------------------#
        # get embeddings from CLAP
        audio_embeds = self.clap(audio_x)
        B, L, H, D = audio_embeds.shape  # L is number of windows, D is feature dim
        assert D == self.clap_embed_dim

        audio_x_out = rearrange(audio_embeds, 'b l h d -> b (l h) d')
        # handle the masks
        expanded_speech_mask = audio_x_mask.repeat_interleave(H, dim=1) # B, (LxH)
        if B > 1 and expanded_speech_mask.shape[0] == 1:
            expanded_speech_mask = expanded_speech_mask.repeat(B, 1)
        assert expanded_speech_mask.shape[0] == B and expanded_speech_mask.shape[1] == L*H, "{} != ({},{})".format(expanded_speech_mask.shape, B, L*H)

        #------------------------------------------------------------------------#
        audio_x_out = self.audio_transformer_clap(audio_x_out, causal_mask = expanded_speech_mask)  # B, LxH, D

        # Unsqueeze to handle Falmingo code
        audio_x_out = audio_x_out.unsqueeze(2)  # B, L, n=1, D
        audio_x_mask = expanded_speech_mask.unsqueeze(2)  # B, L, n=1

        #------------------------------------------------------------------------#

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_audio_x(audio_x_out, audio_x_mask)

    def wrap_fsdp(self, wrapper_kwargs, device_id):
        # unfreeze the decoder layers
        for block in self.lang_encoder.old_decoder_blocks:
            block.requires_grad_(True)

        # wrap in FSDP
        with enable_wrap(wrapper_cls=FSDP, **wrapper_kwargs):

            self.audio_transformer_clap = wrap(wrap(self.audio_transformer_clap))

            self.lang_encoder.old_decoder_blocks = nn.ModuleList(
                wrap(wrap(block)) for block in self.lang_encoder.old_decoder_blocks
            )
            self.lang_encoder.gated_cross_attn_layers_sound = nn.ModuleList(
                wrap(wrap(layer)) if layer is not None else None
                for layer in self.lang_encoder.gated_cross_attn_layers_sound
            )
            self.lang_encoder.init_flamingo_layers(self._use_gradient_checkpointing)
            self.lang_encoder.set_input_embeddings(
                wrap(wrap(self.lang_encoder.get_input_embeddings()))
            )

            if hasattr(self.lang_encoder, 'set_output_embeddings'):
                self.lang_encoder.set_output_embeddings(
                    wrap(wrap(self.lang_encoder.get_output_embeddings()))
                )
            else:
                print('skip wrapping output embeddings')

        # manually move non-FSDP managed parameters to device_id
        # these are all in lang_encoder
        apply_with_stopping_condition(
            module=self.lang_encoder,
            apply_fn=lambda m: m.to(device_id),
            apply_condition=lambda m: len(list(m.children())) == 0,
            stopping_condition=lambda m: isinstance(m, FSDP),
        )

        # clap shouldn't be wrapped; should be on each gpu
        if self.unfreeze_clap:
            apply_with_stopping_condition(
                module=self.clap,
                apply_fn=lambda m: m.to(device_id),
                apply_condition=lambda m: len(list(m.children())) == 0,
                stopping_condition=lambda m: isinstance(m, FSDP),
            )

        # exclude the original decoder layers from the optimizer
        for block in self.lang_encoder.old_decoder_blocks:
            for p in block.parameters():
                p.exclude_from_optimizer = True

        # set up clip_grad_norm_ function
        def clip_grad_norm_(max_norm):

            utils.clip_grad_norm_(self.clap.nvclap.parameters(), max_norm=max_norm)

            self.audio_transformer_clap.clip_grad_norm_(max_norm)
            for layer in self.lang_encoder.gated_cross_attn_layers_sound:
                if layer is not None:
                    layer.clip_grad_norm_(max_norm)
            self.lang_encoder.get_input_embeddings().clip_grad_norm_(max_norm)

        self.clip_grad_norm_ = clip_grad_norm_

    def _condition_media_locations(self, input_ids: torch.Tensor):
        media_locations = (input_ids == self.media_token_id)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_media_locations(media_locations)

    def cache_media(self, input_ids: torch.Tensor, audio_x: torch.Tensor, audio_x_mask: torch.Tensor):
        self._encode_audio_x(audio_x=audio_x, audio_x_mask=audio_x_mask)
        self._condition_media_locations(input_ids=input_ids)
        self.lang_encoder._use_cached_audio_x = True

    def uncache_media(self):
        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_audio_x = False
