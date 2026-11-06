"""HiRaMem with 64 causal future-depth queries before autoregressive actions.

The local base implementation is checkpointed with this package. Depth targets
are used only by the loss; queries and memory never consume action answers.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from .configuration_prismatic import OpenVLAConfig
from .modeling_hiramem_base import (
    OpenVLAForActionPrediction as HiRaMemForActionPrediction,
    PrismaticCausalLMOutputWithPast,
)
from .future_depth import (FutureDepthAux, FutureDepthHead, depth_loss_terms,
                           pack_depth_sequence, prompt_masks)


@dataclass
class FutureDepthOutput(PrismaticCausalLMOutputWithPast):
    action_sum: Optional[torch.Tensor] = None
    action_count: Optional[torch.Tensor] = None
    depth_reg_sum: Optional[torch.Tensor] = None
    depth_reg_count: Optional[torch.Tensor] = None
    depth_edge_sum: Optional[torch.Tensor] = None
    depth_edge_count: Optional[torch.Tensor] = None
    depth_valid_count: Optional[torch.Tensor] = None
    depth_log: Optional[torch.Tensor] = None


class OpenVLAForActionPrediction(HiRaMemForActionPrediction):
    config_class = OpenVLAConfig
    requires_future_depth = True
    auxiliary_module_names = ('explore_model', 'trajectory_memory', 'future_aux')

    def __init__(self, config):
        super().__init__(config)
        self.future_aux = FutureDepthAux(config.text_config.hidden_size)
        self.future_aux.apply(self._init_weights)

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, FutureDepthAux):
            module.reset_queries()
        if isinstance(module, FutureDepthHead):
            module.reset_output()

    def forward_memory_window(self, steps):
        if not 1 <= len(steps) <= 2:
            raise ValueError('TBPTT windows must contain one or two consecutive steps')
        outputs = []
        for batch in steps:
            for key in ('future_depth', 'future_depth_mask', 'future_valid'):
                if key not in batch:
                    raise ValueError(f'Depth training window is missing {key}')
            outputs.append(self.forward(**batch, return_dict=True, use_cache=False))
            for key, is_last in zip(batch['episode_ids'], batch['episode_ends']):
                if bool(is_last):
                    self.reset_memory(key)
        def summed(name):
            return torch.stack([getattr(out, name) for out in outputs]).sum()
        # Separate denominators: action tokens, weighted valid pixels, valid edges.
        terms = {
            'action': (summed('action_sum'), summed('action_count'), 1.0),
            'depth_reg': (summed('depth_reg_sum'), summed('depth_reg_count'), self.config.depth_loss_weight),
            'depth_edge': (summed('depth_edge_sum'), summed('depth_edge_count'),
                           self.config.depth_loss_weight * self.config.depth_edge_weight),
        }
        loss = sum(weight * numerator / denominator.clamp_min(1e-6)
                   for numerator, denominator, weight in terms.values())
        return {'loss': loss, 'token_count': terms['action'][1], 'loss_terms': terms,
                'depth_valid_count': summed('depth_valid_count'),
                'sample_count': loss.new_tensor(sum(len(b['episode_ids']) for b in steps))}

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None,
                inputs_embeds=None, past_key_values=None, camera_images=None,
                camera_depths=None, camera_poses=None, camera_params=None,
                camera_image_features=None, target_poses=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                output_projector_features=None, return_dict=None, memory_window=None,
                episode_ids=None, step_ids=None, episode_ends=None, prompt_lengths=None,
                future_depth=None, future_depth_mask=None, future_valid=None,
                return_depth=False, cache_position=None):
        if memory_window is not None:
            return self.forward_memory_window(memory_window)
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        cache_default = getattr(self.config, 'use_cache', self.config.text_config.use_cache)
        use_cache = bool(cache_default if use_cache is None else use_cache) and not self.training and labels is None
        # The original wrapper intentionally derives positions from the actual
        # multimodal cache, not GenerationMixin's shorter text-only positions.
        cache_length = 0
        if past_key_values is not None:
            cache_length = (past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length')
                            else past_key_values[0][0].shape[-2])
        if cache_length:
            if labels is not None or input_ids is None or input_ids.shape != (1, 1):
                raise ValueError('Cached action decoding expects one token, one episode, and no labels')
            output = self.language_model(
                input_ids=input_ids, past_key_values=past_key_values, attention_mask=None,
                use_cache=use_cache, return_dict=True, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            result = FutureDepthOutput(logits=output.logits, past_key_values=output.past_key_values,
                                       hidden_states=output.hidden_states, attentions=output.attentions)
            return result if return_dict else result.to_tuple()
        if pixel_values is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   use_cache=use_cache, return_dict=return_dict,
                                   output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        if input_ids is None or inputs_embeds is not None:
            raise ValueError('Multimodal prefill requires input_ids, not preassembled inputs_embeds')
        batch_size = len(input_ids)
        if use_cache and batch_size != 1:
            raise ValueError('Cached generation currently supports one episode at a time')
        if episode_ids is None or step_ids is None or len(episode_ids) != batch_size or len(step_ids) != batch_size:
            raise ValueError('Every observation needs episode_ids and consecutive step_ids')
        if len(set(map(str, episode_ids))) != batch_size:
            raise ValueError('A batch must contain distinct episodes')
        if len(pixel_values) != batch_size:
            raise ValueError('Image and text batch sizes differ')
        if any(x is None or len(x) != batch_size for x in (camera_depths, camera_params, camera_poses)):
            raise ValueError('Camera geometry must match the batch')
        observation = camera_image_features if camera_image_features is not None else camera_images
        if observation is None or len(observation) != batch_size:
            raise ValueError('Supply one camera frame or feature grid per episode')
        attention_mask = input_ids.ne(self.pad_token_id) if attention_mask is None else attention_mask.bool()
        text = self.get_input_embeddings()(input_ids)
        language_mask, _ = prompt_masks(attention_mask, labels, prompt_lengths)
        language = (text * language_mask[..., None]).sum(1) / language_mask.sum(1, keepdim=True)
        pixels = pixel_values.to(dtype=next(self.vision_backbone.parameters()).dtype)
        current_rgb = self.encode_images(pixels)
        entries = []
        for i in range(batch_size):
            if any(len(x[i]) != 1 for x in (observation, camera_depths, camera_params, camera_poses)):
                raise ValueError('Memory must receive exactly one current observation per policy step')
            images = None if camera_images is None else camera_images[i][None].to(pixels.dtype)
            features = None if camera_image_features is None else camera_image_features[i][None]
            entries.append(self.prepare_explore_rgbd_memory(
                images, camera_depths[i][None], camera_params[i][None], camera_poses[i][None],
                image_features=features, episode_id=episode_ids[i], step_id=int(step_ids[i]),
            ))
        memory_tokens = self.explore_model.forward_memory_batch(entries, language)
        visual = [torch.cat((current_rgb[i], memory_tokens[i])) for i in range(batch_size)]
        packed = pack_depth_sequence(text, visual, attention_mask, labels, prompt_lengths, self.future_aux)
        if packed.embeddings.shape[1] > self.config.text_config.max_position_embeddings:
            raise ValueError('Multimodal/query sequence exceeds the language model context limit')
        body = self.language_model.get_decoder()(
            inputs_embeds=packed.embeddings, attention_mask=packed.attention_mask,
            position_ids=packed.position_ids, use_cache=use_cache, return_dict=True,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        )
        hidden = body.last_hidden_state
        # Preserve the familiar logits contract; CE only materializes FP32
        # vocabulary logits at the supervised action/EOS positions.
        logits = self.get_output_embeddings()(hidden)
        zero = hidden.sum().float() * 0.
        action_sum, action_count = zero, zero.detach()
        if labels is not None:
            targets = packed.labels[:, 1:]
            supervised = targets.ne(-100)
            action_count = supervised.sum().float()
            if not bool(action_count):
                raise ValueError('No supervised action tokens')
            action_sum = F.cross_entropy(logits[:, :-1][supervised].float(), targets[supervised], reduction='sum')
        reg_sum = edge_sum = zero
        reg_count = edge_count = valid_count = zero.detach()
        depth_log = None
        if future_depth is not None or return_depth:
            rows = torch.arange(batch_size, device=hidden.device)[:, None]
            depth_hidden = hidden[rows, packed.depth_positions]
            depth_log = self.future_aux.head(depth_hidden)
            if future_depth is not None:
                if future_depth_mask is None or future_valid is None:
                    raise ValueError('Future depth labels require explicit pixel and transition masks')
                reg_sum, reg_count, edge_sum, edge_count, valid_count = depth_loss_terms(
                    depth_log, future_depth, future_depth_mask, future_valid,
                    min_depth=self.config.depth_min, max_depth=self.config.depth_max,
                    near_distance=self.config.depth_near_distance, near_weight=self.config.depth_near_weight,
                    huber_delta=self.config.depth_huber_delta,
                )
        loss = None
        if labels is not None:
            loss = action_sum / action_count.clamp_min(1.) + self.config.depth_loss_weight * (
                reg_sum / reg_count.clamp_min(1e-6)
                + self.config.depth_edge_weight * edge_sum / edge_count.clamp_min(1e-6))
        result = FutureDepthOutput(
            loss=loss, logits=logits, past_key_values=body.past_key_values,
            hidden_states=body.hidden_states, attentions=body.attentions,
            action_sum=action_sum, action_count=action_count,
            depth_reg_sum=reg_sum, depth_reg_count=reg_count,
            depth_edge_sum=edge_sum, depth_edge_count=edge_count,
            depth_valid_count=valid_count, depth_log=depth_log,
        )
        return result if return_dict else result.to_tuple()

    def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None, **kwargs):
        if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
            if past_key_values.get_seq_length() == 0:
                past_key_values = None
        result = super().prepare_inputs_for_generation(input_ids=input_ids,
                                                       past_key_values=past_key_values, **kwargs)
        result['return_depth'] = kwargs.get('return_depth', False)
        return result

    def predict_action(self, input_ids=None, unnorm_key=None, **kwargs):
        # Keep the legacy action vocabulary and decoding. Fix the optional
        # whitespace append so input_ids and the text mask always agree.
        if kwargs.get('num_beams', 1) != 1 or kwargs.get('num_return_sequences', 1) != 1:
            raise ValueError('Memory generation supports one sequence with num_beams=1')
        if input_ids is not None and not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat((input_ids, input_ids.new_full((len(input_ids), 1), 29871)), 1)
            if kwargs.get('attention_mask') is not None:
                mask = kwargs['attention_mask']
                kwargs['attention_mask'] = torch.cat((mask, mask.new_ones((len(mask), 1))), 1)
        if kwargs.get('use_cache', True) is False:
            raise ValueError('Depth policy generation requires use_cache=True')
        kwargs['use_cache'] = True
        return super().predict_action(input_ids=input_ids, unnorm_key=unnorm_key, **kwargs)
