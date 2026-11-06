
import torch
import torch.nn as nn
import time
from torch_scatter import scatter_mean
from .multimodal_blip2.configuration_blip_2 import Blip2QFormerConfig
from .multimodal_blip2.modeling_blip_2 import Blip2QFormerModel
from typing import Any, Optional, Tuple, Union


class ExploraTokenModel(nn.Module):
    def __init__(self,num_query_tokens=64,num_frontier_tokens=32,qformer_hidden_size=768,vision_hidden_size=768,num_attention_heads=16,llm_dim=4096):
        super().__init__()
    
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, qformer_hidden_size))
        self.frontier_tokens =  nn.Parameter(torch.zeros(1, num_frontier_tokens, qformer_hidden_size))
        self.null_memory = nn.Parameter(torch.zeros(1, 1, vision_hidden_size))
        
        qformer_config = Blip2QFormerConfig(
            hidden_size=qformer_hidden_size,
            num_attention_heads=num_attention_heads,
            encoder_hidden_size = vision_hidden_size,
            num_hidden_layers = 16,
        )
        self.scale = nn.Linear(llm_dim, qformer_hidden_size)
        self.shift = nn.Linear(llm_dim, qformer_hidden_size)
        
        self.qformer = Blip2QFormerModel(qformer_config)
        self.qformer_projector = nn.Sequential(
            nn.Linear(qformer_hidden_size, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        total_params = sum(p.numel() for p in self.qformer.parameters())
        self.init_weights()
        
    def init_weights(self):
        def _init_weights(module):
            """Initialize the weights"""
            factor = 0.02
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Embedding) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=factor)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        self.qformer.apply(_init_weights)
        
        self.qformer_projector.apply(_init_weights)
        nn.init.normal_(self.query_tokens, std=0.02)
        nn.init.normal_(self.frontier_tokens, std=0.02)
        nn.init.normal_(self.null_memory, std=0.02)
        
    def forward(self,
                image_embeds,
                average_language_embedding,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
        ):
        
        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        gamma = self.scale(average_language_embedding)  # (batch_size, vision_dim)
        beta = self.shift(average_language_embedding)  # (batch_size, vision_dim)

        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_tokens = query_tokens * (1 + gamma.view(gamma.shape[0], 1, gamma.shape[1])) + beta.view(beta.shape[0], 1, beta.shape[1])
    
    
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        query_output = query_outputs[0]
        query_output = self.qformer_projector(query_output)
        return query_output

    def forward_memory_batch(self, memory_entries, average_language_embeddings):
        """Read independent episode memories with one padded Q-Former batch.

        Entries contain (tokens, world_xyz, fine_mask, coarse_mask, query_anchor).
        Padding is excluded from both query groups. pad_sequence preserves the
        recurrent graph, including the preceding step in a TBPTT window.
        """
        if not memory_entries:
            raise ValueError("Cannot read an empty batch of episodes")
        if len(memory_entries) != average_language_embeddings.shape[0]:
            raise ValueError("Memory and language batch sizes must match")
        tokens, xyz, fine, coarse, anchors = zip(*memory_entries)
        from torch.nn.utils.rnn import pad_sequence
        memory_feats = pad_sequence(tokens, batch_first=True, padding_value=0.0)
        memory_xyz = pad_sequence(xyz, batch_first=True, padding_value=0.0)
        fine_mask = pad_sequence(fine, batch_first=True, padding_value=False)
        coarse_mask = pad_sequence(coarse, batch_first=True, padding_value=False)
        query_count = self.query_tokens.shape[1] + self.frontier_tokens.shape[1]
        query_xyz = torch.stack(anchors).float()[:, None, :].expand(-1, query_count, -1)
        return self.forward_exp(
            memory_feats, average_language_embeddings, fine_mask, coarse_mask,
            memory_world_xyz=memory_xyz, query_world_xyz=query_xyz,
        )

    def forward_exp(
                self,
                memory_feats,
                average_language_embedding,
                fine_mask,
                coarse_mask,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_split: bool = False,
                memory_world_xyz=None,
                query_world_xyz=None,
        ):
        batch_size = memory_feats.shape[0]
        memory_shape = memory_feats.shape[:2]
        
        num_scene_queries = self.query_tokens.shape[1]
        num_explore_queries = self.frontier_tokens.shape[1]
        # num_global_scene_queries = min(16, num_scene_queries) 
        

        gamma = self.scale(average_language_embedding)
        beta = self.shift(average_language_embedding)

        scene_queries = self.query_tokens.expand(batch_size, -1, -1)
        scene_queries = scene_queries * (1 + gamma[:, None, :]) + beta[:, None, :]
        explore_queries = self.frontier_tokens.expand(batch_size, -1, -1)
        explore_queries = explore_queries * (1 + gamma[:, None, :]) + beta[:, None, :]
        query_tokens = torch.cat([scene_queries, explore_queries], dim=1)

        
        fine_mask = fine_mask.to(device=memory_feats.device, dtype=torch.bool)
        coarse_mask = coarse_mask.to(device=memory_feats.device, dtype=torch.bool)
        if fine_mask.dim() != 2 or coarse_mask.dim() != 2:
            raise ValueError("fine_mask and coarse_mask must have shape [B, Nv].")
        if fine_mask.shape != memory_shape or coarse_mask.shape != memory_shape:
            raise ValueError("fine_mask/coarse_mask shape must match memory_feats[:, :, 0].")

        null_memory = self.null_memory.expand(batch_size, -1, -1).to(dtype=memory_feats.dtype)
        # Preserve updater/embedding graph participation even for empty geometric memory.
        null_memory = null_memory + memory_feats.sum() * 0.0
        memory_feats = torch.cat([memory_feats, null_memory], dim=1)
        null_valid = torch.ones(batch_size, 1, dtype=torch.bool, device=memory_feats.device)
        fine_mask = torch.cat([fine_mask, null_valid], dim=1)
        coarse_mask = torch.cat([coarse_mask, null_valid], dim=1)

        # Without frontier queries, scene queries read both levels so coarse memory stays reachable.
        scene_mask = fine_mask if num_explore_queries > 0 else (fine_mask | coarse_mask)
        scene_cross_mask = scene_mask[:, None, :].expand(-1, num_scene_queries, -1)
        explore_cross_mask = coarse_mask[:, None, :].expand(-1, num_explore_queries, -1)
        cross_mask = torch.cat([scene_cross_mask, explore_cross_mask], dim=1).to(dtype=memory_feats.dtype)

        if query_world_xyz is None or memory_world_xyz is None:
            raise ValueError("Hierarchical memory requires query and memory world coordinates")
        memory_world_xyz = torch.cat([memory_world_xyz, query_world_xyz[:, :1]], dim=1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=memory_feats,
            encoder_attention_mask=cross_mask,
            query_world_xyz=query_world_xyz,
            memory_world_xyz=memory_world_xyz,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )
        query_output = self.qformer_projector(query_outputs[0])

        if return_split:
            return (
                query_output[:, :num_scene_queries, :],
                query_output[:, num_scene_queries:, :],
            )
        return query_output
