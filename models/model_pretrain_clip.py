'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertForMaskedLM, BertPrefixModel
from models.visual_transformers import initialize_clip

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 text_decoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        self.visual_encoder, _ = initialize_clip(config)
        # self.visual_encoder = VisionTransformer(
        #     img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
        #     mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #
        # if init_deit:
        #     checkpoint = torch.hub.load_state_dict_from_url(
        #         url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #         map_location="cpu", check_hash=True)
        #     state_dict = checkpoint["model"]
        #     pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
        #     state_dict['pos_embed'] = pos_embed_reshaped
        #     msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
        #     print(msg)

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(text_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.large = False
        if bert_config.hidden_size != 768:
            self.visn_fc = nn.Linear(vision_width, bert_config.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(bert_config.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
            self.large = True

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models
        self.visual_encoder_m, _ = initialize_clip(config)
        self.vision_proj_m = nn.Linear(text_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        if bert_config.hidden_size != 768:
            self.visn_fc_m = nn.Linear(vision_width, bert_config.hidden_size)
            self.visn_layer_norm_m = nn.LayerNorm(bert_config.hidden_size, eps=1e-12)
            self.dropout_m = nn.Dropout(bert_config.hidden_dropout_prob)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m]
                            ]
        if config['prefix_task']:
            config_decoder = BertConfig.from_json_file(config['bert_config'])
            config_decoder.fusion_layer = 0
            config_decoder.cross_layer = 0
            config_decoder.num_hidden_layers = 12
            self.text_decoder = BertPrefixModel.from_pretrained(text_decoder, config=config_decoder)    
            self.text_decoder_m = BertPrefixModel.from_pretrained(text_decoder, config=config_decoder)   

            self.model_pairs.append([self.text_decoder, self.text_decoder_m])

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.mid_cross = config['mid_cross']
        self.merge_attention = config['merge_attention']
        self.distill = config['distill']
        self.mlm_task = config['mlm_task']
        self.itm_task = config['itm_task']
        self.ita_task = config['ita_task']
        self.prefix_task = config['prefix_task']
        self.concat_last_layer = config['concat_last_layer']

    def forward(self, image, text, alpha=0, prefix_input=None, prefix_target=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m.visual(image, skip_last_layer=True)
            if self.large:
                image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        if self.merge_attention:
            merge_embeds = torch.cat([text_embeds, image_embeds], 1)
            merge_attention_mask = torch.cat([text.attention_mask, image_atts], 1) 
            output_pos = self.text_encoder.bert(encoder_embeds=merge_embeds,
                                            attention_mask=merge_attention_mask,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        else:
            output_pos = self.text_encoder.bert(encoder_embeds=text_embeds,
                                            attention_mask=text.attention_mask,
                                            encoder_hidden_states=image_embeds,
                                            encoder_attention_mask=image_atts,
                                            return_dict=True,
                                            mode='fusion',
                                            )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)
        if self.merge_attention:
            merge_text_embeds_all = torch.cat([text_embeds_all, image_embeds_all], 1)
            merge_text_atts_all = torch.cat([text_atts_all, image_atts_all], 1) 
            output_neg = self.text_encoder.bert(encoder_embeds=merge_text_embeds_all,
                                            attention_mask=merge_text_atts_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        else:
            output_neg = self.text_encoder.bert(encoder_embeds=text_embeds_all,
                                            attention_mask=text_atts_all,
                                            encoder_hidden_states=image_embeds_all,
                                            encoder_attention_mask=image_atts_all,
                                            return_dict=True,
                                            mode='fusion',
                                            )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix)

        if self.mid_cross:
            if self.distill:
                with torch.no_grad():
                    text_output_m = self.text_encoder_m.bert(input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
                    text_embeds_m = text_output_m.last_hidden_state
                    if self.merge_attention:
                        merge_text_embeds_m = torch.cat([text_embeds_m, image_embeds_m], 1)
                        logits_m = self.text_encoder_m(encoder_embeds=merge_text_embeds_m,
                                           attention_mask=merge_attention_mask,
                                           return_dict=True,
                                           return_logits=True,
                                           mode='fusion'
                                           )
                    else:
                        logits_m = self.text_encoder_m(encoder_embeds=text_embeds_m,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           mode='fusion'
                                           )
                soft_labels = F.softmax(logits_m, dim=-1)
            else:
                soft_labels = None
            text_output = self.text_encoder.bert(input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_embeds = text_output.last_hidden_state
            if self.merge_attention:
                merge_embeds = torch.cat([text_embeds, image_embeds], 1) 
                image_labels = -100 * torch.ones([image_embeds.shape[0], image_embeds.shape[1]], dtype=labels.dtype, device=labels.device) 
                labels = torch.cat([labels, image_labels], 1) 
                mlm_output = self.text_encoder(encoder_embeds=merge_embeds,
                                       attention_mask=merge_attention_mask,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=soft_labels,
                                       alpha=alpha,
                                       mode='fusion'
                                       )
            else:
                mlm_output = self.text_encoder(encoder_embeds=text_embeds,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=soft_labels,
                                       alpha=alpha,
                                       mode='fusion'
                                       )
            loss_mlm = mlm_output.loss
        else:
            with torch.no_grad():
                logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           )
            mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                       )
            loss_mlm = mlm_output.loss

        if self.prefix_task:
            
            if self.distill:
                with torch.no_grad():
                    text_output_m = self.text_encoder_m.bert(prefix_input.input_ids, attention_mask=prefix_input.attention_mask,
                                                return_dict=True, mode='text')
                    text_embeds_m = text_output_m.last_hidden_state
                    if self.merge_attention:
                        merge_text_embeds_m = torch.cat([text_embeds_m, image_embeds_m], 1)
                        logits_m = self.text_encoder_m(encoder_embeds=merge_text_embeds_m,
                                           attention_mask=merge_attention_mask,
                                           return_dict=True,
                                           mode='fusion'
                                           )
                    else:
                        text_output_m = self.text_encoder_m.bert(encoder_embeds=text_embeds_m,
                                           attention_mask=prefix_input.attention_mask,
                                           encoder_hidden_states=image_embeds_m,
                                           encoder_attention_mask=image_atts,
                                           return_dict=True,
                                           mode='fusion'
                                           )
                    text_output_m = text_output_m.last_hidden_state
                    if self.concat_last_layer:
                        # text_output_m = torch.cat([text_output_m, image_embeds_m], 1)
                        # merge_text_attention = torch.cat([prefix_input.attention_mask, image_atts], 1)
                        text_output_m = torch.cat([image_embeds_m, text_output_m], 1)
                        merge_text_attention = torch.cat([image_atts, prefix_input.attention_mask], 1)

                    logits_m = self.text_decoder_m(prefix_target.input_ids, 
                                                   attention_mask = prefix_target.attention_mask, 
                                                   encoder_hidden_states = text_output_m,
                                                   encoder_attention_mask = merge_text_attention if self.concat_last_layer else prefix_input.attention_mask,
                                                   return_logits = True,
                                                   )                       
            
            text_output = self.text_encoder.bert(prefix_input.input_ids, attention_mask=prefix_input.attention_mask,
                                                return_dict=True, mode='text')
            text_output = text_output.last_hidden_state
            if self.merge_attention:
                merge_text_embeds = torch.cat([text_embeds, image_embeds], 1)
                merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
                text_output = self.text_encoder.bert(encoder_embeds=merge_text_embeds, 
                                            attention_mask = merge_text_attention, 
                                            return_dict = True, mode='fusion')    
                    
            else:
                text_output = self.text_encoder.bert(encoder_embeds=text_output, 
                                                attention_mask = prefix_input.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True, mode='fusion')    
            text_output = text_output.last_hidden_state
            if self.concat_last_layer:
                # text_output = torch.cat([text_output, image_embeds], 1)
                text_output = torch.cat([image_embeds, text_output], 1)
            answer_targets = prefix_target.input_ids.masked_fill(prefix_target.input_ids == self.tokenizer.pad_token_id, -100)      
            answer_output = self.text_decoder(prefix_target.input_ids, 
                                                  attention_mask = prefix_target.attention_mask, 
                                                  encoder_hidden_states = text_output,
                                                   encoder_attention_mask = merge_text_attention if self.concat_last_layer else prefix_input.attention_mask,
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1) if self.distill else None,
                                                  reduction = 'none',
                                                 )   
            loss_prefix = answer_output.loss
        else:
            loss_prefix = torch.tensor(0.0)
        return loss_mlm, loss_ita, loss_itm, loss_prefix

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

