from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        #self.visual_encoder = VisionTransformer(
        #    img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #    mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    
        self.visual_encoder, _ = initialize_clip(config)
        vision_width = config['vision_width']

        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
        vision_width = config['vision_width']
        
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.cross_layer = 0
        config_decoder.num_hidden_layers = config['decode_layers']#12 if config['has_decode'] else 6

        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    

        self.large = False
        if config_encoder.hidden_size != vision_width:
            self.visn_fc = nn.Linear(vision_width, config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(config_encoder.hidden_dropout_prob)
            self.large = True

        if self.distill:
            #self.visual_encoder_m = VisionTransformer(
            #    img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            #    mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))             
            self.visual_encoder_m, _ = initialize_clip(config)
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)   
            self.text_decoder_m = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   
            if config_encoder.hidden_size != vision_width:
                self.visn_fc_m = nn.Linear(vision_width, config_encoder.hidden_size)
                self.visn_layer_norm_m = nn.LayerNorm(config_encoder.hidden_size, eps=1e-12)
                self.dropout_m = nn.Dropout(config_encoder.hidden_dropout_prob)
            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.text_decoder,self.text_decoder_m],
                               ]
            self.copy_params() 
            self.momentum = 0.995
        self.mid_cross = config['mid_cross']
        self.open_generation = config['open_generation']
        self.merge_attention = config['merge_attention']
        self.concat_last_layer = config['concat_last_layer']
        if self.open_generation:
            self.beam_generator = TextGenerator(config, self.text_decoder) 
            
        

    def forward(self, image, quesiton, answer=None, alpha=0, k=None, weights=None, train=True):
        image = image.to(dtype=next(self.parameters()).dtype) 
        # image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        image_embeds = self.visual_encoder(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      
            if self.mid_cross:
                text_output = self.text_encoder(quesiton.input_ids, attention_mask=quesiton.attention_mask,
                                                return_dict=True, mode='text')
                text_embeds = text_output.last_hidden_state
                if self.merge_attention:
                    merge_text_embeds = torch.cat([text_embeds, image_embeds], 1)
                    merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
                    question_output = self.text_encoder(encoder_embeds=merge_text_embeds, 
                                                attention_mask = merge_text_attention, 
                                                return_dict = True, mode='fusion')    
                    
                else:
                    question_output = self.text_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True, mode='fusion')    
            else:
                question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    
            
            question_output = question_output.last_hidden_state
            if self.concat_last_layer:
                #question_output = torch.cat([question_output, image_embeds], 1)
                #merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
                question_output = torch.cat([image_embeds, question_output], 1)
                merge_text_attention = torch.cat([image_atts, quesiton.attention_mask], 1)
            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output[b]]*n
                if self.merge_attention or self.concat_last_layer:
                    question_atts += [merge_text_attention[b]]*n 
                else:
                    question_atts += [quesiton.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            if self.distill:                    
                with torch.no_grad():
                    self._momentum_update()
                    # image_embeds_m = self.visual_encoder_m.visual(image, skip_last_layer=True) 
                    image_embeds_m = self.visual_encoder_m(image, skip_last_layer=True) 
                    if self.large:
                        image_embeds_m = self.dropout_m(self.visn_layer_norm_m(self.visn_fc_m(image_embeds_m)))
                    if self.mid_cross:
                        text_output_m = self.text_encoder_m(quesiton.input_ids, attention_mask=quesiton.attention_mask,
                                                return_dict=True, mode='text')
                        text_embeds_m = text_output_m.last_hidden_state
                        if self.merge_attention:
                            merge_text_embeds_m = torch.cat([text_embeds_m, image_embeds_m], 1)
                            question_output_m = self.text_encoder_m(encoder_embeds=merge_text_embeds_m, 
                                                            attention_mask = merge_text_attention, 
                                                            return_dict = True, mode='fusion')    
                        else:
                            question_output_m = self.text_encoder_m(encoder_embeds=text_embeds_m, 
                                                            attention_mask = quesiton.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts,                             
                                                            return_dict = True, mode='fusion')    

                         
                    else:
                        question_output_m = self.text_encoder_m(quesiton.input_ids, 
                                                            attention_mask = quesiton.attention_mask, 
                                                            encoder_hidden_states = image_embeds_m,
                                                            encoder_attention_mask = image_atts,                             
                                                            return_dict = True)    
                    question_output_m = question_output_m.last_hidden_state
                    if self.concat_last_layer:
                        # question_output_m = torch.cat([question_output_m, image_embeds_m], 1)
                        question_output_m = torch.cat([image_embeds_m, question_output_m], 1)
                    question_states_m = []                
                    for b, n in enumerate(k):
                        question_states_m += [question_output_m[b]]*n
                    question_states_m = torch.stack(question_states_m,0)    

                    logits_m = self.text_decoder_m(answer.input_ids, 
                                                   attention_mask = answer.attention_mask, 
                                                   encoder_hidden_states = question_states_m,
                                                   encoder_attention_mask = question_atts,                                  
                                                   return_logits = True,
                                                   )                       

                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  soft_labels = F.softmax(logits_m,dim=-1),
                                                  reduction = 'none',
                                                 )   
            else:
                answer_output = self.text_decoder(answer.input_ids, 
                                                  attention_mask = answer.attention_mask, 
                                                  encoder_hidden_states = question_states,
                                                  encoder_attention_mask = question_atts,                  
                                                  labels = answer_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
            

        else: 
            if self.mid_cross:
                text_output = self.text_encoder(quesiton.input_ids, attention_mask=quesiton.attention_mask,
                                                return_dict=True, mode='text')
                text_embeds = text_output.last_hidden_state
                if self.merge_attention:
                    merge_text_embeds = torch.cat([text_embeds, image_embeds], 1)
                    merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
                    question_output = self.text_encoder(encoder_embeds=merge_text_embeds, 
                                                attention_mask = merge_text_attention, 
                                                return_dict = True, mode='fusion')    
                else:
                    question_output = self.text_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True, mode='fusion')    
            else:
                question_output = self.text_encoder(quesiton.input_ids, 
                                                attention_mask = quesiton.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)                    
            question_output = question_output.last_hidden_state
            if self.concat_last_layer:
                # question_output = torch.cat([question_output, image_embeds], 1)
                # merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
                question_output = torch.cat([image_embeds, question_output], 1)
                merge_text_attention = torch.cat([image_atts, quesiton.attention_mask], 1)
            if self.open_generation:
                topk_ids, topk_probs = self.generation(question_output, quesiton.attention_mask if (not self.merge_attention and not self.concat_last_layer) else merge_text_attention, 
                                                            answer.input_ids, answer.attention_mask, k) 
            else:
                topk_ids, topk_probs = self.rank_answer(question_output, quesiton.attention_mask if (not self.merge_attention and not self.concat_last_layer) else merge_text_attention, 
                                                        answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs
 


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
                
    def generation(self, question_states, question_atts, answer_ids, answer_atts, k):
        encoder_inputs = [question_states, question_atts]
        topk_ids = self.beam_generator.translate_batch(encoder_inputs)  
        return topk_ids, topk_ids

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
