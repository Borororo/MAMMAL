from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator
from models.nmn import NMN as NMN_5
from models.nmn_clevr import NMN as NMN_9
from models.controller  import Controller
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
                 task = '5'   
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.config = config
        self.distill = config['distill']

        #visual encoder: CLIP
        self.visual_encoder, _ = initialize_clip(config)
        # self.visual_encoder = self.visual_encoder.visual
        vision_width = config['vision_width']

        #text encoder
        config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
        
        # text decoder
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.cross_layer = 0
        config_decoder.num_hidden_layers = config['decode_layers']#12 if config['has_decode'] else 6

        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)   

        #check whether large or small vit
        self.large = False
        if config_encoder.hidden_size != vision_width:
            print("Using Large model !!!")
            self.visn_fc = nn.Linear(vision_width, config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(config_encoder.hidden_dropout_prob)
            self.large = True

        #MLP used for answer classification
        # if self.config['use_cat_cls']:
        #     self.answer_predictor = nn.Sequential(
        #         nn.Linear(config_encoder.hidden_size*3,config_encoder.hidden_size),
        #         nn.GELU(),
        #         nn.Dropout(config["hidden_dropout_prob"]),
        #         nn.Linear(config_encoder.hidden_size, config['answer_vocab_size']),
        #     )
        # else:
        #     self.answer_predictor = nn.Sequential(
        #         nn.Linear(config_encoder.hidden_size*2,config_encoder.hidden_size),
        #         nn.GELU(),
        #         nn.Dropout(config["hidden_dropout_prob"]),
        #         nn.Linear(config_encoder.hidden_size, config['answer_vocab_size']),
        #     )

        #Controller & nmn for SNMN
        self.controller = Controller(cfg = config)
        if task =="5":
            self.nmn = NMN_5(cfg = config)
        elif task =='9':
            self.nmn = NMN_9(cfg = config)
        self.init_ctrl = nn.Parameter(
            torch.empty(config['internal_dim']).normal_(mean=0, std=np.sqrt(1 / config['internal_dim']))
        )

        # create image for momentum update
        # if self.distill:           
        #     self.visual_encoder_m, _ = initialize_clip(config)
        #     self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)   
        #     self.answer_predictor_m = nn.Sequential(
        #         nn.Linear(config_encoder.hidden_size*2,config_encoder.hidden_size),
        #         nn.GELU(),
        #         nn.Dropout(config["hidden_dropout_prob"]),
        #         nn.Linear(config_encoder.hidden_size, config['answer_vocab_size']),)
        #     self.controller_m = Controller(cfg = config)
        #     self.nmn_m = NMN(cfg = config)
        #     if config_encoder.hidden_size != vision_width:
        #         self.visn_fc_m = nn.Linear(vision_width, config_encoder.hidden_size)
        #         self.visn_layer_norm_m = nn.LayerNorm(config_encoder.hidden_size, eps=1e-12)
        #         self.dropout_m = nn.Dropout(config_encoder.hidden_dropout_prob)
        #     self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                         [self.text_encoder,self.text_encoder_m],
        #                         [self.controller,self.controller_m],
        #                         [self.nmn,self.nmn_m],
        #                         [self.answer_predictor,self.answer_predictor_m]
        #                        ]
        #     self.copy_params() 
        #     self.momentum = 0.995
        
        # load hyperparameters
        self.steps = config['controller_nodes']
        self.module_names = config['module_names']
        self.num_module = len(self.module_names)
        self.mid_cross = config['mid_cross']
        self.open_generation = config['open_generation']
        self.merge_attention = config['merge_attention']
        self.concat_last_layer = config['concat_last_layer']
        if self.open_generation:
            self.beam_generator = TextGenerator(config, self.text_decoder) 
        self.apply(self._init_weights)
         
    def _init_weights(self,m):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        # if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            # print(f"initialize: {m}")
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
       
    def forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True):
        image = image.to(dtype=next(self.parameters()).dtype) 
        # print(image.shape)
        # image_embeds = self.visual_encoder.visual(image, skip_last_layer=True)
        image_embeds = self.visual_encoder(image, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        #Image size: B, 577, 1024, 

        # remove cls
        nocls_image_embeds = image_embeds[:,1:,:]
        # reshape to [B, H, W, D]
        nocls_image_embeds = nocls_image_embeds.view(nocls_image_embeds.size(0),self.config['h_feature'],self.config['w_feature'],image_embeds.size(-1))
        


  
        if self.mid_cross:
            text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask,
                                            return_dict=True, mode='text')
            text_embeds = text_output.last_hidden_state
            if self.merge_attention:
                merge_text_embeds = torch.cat([text_embeds, image_embeds], 1)
                merge_text_attention = torch.cat([question.attention_mask, image_atts], 1)
                question_output = self.text_encoder(encoder_embeds=merge_text_embeds, 
                                            attention_mask = merge_text_attention, 
                                            return_dict = True, mode='fusion')    
                
            else:
                question_output = self.text_encoder(encoder_embeds=text_embeds, 
                                            attention_mask = question.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,                             
                                            return_dict = True, mode='fusion')    
        else:
            question_output = self.text_encoder(question.input_ids, 
                                            attention_mask = question.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,                             
                                            return_dict = True)    
                
        # 将CLS的向量表示作为NMN中query vector，会经过线性变换
        if self.config['use_image_cls']:
            cls_hidden = image_embeds[:,0,:]
        elif self.config['use_cat_cls']:
            cls_hidden =  torch.cat([image_embeds[:,0,:],question_output.last_hidden_state[:,0,:]],1)
        elif self.config['use_mean_cls']:
            cls_hidden =  torch.stack((image_embeds[:,0,:],question_output.last_hidden_state[:,0,:]),1).mean(dim=1)
        else:
            cls_hidden = question_output.last_hidden_state[:,0,:]


        context_hidden = question_output.last_hidden_state
        # context_hidden = question_output.last_hidden_state[:,1:,:]
        if self.config['mask_pad']:
            context_mask = question.attention_mask
            context_mask[:,0] = 0
        else:
            context_mask = question.attention_mask 
        
        control = self.init_ctrl.unsqueeze(0).repeat(nocls_image_embeds.size(0), 1)
        att_stack, stack_ptr, mem = self.nmn.get_init_values(
            nocls_image_embeds.size(0), nocls_image_embeds.device
        )

        module_logits = []
        question_attns = []
        image_attns = []
        att_stacks = []
        # att_stacks.append(att_stack)
        stack_ptrs = []
        # stack_ptrs.append(stack_ptr)
        mem_stacks = []
        for i in range(self.steps):
            control, module_logit, module_probs, qattn = self.controller(
            context_hidden, cls_hidden, control, context_mask, i
            )
            question_attns.append(qattn)
            module_logits.append(module_logit)
            if self.config["validate_module"]:
                # 这里只根据stack 的ptr的位置进行限制，这样仍然还是会有不足，比如最后一步除了find，其它都可能会出现。第一步noop和find也都会出现。
                # 所以，感觉这里需要对steps的维度加以限制，限制第一步只能是find？最后一步是describe。
                module_validity = stack_ptr.float() @ self.nmn.module_validity_mat.to(stack_ptr.device)
                module_probs = module_probs * module_validity
                module_probs = module_probs / module_probs.sum(1).unsqueeze(1)
            # nmn
            att_stack, stack_ptr, mem = self.nmn(
                control, nocls_image_embeds, module_probs, mem, att_stack, stack_ptr
            )
            # print(att_stack.shape)
            # print(stack_ptr.shape)
            # print(mem.shape)
            # scores = (att_stack * stack_ptr[:, None,None]).sum(-1)
            # print(f'step{i}')
            # N = scores.size(0)
            # scores = F.softmax(scores.view(N, -1), dim=1)
            # print(scores)
            # print(torch.max(scores,-1))
            # print(torch.min(scores,-1))
            image_attns.append((att_stack * stack_ptr[:, None, None]).sum(-1))
            att_stacks.append(att_stack)
            stack_ptrs.append(stack_ptr)
            mem_stacks.append(mem)

        outputs = {
            "qattns": torch.stack(question_attns, 1),
            "iattns": torch.stack(image_attns, 1),
        }
        # outputs["answer_logits"] = output_logits
        outputs["module_logits"] = torch.stack(module_logits, 1)
        outputs['context_mask'] = context_mask
        outputs['att_stack'] = torch.stack(att_stacks,1)
        outputs['stack_ptr'] = torch.stack(stack_ptrs,1)

        patch_size = self.config['h_feature']*self.config['w_feature']
        batch_size = nocls_image_embeds.size(0)
        if self.config['decoder_method'] == "planA":
            # 使用最后一层加权图片分布
            # nocls_image_embeds = nocls_image_embeds.view(nocls_image_embeds.size(0),self.config['h_feature'],self.config['w_feature'],image_embeds.size(-1))
            # print(image_attns[-1].shape)
            #B,H,W
            # print(nocls_image_embeds.shape)
            #B,H,W,D
            # exit()
         
            #转化为B,H*W,1  B,H*W,D
            last_stack_image_hidden = _spatial_softmax(image_attns[-1].view(batch_size,patch_size)).unsqueeze(-1)* nocls_image_embeds.view(batch_size,patch_size,image_embeds.size(-1))
            # print(f"last_stack_image_hidden.shape {last_stack_image_hidden.shape}")
            last_stack_image_attn_mask = image_atts
            # print(f"last_stack_image_attn_mask.shape {last_stack_image_attn_mask.shape}")

            
            decoder_side_input = torch.cat([mem.unsqueeze(1),last_stack_image_hidden],dim=1)
            decoder_side_mask= last_stack_image_attn_mask

        elif self.config['decoder_method'] == "planB":
            # 使用每个步骤的mem拼接
            last_mem_hidden = torch.stack(mem_stacks,dim=1)
            # print(f"last_mem_hidden.shape {last_mem_hidden.shape}")
            last_mem_attn_mask =  torch.ones(last_mem_hidden.size()[:-1],dtype=torch.long).to(mem.device)
            # print(f"last_mem_attn_mask.shape {last_mem_attn_mask.shape}")
            decoder_side_input = last_mem_hidden
            decoder_side_mask= last_mem_attn_mask
        elif self.config['decoder_method'] == "planC":
            # 使用每个步骤的mem拼接
            last_mem_hidden = mem.unsqueeze(1)
            # print(f"last_mem_hidden.shape {last_mem_hidden.shape}")
            last_mem_attn_mask =  torch.ones(last_mem_hidden.size()[:-1],dtype=torch.long).to(mem.device)
            # print(f"last_mem_attn_mask.shape {last_mem_attn_mask.shape}")
            decoder_side_input = last_mem_hidden
            decoder_side_mask= last_mem_attn_mask
        
        else:
            print('invalid decoder method')
            exit()


        question_output = question_output.last_hidden_state
        if self.concat_last_layer:
            # question_output = torch.cat([question_output, image_embeds], 1)
            # merge_text_attention = torch.cat([quesiton.attention_mask, image_atts], 1)
            decoder_side_input = torch.cat([question_output,decoder_side_input], 1)
            decoder_side_mask = torch.cat([question.attention_mask,decoder_side_mask], 1)
        # print(f"decoder_side_input.shape {decoder_side_input.shape}")
        # print(f"decoder_side_mask.shape {decoder_side_mask.shape}")

        if train:
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)    
            #### build mutiple asnwer candidate ####
            decoder_side_inputs = []  
            decoder_side_masks = []   

            for b, n in enumerate(k):
                decoder_side_inputs += [decoder_side_input[b]]*n
                decoder_side_masks += [decoder_side_mask[b]]*n 
            decoder_side_inputs = torch.stack(decoder_side_inputs,0)    
            decoder_side_masks = torch.stack(decoder_side_masks,0)    

            answer_output = self.text_decoder(answer.input_ids, 
                                    attention_mask = answer.attention_mask, 
                                    encoder_hidden_states = decoder_side_inputs,
                                    encoder_attention_mask = decoder_side_masks,                  
                                    labels = answer_targets,
                                    return_dict = True,   
                                    reduction = 'none',
                                    )                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)
            return loss, outputs
        else:

            if self.open_generation:
                topk_ids, topk_probs = self.generation(decoder_side_input,decoder_side_mask, 
                                                            answer.input_ids, answer.attention_mask, k) 
            else:
                topk_ids, topk_probs = self.rank_answer(decoder_side_input,decoder_side_mask,
                                                        answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs,outputs

 



              
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

def _spatial_softmax(att_raw):
    N = att_raw.size(0)
    att_softmax = F.softmax(att_raw.view(N, -1), dim=1)
    att_softmax = att_softmax.view(att_raw.size())
    return att_softmax
