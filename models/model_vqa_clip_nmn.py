from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator
from models.nmn import NMN
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
        
        #check whether large or small vit
        self.large = False
        if config_encoder.hidden_size != vision_width:
            print("Using Large model !!!")
            self.visn_fc = nn.Linear(vision_width, config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(config_encoder.hidden_dropout_prob)
            self.large = True

        #MLP used for answer classification
        if self.config['use_cat_cls']:
            self.answer_predictor = nn.Sequential(
                nn.Linear(config_encoder.hidden_size*3,config_encoder.hidden_size),
                nn.GELU(),
                nn.Dropout(config["hidden_dropout_prob"]),
                nn.Linear(config_encoder.hidden_size, config['answer_vocab_size']),
            )
        else:
            self.answer_predictor = nn.Sequential(
                nn.Linear(config_encoder.hidden_size*2,config_encoder.hidden_size),
                nn.GELU(),
                nn.Dropout(config["hidden_dropout_prob"]),
                nn.Linear(config_encoder.hidden_size, config['answer_vocab_size']),
            )

        #Controller & nmn for SNMN
        self.controller = Controller(cfg = config)
        self.nmn = NMN(cfg = config)
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
        


        answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      
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
            image_attns.append((att_stack * stack_ptr[:, None, None]).sum(-1))
            att_stacks.append(att_stack)
            stack_ptrs.append(stack_ptr)
        outputs = {
            "qattns": torch.stack(question_attns, 1),
            "iattns": torch.stack(image_attns, 1),
        }

        output_logits = self.answer_predictor(torch.cat([cls_hidden, mem], 1))
        outputs["answer_logits"] = output_logits
        outputs["module_logits"] = torch.stack(module_logits, 1)
        outputs['context_mask'] = context_mask
        outputs['att_stack'] = torch.stack(att_stacks,1)
        outputs['stack_ptr'] = torch.stack(stack_ptrs,1)


        return outputs


