from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig
from models.nlvr_encoder import BertModel
from models.visual_transformers import initialize_clip
# from models.nmn_v2 import NMN
from models.transnmn import NMN
from models.controller  import Controller
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
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

           
        self.cls_head = nn.Sequential(
            nn.Linear(config_encoder.hidden_size*2,config_encoder.hidden_size),
            nn.ReLU(),
            nn.Linear(config_encoder.hidden_size, 2)
        )  

                
                

        #Controller & nmn for SNMN
        self.controller = Controller(cfg = config)
        self.nmn = NMN(cfg = config)
        self.init_ctrl = nn.Parameter(
            torch.empty(config['internal_dim']).normal_(mean=0, std=np.sqrt(1 / config['internal_dim']))
        )

    
        # load hyperparameters
        self.steps = config['controller_nodes']
        self.module_names = config['module_names']
        self.num_module = len(self.module_names)
        self.mid_cross = config['mid_cross']
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
       
    def forward(self, image0, image1, question, targets, train=True):
        image0 = image0.to(dtype=next(self.parameters()).dtype) 
        image1 = image1.to(dtype=next(self.parameters()).dtype) 
        images = torch.cat([image0,image1],dim=0)
        image_embeds= self.visual_encoder(images, skip_last_layer=True)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))

        image0_embeds, image1_embeds = torch.split(image_embeds,targets.size(0))    
        # image0_embeds = self.visual_encoder(image0, skip_last_layer=True)
        # image1_embeds = self.visual_encoder(image1, skip_last_layer=True)
        # if self.large:
        #     image0_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image0_embeds)))
        #     image1_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image1_embeds)))

        image0_atts = torch.ones(image0_embeds.size()[:-1],dtype=torch.long).to(image0.device)
        image1_atts = torch.ones(image1_embeds.size()[:-1],dtype=torch.long).to(image1.device)

        # remove cls
        #[B,N,D]
        nocls_image0_embeds = image0_embeds[:,1:,:]
        nocls_image0_mask = image0_atts[:,1:]
        nocls_image1_embeds = image1_embeds[:,1:,:]
        nocls_image1_mask = image1_atts[:,1:]
        #[B,2xN,D]
        nocls_image_embeds = torch.cat([nocls_image0_embeds,nocls_image1_embeds],dim=1)
        nocls_image_mask = torch.cat([nocls_image0_mask,nocls_image1_mask],dim=1)
  

        text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask,
                                                return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        question_output = self.text_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = [image0_embeds, image1_embeds],
                                                encoder_attention_mask = [image0_atts, image1_atts],                            
                                                return_dict = True, mode='fusion') 
            
        cls_hidden = question_output.last_hidden_state[:,0,:]
        context_hidden = question_output.last_hidden_state
        context_mask = question.attention_mask 

        control = self.init_ctrl.unsqueeze(0).repeat(nocls_image_embeds.size(0), 1)
        att_stack, stack_ptr, mem = self.nmn.get_init_values(
            nocls_image_embeds.size(0), nocls_image_embeds.device
        )

        module_logits = []
        question_attns = []
        image_attns = []
        att_stacks = []
        stack_ptrs = []
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
                control, module_probs, mem,att_stack, stack_ptr,qattn,context_hidden, context_mask, nocls_image_embeds,nocls_image_mask
            )
            scores = (att_stack * stack_ptr[:, None]).sum(-1)
            image_attns.append((att_stack * stack_ptr[:, None]).sum(-1))
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


        prediction = self.cls_head(torch.cat([cls_hidden,mem],dim=-1))
        # prediction = self.cls_head(cls_hidden)
        if train:
            loss = F.cross_entropy(prediction, targets)
            return loss,outputs
            # return loss
        else:
            return prediction,outputs




