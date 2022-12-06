import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_

def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()
    return lin

class Controller(nn.Module):
    """
    Controller that decides which modules to use and maintains control state.
    Essentially same as MAC control unit.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.module_names = cfg['module_names']
        self.num_modules = len(cfg['module_names'])
        # self.num_modules = cfg['layout_vocab_size']
        control_dim = cfg['internal_dim']
        # if cfg['use_word_emb']:
        #     control_dim = cfg['embedding_dim']
        dim = cfg['internal_dim']

        # self.shared_control_proj = linear(dim, dim)
        self.position_aware = nn.ModuleList()
        if cfg["use_cat_cls"]:
            for i in range(cfg['controller_nodes']):
                self.position_aware.append(linear(dim*2, dim))
        else:
            for i in range(cfg['controller_nodes']):
                self.position_aware.append(linear(dim, dim))

        self.control_question = linear(dim + control_dim, dim)
        self.attn = linear(dim, 1)

        self.module_fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, self.num_modules))

    def forward(
        self, lstm_context, question, control, question_mask, step
    ):
        position_aware = self.position_aware[step](question)

        control_question = torch.cat([control, position_aware], 1)
        control_question = self.control_question(control_question)

        module_logits = self.module_fc(control_question)
        module_probs = F.softmax(module_logits, 1)

        context_prod = control_question.unsqueeze(1) * lstm_context

        attn_weight = self.attn(context_prod).squeeze(-1) - 1e30 * (1 - question_mask)

        attn = F.softmax(attn_weight, 1).unsqueeze(2)

        next_control = (attn * lstm_context).sum(1)

        return next_control, module_logits, module_probs, attn.squeeze(2)
