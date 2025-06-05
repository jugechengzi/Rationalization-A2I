import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_embedding=None):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embedding is not None:
            self.embedding.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding.weight.requires_grad = False

    def forward(self, x):
        """
        Inputs:
        x -- (batch_size, seq_length)
        Outputs
        shape -- (batch_size, seq_length, embedding_dim)
        """
        return self.embedding(x)

class GenEncNoShareModel(nn.Module):

    def __init__(self, args):
        super(GenEncNoShareModel, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs) 
        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  

        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        if self.lay:
            cls_outputs = self.layernorm2(cls_outputs)
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = self.layernorm2(outputs)
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits



class Sp_norm_model(nn.Module):         
    def __init__(self, args):
        super(Sp_norm_model, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)
        if args.sp_norm==1:
            self.cls_fc = spectral_norm(nn.Linear(args.hidden_dim, args.num_class))
        elif args.sp_norm==0:
            self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)
        else:
            print('wrong norm')
        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def _independent_soft_sampling(self, rationale_logits):
        """
        Use the hidden states at all time to sample whether each word is a rationale or not.
        No dependency between actions. Return the sampled (soft) rationale mask.
        Outputs:
                z -- (batch_size, sequence_length, 2)
        """
        z = torch.softmax(rationale_logits, dim=-1)

        return z

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = self._independent_soft_sampling(rationale_logits)
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def forward(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  

        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  

        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding) 
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def grad(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  

        gen_logits=self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  

        embedding2=embedding.clone().detach()
        embedding2.requires_grad=True
        cls_embedding =embedding2  * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits,embedding2,cls_embedding

    def g_skew(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        gen_output, _ = self.gen(embedding)  
        gen_output = self.layernorm1(gen_output)
        gen_logits = self.gen_fc(self.dropout(gen_output))  
        soft_log=self._independent_soft_sampling(gen_logits)
        return soft_log


class Two_head_adv(nn.Module):
    def __init__(self, args):
        super(Two_head_adv, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        if args.fr==1:  
            self.cls=self.gen
        else:
            self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)

        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.gen_fc_adv = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        self.gen_pred=[self.gen,self.gen_fc,self.layernorm1,self.cls,self.cls_fc]
        self.gen_para=[self.gen,self.gen_fc,self.layernorm1]
        self.pred_para=[self.cls,self.cls_fc]


    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def get_rationale(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  

        gen_emb,_=self.gen(embedding)
        gen_emb=self.layernorm1(gen_emb)
        gen_emb_adv=torch.detach(gen_emb)

        gen_logits=self.gen_fc(self.dropout(gen_emb))
        gen_logits_adv=self.gen_fc_adv(self.dropout(gen_emb_adv))


        z = self.independent_straight_through_sampling(gen_logits)  
        z_adv=self.independent_straight_through_sampling(gen_logits_adv)

        return z,z_adv

    def pred_with_rationale(self,inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def forward(self, inputs, masks):
        """
        :param inputs:
        :param masks:
        :return:
        """
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits = self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  

        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits

class RNP_0908(nn.Module):
    def __init__(self, args):
        super(RNP_0908, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)
        if args.fr==1:  
            self.cls=self.gen
        else:
            self.cls = nn.GRU(input_size=args.embedding_dim,
                          hidden_size=args.hidden_dim // 2,
                          num_layers=args.num_layers,
                          batch_first=True,
                          bidirectional=True)

        self.cls_fc = nn.Linear(args.hidden_dim, args.num_class)

        self.z_dim = 2
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)
        self.gen_pred=[self.gen,self.gen_fc,self.layernorm1,self.cls,self.cls_fc]
        self.gen_para=[self.gen,self.gen_fc,self.layernorm1]
        self.pred_para=[self.cls,self.cls_fc]


    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def get_rationale(self,inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  
        gen_emb,_=self.gen(embedding)
        gen_emb=self.layernorm1(gen_emb)


        gen_logits=self.gen_fc(self.dropout(gen_emb))

        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z

    def pred_with_rationale(self,inputs, masks,z):
        masks_ = masks.unsqueeze(-1)

        embedding = masks_ * self.embedding_layer(inputs)  
        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return cls_logits

    def train_one_step(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)
        embedding = masks_ * self.embedding_layer(inputs)
        outputs, _ = self.cls(embedding)  
        outputs = outputs * masks_ + (1. -
                                      masks_) * (-1e6)
        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.cls_fc(self.dropout(outputs))
        return logits

    def forward(self, inputs, masks):
        """
        :param inputs:
        :param masks:
        :return:
        """
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  
        gen_logits = self.generator(embedding)

        z = self.independent_straight_through_sampling(gen_logits)  

        cls_embedding = embedding * (z[:, :, 1].unsqueeze(-1))  
        cls_outputs, _ = self.cls(cls_embedding)  
        cls_outputs = cls_outputs * masks_ + (1. -
                                              masks_) * (-1e6)
        cls_outputs = torch.transpose(cls_outputs, 1, 2)
        cls_outputs, _ = torch.max(cls_outputs, axis=2)
        cls_logits = self.cls_fc(self.dropout(cls_outputs))
        return z, cls_logits





class Adv_generator(nn.Module):
    def __init__(self, args):
        super(Adv_generator, self).__init__()
        self.lay=True
        self.args = args
        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        self.gen = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)


        self.z_dim = 2
        self.dropout = nn.Dropout(args.dropout)
        self.gen_fc = nn.Linear(args.hidden_dim, self.z_dim)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.generator=nn.Sequential(self.gen,
                                     SelectItem(0),
                                     self.layernorm1,
                                     self.dropout,
                                     self.gen_fc)

    def independent_straight_through_sampling(self, rationale_logits):
        """
        Straight through sampling.
        Outputs:
            z -- shape (batch_size, sequence_length, 2)
        """
        z = F.gumbel_softmax(rationale_logits, tau=1, hard=True)
        return z

    def get_rationale(self, inputs, masks):
        masks_ = masks.unsqueeze(-1)


        embedding = masks_ * self.embedding_layer(inputs)  

        gen_emb, _ = self.gen(embedding)
        gen_emb = self.layernorm1(gen_emb)

        gen_logits = self.gen_fc(self.dropout(gen_emb))


        z = self.independent_straight_through_sampling(gen_logits)  # (batch_size, seq_length, 2)

        return z


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args

        self.embedding_layer = Embedding(args.vocab_size,
                                         args.embedding_dim,
                                         args.pretrained_embedding)
        
        self.encoder = nn.GRU(input_size=args.embedding_dim,
                                  hidden_size=args.hidden_dim // 2,
                                  num_layers=args.num_layers,
                                  batch_first=True,
                                  bidirectional=True)

        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.hidden_dim, args.num_class)

    def forward(self, inputs, masks, z=None):
        """
        Inputs:
            inputs -- (batch_size, seq_length)
            masks -- (batch_size, seq_length)
        Outputs:
            logits -- (batch_size, num_class)
        """

        masks_ = masks.unsqueeze(-1)

        embeddings = masks_ * self.embedding_layer(inputs)

        if z is not None:
            embeddings = embeddings * (z.unsqueeze(-1))
        outputs, _ = self.encoder(embeddings)

        outputs = outputs * masks_ + (1. - masks_) * (-1e6)

        outputs = torch.transpose(outputs, 1, 2)
        outputs, _ = torch.max(outputs, axis=2)
        logits = self.fc(self.dropout(outputs))
        return logits












