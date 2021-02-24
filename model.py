import torch
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModelForQuestionAnswering, BertPreTrainedModel,BertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from utils import CRF

 
@dataclass        
class QuestionAnsweringModelOutputWithMultiTask(QuestionAnsweringModelOutput):
    loss: Optional[torch.FloatTensor] = None
    categorized_logits: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
        
@dataclass        
class QuestionAnsweringModelOutputWithMultiTask_CRF(QuestionAnsweringModelOutput):
    loss: Optional[torch.FloatTensor] = None
    categorized_logits: torch.FloatTensor = None
    crf_logits: torch.FloatTensor = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

        
# Example Usage:- smooth_one_hot(torch.tensor([2, 3]), classes=10, smoothing=0.1)
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
  if smoothing == 0, it's one-hot method
  if 0 < smoothing < 1, it's smooth method
  """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    #print(f"Confidence:{confidence}")
    label_shape = torch.Size((true_labels.size(0), classes))
    #print(f"Label Shape:{label_shape}")
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        #print(f"True Distribution:{true_dist}")
        true_dist.fill_(smoothing / (classes - 1))
        #print(f"First modification to True Distribution:{true_dist}")
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    #print(f"Modified Distribution:{true_dist}")
    return true_dist

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
  Args:
        pred: predictions for neural network
        targets: targets, can be soft
        size_average: if false, sum is returned instead of mean
  """
    logsoftmax = nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))
      
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    
    smooth_start_positions = smooth_one_hot(start_positions, classes=384, smoothing=0.1)
    smooth_end_positions = smooth_one_hot(end_positions, classes=384, smoothing=0.1)

    start_loss = cross_entropy(start_logits, smooth_start_positions)
    end_loss = cross_entropy(end_logits, smooth_end_positions)
    total_loss = (start_loss + end_loss)
  
    return total_loss        


class BertForQuestionAnsweringWithMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.span_classifier = nn.Linear(config.hidden_size*2, config.num_labels, bias=True)
        self.include_classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        assert config.num_labels == 2
        self.high_dropout = nn.Dropout(p=0.5) 
        self.dropout = nn.Dropout(p=0.2) 
        torch.nn.init.normal_(self.span_classifier.weight, std=0.02)
        torch.nn.init.normal_(self.include_classifier.weight, std=0.02)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids= None,
        labels=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states = True,
            return_dict= True,
        )
        
        span_hidden_states = bert_output.hidden_states # (batch_size, sequence_length, hidden_size)
        pooled_output = bert_output.pooler_output # (batch_size, sequence_length, hidden_size)
        pooled_output = self.dropout(pooled_output)
        include_logits = self.include_classifier(pooled_output)
        #################################### Span #############################################################################
        span_out = torch.stack((span_hidden_states[-1], span_hidden_states[-2], span_hidden_states[-3], span_hidden_states[-4]), dim=0)  #最后四层拼接
        span_out_mean = torch.mean(span_out, dim=0)
        span_out_max, _ = torch.max(span_out, dim=0)
        span_out = torch.cat((span_out_mean, span_out_max), dim=-1)
        span_logits = torch.mean(torch.stack([ self.span_classifier(self.high_dropout(span_out))for _ in range(5) ], dim=0), dim=0)
        #print(span_logits)
        #######################################################################################################################
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)
    
        total_loss = None
        if start_positions is not None and end_positions is not None and labels is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            #sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            span_loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            include_loss = nn.CrossEntropyLoss()(include_logits, labels)
            total_loss = span_loss + include_loss
        return  QuestionAnsweringModelOutputWithMultiTask(
                        loss= total_loss,
                        start_logits=start_logits,
                        end_logits=end_logits,
                        classifed_logits = include_logits  ,
                        hidden_states= bert_output.hidden_states,
                        attentions= bert_output.attentions  )
  

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)  # start/end
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward( self,
        input_ids=None,
        attention_mask=None,
        token_type_ids= None,
        labels=None,
        head_mask=None,
        start_positions=None,
        end_positions=None,
        return_dict=None,
               ):
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            return_dict= True,
            
            )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # predict start & end position
        sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        # classification
        pooled_output = self.dropout(pooled_output)
        classifier_logits = self.classifier(pooled_output)
        
        if labels is not None:
            #start_labels, end_labels, class_labels = labels
            start_loss = nn.CrossEntropyLoss(ignore_index=-1)(start_logits, start_positions)
            end_loss = nn.CrossEntropyLoss(ignore_index=-1)(end_logits, end_positions)
            class_loss = nn.CrossEntropyLoss()(classifier_logits, labels)
            outputs = start_loss + end_loss + 2*class_loss
            print(outputs)
        else:
            outputs = (start_logits, end_logits, classifier_logits)

        return  outputs
    
class BertForQuestionAnsweringWithCRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.hidden_size = self.bert.config.hidden_size
        self.CRF_fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, config.num_labels + 2, bias=True),
        )
        
        self.CRF = CRF(target_size = self.bert.config.num_labels,device= torch.device("cuda"))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.fc2 = nn.Linear(self.hidden_size, 2, bias=True)

    def forward(self,tokens_id_l, token_type_ids_l, answer_offset_l, answer_seq_label_l, IsQA_l):


        ## 字符ID [batch_size, seq_length]
        tokens_x_2d = torch.LongTensor(tokens_id_l).to(self.device)
        token_type_ids_2d = torch.LongTensor(token_type_ids_l).to(self.device)

        # 计算sql_len 不包含[CLS]
        batch_size, seq_length = tokens_x_2d[:,1:].size()

        ## CRF答案ID [batch_size, seq_length]
        y_2d = torch.LongTensor(answer_seq_label_l).to(self.device)[:,1:]
        ## (batch_size,)
        y_IsQA_2d = torch.LongTensor(IsQA_l).to(self.device)


        if self.training:    # self.training基层的外部类
            self.bert.train()
            output = self.bert(input_ids=tokens_x_2d, token_type_ids=token_type_ids_2d, output_hidden_states= True,return_dict= True)  #[batch_size, seq_len, hidden_size]
        else:
            self.bert.eval()
            with torch.no_grad():
                output = self.bert(input_ids=tokens_x_2d, token_type_ids=token_type_ids_2d, output_hidden_states= True,return_dict= True)

        ## [CLS] for IsQA  [batch_size, hidden_size]
        cls_emb = output.last_hidden_state[:,0,:] 
        
        IsQA_logits = self.fc2(cls_emb) ## [batch_size, 2]
        IsQA_loss = self.CrossEntropyLoss.forward(IsQA_logits,y_IsQA_2d)

        ## [batch_size, 1]
        IsQA_prediction = IsQA_logits.argmax(dim=-1).unsqueeze(dim=-1)

        # CRF mask
        mask = np.ones(shape=[batch_size, seq_length], dtype=np.uint8)
        mask = torch.ByteTensor(mask).to(self.device)  # [batch_size, seq_len, 4]
      

        # No [CLS]
        crf_logits = self.CRF_fc1(output.last_hidden_state[:,1:,:] )
        crf_loss = self.CRF.neg_log_likelihood_loss(feats=crf_logits, mask=mask, tags=y_2d )
        _, CRFprediction = self.CRF.forward(feats=crf_logits, mask=mask)

        return IsQA_prediction, CRFprediction, IsQA_loss, crf_loss, y_2d, y_IsQA_2d.unsqueeze(dim=-1)# (batch_size,) -> (batch_size, 1)

    
class BertForQuestionAnsweringWithMultiTask_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.high_dropout = nn.Dropout(p=0.5) 
        self.dropout = nn.Dropout(p=0.2) 
        self.bert = BertModel(config)
        self.span_classifier = nn.Linear(config.hidden_size*2, config.num_labels, bias=True)
        self.include_classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.CRF_fc1 = nn.Sequential(
            self.high_dropout,
            nn.Linear(config.hidden_size, config.num_labels + 2, bias=True),
            )
        
        self.CRF = CRF(target_size = self.bert.config.num_labels,device= torch.device("cuda"))
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.fc2 = nn.Linear(config.hidden_size, 2, bias=True)
        
        assert config.num_labels == 2
        
        torch.nn.init.normal_(self.span_classifier.weight, std=0.02)
        torch.nn.init.normal_(self.include_classifier.weight, std=0.02)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids= None,
        answer_offset= None,
        answer_seq_label= None,
        labels=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
       
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states = True,
            return_dict= True,
        )
        
        
        last_hidden_state_output = bert_output.last_hidden_state[:,0,:]  # (batch_size, sequence_length, hidden_size)
        last_hidden_state_output = self.dropout(last_hidden_state_output)
        include_logits = self.include_classifier(last_hidden_state_output)
        
        #################################### CRF #############################################################################
        
        batch_size, seq_length = input_ids[:,1:].size() # 计算sql_len 不包含[CLS]
 
        # CRF mask
        mask = np.ones(shape=[batch_size, seq_length], dtype=np.uint8)
        mask = torch.ByteTensor(mask).bool().to('cuda')  # [batch_size, seq_len, 4]
        #print('mask',mask.shape )

        # No [CLS]
        #print(bert_output.last_hidden_state[:,1:,:].shape)
        crf_logits = self.CRF_fc1(bert_output.last_hidden_state[:,1:,:] )
        #_, CRFprediction = self.CRF.forward(feats=crf_logits, mask=mask)   
        
        #################################### Span #############################################################################
        span_hidden_states = bert_output.hidden_states # (batch_size, sequence_length, hidden_size)
        span_out = torch.stack((span_hidden_states[-1], span_hidden_states[-2], span_hidden_states[-3], span_hidden_states[-4]), dim=0)  #最后四层拼接
        span_out_mean = torch.mean(span_out, dim=0)
        span_out_max, _ = torch.max(span_out, dim=0)
        span_out = torch.cat((span_out_mean, span_out_max), dim=-1)
        span_logits = torch.mean(torch.stack([ self.span_classifier(self.high_dropout(span_out))for _ in range(5) ], dim=0), dim=0)
        #######################################################################################################################
        start_logits, end_logits = span_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)
    
        total_loss = None
        if start_positions is not None and end_positions is not None and labels is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            #sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            span_loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
            include_loss = nn.CrossEntropyLoss()(include_logits, labels)
            crf_loss = self.CRF.neg_log_likelihood_loss(feats=crf_logits, mask=mask, tags=answer_seq_label[:,1:] )
            total_loss = span_loss + include_loss + crf_loss
            
        return QuestionAnsweringModelOutputWithMultiTask_CRF(
                        loss= total_loss,
                        start_logits=start_logits,
                        end_logits=end_logits,
                        categorized_logits = include_logits ,
                        crf_logits=crf_logits,
                        hidden_states= bert_output.hidden_states,
                        attentions= bert_output.attentions  )
  