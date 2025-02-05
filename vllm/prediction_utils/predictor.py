import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
from vllm.config import SchedulerConfig
from vllm.inputs import ProcessorInputs
from typing import List, Union

class BertClassificationModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim, num_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        for param in self.bert.parameters():
            param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.logsoftmax(self.fc2(output))
        return output

class BertRegressionModel(nn.Module):
    def __init__(self, config, model_name, hidden_dim):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name)
        # Fix the weights of the pretrained model
        for param in self.bert.parameters():
            param.requires_grad = False

        # The output layer that takes the [CLS] representation and gives an output
        self.cls = nn.Linear(config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, model_name=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Obtain the representations of [CLS] heads
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:,0,:]
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)
        return output

class OutputTokenLengthPredictor:
    multi_cls_thresholds =  [58, 147, 280, 499, 512]
    def __init__(self,
                 scheduler_config: SchedulerConfig,):
        self.device = 'cpu'    
        self.tokenizer = AutoTokenizer.from_pretrained('/home/panenbao/models/bert-base-uncased')
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.model: BertClassificationModel = torch.load(scheduler_config.predictor_path, map_location=self.device)
        self.model.to(device=self.device)
        self.model.eval()

    def _predict(self, prompt: Union[str, List[int]], tokenizer: AutoTokenizer) -> int:
        if isinstance(prompt, list):
            prompt = tokenizer.decode(prompt, skip_special_tokens=True)
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(encoding['input_ids'], encoding['attention_mask'])
            pred = torch.argmax(logits, dim=-1).item()
        return self.multi_cls_thresholds[pred]

    def predict(self, inputs: ProcessorInputs, tokenizer: AutoTokenizer) -> int:
        if 'prompt' in inputs:
            return self._predict(inputs['prompt'], tokenizer)
        elif 'prompt_token_ids' in inputs:
            return self._predict(inputs['prompt_token_ids'], tokenizer)
        else:
            raise ValueError("Invalid input format for prediction.")