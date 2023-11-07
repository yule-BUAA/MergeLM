import torch
import torch.nn as nn

from utils.glue_data_loader import rev_glue_data_id_map


class MultiTaskModel(nn.Module):
    def __init__(self, base_model: nn.Module, headers: nn.ModuleDict):
        """
        MultiTask Model.
        :param base_model: nn.Module, the base model (encoder-only model, e.g., BERT and RoBERTa)
        :param headers: nn.ModuleDict, headers for different tasks
        """
        super(MultiTaskModel, self).__init__()
        self.base_model = base_model
        self.headers = headers

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, dataset_ids: torch.Tensor, **kwargs):
        """

        :param input_ids: torch.Tensor, shape (batch_size, seq_len), ids of input tokens
        :param attention_mask: torch.Tensor, shape (batch_size, seq_len), attention mask
        :param dataset_ids: torch.Tensor, shape (batch_size, ), mapped indices of datasets
        :return:
        """
        if "token_type_ids" in kwargs.keys():
            # token_type_ids, torch.Tensor, shape(batch_size, seq_len), token type ids(needed for BERT)
            base_model_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=kwargs["token_type_ids"])
        else:
            base_model_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # dictionary, {"last_hidden_state": Tensor, hidden states of the last layer with shape (batch_size, seq_length, hidden_dim),
        # "pooler_output": Tensor, hidden state of the first [CLS] token after a linear layer and tanh activation function with shape (batch_size, hidden_dim)}
        # Tensor, shape (batch_size, hidden_dim)
        cls_outputs = base_model_output["pooler_output"]
        outputs, output_dims = [], []
        for idx, dataset_id in enumerate(dataset_ids):
            # Tensor, shape (1, output_dim)
            single_output = self.headers[rev_glue_data_id_map[dataset_id.item()]](cls_outputs[idx].unsqueeze(dim=0))
            outputs.append(single_output)
            output_dims.append(single_output.shape[1])
        max_output_dim = max(output_dims)
        for idx, single_output in enumerate(outputs):
            if single_output.shape[1] < max_output_dim:
                outputs[idx] = torch.concat([single_output, torch.full(size=(1, max_output_dim - single_output.shape[1]),
                                                                       fill_value=-1000).to(single_output.device)], dim=1)
        # Tensor, shape (batch_size, max_output_dim)
        logits = torch.concat(outputs, dim=0)
        return {"logits": logits}
