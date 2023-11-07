import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from utils.glue_data_loader import glue_data_num_labels_map, rev_glue_data_id_map


class CustomizedTrainer(Trainer):

    def __init__(self, use_multitask_setting: bool = False, *args, **kwargs):
        """
        Customized trainer with user-defined train loss function.
        :param use_multitask_setting: boolean, whether to use multitask setting
        """
        super(CustomizedTrainer, self).__init__(*args, **kwargs)
        self.use_multitask_setting = use_multitask_setting

    def compute_loss(self, model: nn.Module, inputs: dict, return_outputs: bool = False):
        """
        how the loss is computed by CustomizedTrainer
        :param model: nn.Module
        :param inputs: dict, model inputs
        :param return_outputs: boolean, whether return the outputs or not
        :return:
        """
        assert "labels" in inputs, "labels are not involved in inputs!"
        labels = inputs.pop("labels")
        if self.use_multitask_setting:
            assert "dataset_ids" in inputs.keys(), "key dataset_ids is missing in the inputs!"
            # Tensor
            dataset_ids = inputs["dataset_ids"]
            outputs = model(**inputs)
            logits = outputs["logits"]
            total_loss = None
            for dataset_id in dataset_ids.unique():
                single_dataset_indices = dataset_ids == dataset_id
                single_dataset_num_labels = glue_data_num_labels_map[rev_glue_data_id_map[dataset_id.item()]]
                # cross-entropy loss for classification
                if single_dataset_num_labels > 1:
                    loss = F.cross_entropy(input=logits[single_dataset_indices][:, :single_dataset_num_labels], target=labels[single_dataset_indices].long())
                # mse loss for regression
                else:
                    assert single_dataset_num_labels == 1, "wrong number of labels!"
                    loss = F.mse_loss(input=logits[single_dataset_indices][:, 0], target=labels[single_dataset_indices])
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss += loss
            return (total_loss, outputs) if return_outputs else total_loss
        else:
            outputs = model(**inputs)
            logits = outputs["logits"]
            if logits.shape[1] > 1:
                # cross-entropy loss for classification
                loss = F.cross_entropy(input=logits, target=labels)
            else:
                # mse loss for regression
                assert logits.shape[1] == 1, "wrong number of labels!"
                loss = F.mse_loss(input=logits.squeeze(dim=1), target=labels)
            return (loss, outputs) if return_outputs else loss
