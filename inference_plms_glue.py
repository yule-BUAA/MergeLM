import os
import sys
import json
import argparse
import time
import logging
from functools import partial
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from utils.glue_data_loader import GLUEDataLoader
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from model_merging_methods.mask_weights_utils import mask_model_weights
from utils.load_config import cache_dir


dataset_model_learning_rate_mapping_dict = {
    "cola_bert-base-uncased": 5e-5,
    "sst2_bert-base-uncased": 1e-5,
    "mrpc_bert-base-uncased": 5e-5,
    "stsb_bert-base-uncased": 5e-5,
    "qqp_bert-base-uncased": 1e-5,
    "mnli_bert-base-uncased": 1e-5,
    "qnli_bert-base-uncased": 1e-5,
    "rte_bert-base-uncased": 1e-5,
    "cola_roberta-base": 1e-5,
    "sst2_roberta-base": 1e-5,
    "mrpc_roberta-base": 5e-5,
    "stsb_roberta-base": 1e-5,
    "qqp_roberta-base": 1e-5,
    "mnli_roberta-base": 1e-5,
    "qnli_roberta-base": 1e-5,
    "rte_roberta-base": 1e-5
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for inference PLMs on glue")
    parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["bert-base-uncased", "roberta-base"])
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])

    try:
        args = parser.parse_args()
        args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    except:
        parser.print_help()
        sys.exit()

    datasets_test_metrics = []
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
    except:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
    glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

    dataset_names = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        # best checkpoint setting
        learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
        load_model_path = f"./save_models/{dataset_name}/{args.language_model_name}_lr{learning_rate}"
        if args.weight_mask_rate == 0.0:
            save_model_name = f"{args.language_model_name}_lr{learning_rate}_inference_mask_{args.weight_mask_rate}"
        else:
            save_model_name = f"{args.language_model_name}_lr{learning_rate}_inference_mask_{args.weight_mask_rate}_rescale_{args.use_weight_rescale}"
            if args.mask_strategy == "magnitude":
                save_model_name = f"{save_model_name}_strategy_{args.mask_strategy}"
            if args.weight_format == "finetuned_weight":
                save_model_name = f"{save_model_name}_weight_format_{args.weight_format}"
        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./save_logs/{args.dataset_name}/{save_model_name}", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./save_logs/{args.dataset_name}/{save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.INFO)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"********** Run starts. **********")
        logger.info(f"configuration is {args}")

        train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                             train_split_ratio_for_val=0.1,
                                                                                             max_seq_length=128)
        training_args = TrainingArguments(
            output_dir=load_model_path,                        # save model directory
            per_device_train_batch_size=args.batch_size,       # batch size per device during training
            per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
        )
        assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
        finetuned_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=training_args.output_dir,
                                                                             num_labels=num_labels).to(args.device)
        try:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name),
                                                                                  num_labels=num_labels).to(args.device)
        except:
            pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir,
                                                                                  num_labels=num_labels).to(args.device)

        trainer = CustomizedTrainer(
            model=finetuned_model,              # model
            args=training_args,                 # training arguments
            train_dataset=train_dataset,        # training dataset
            eval_dataset=test_dataset,          # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
            tokenizer=tokenizer                 # tokenizer
        )
        if args.weight_mask_rate != 0.0:
            masked_param_dict = mask_model_weights(finetuned_model=finetuned_model, pretrained_model=pretrained_model, exclude_param_names_regex=[".*classifier.*"],
                                                   weight_format=args.weight_format, weight_mask_rate=args.weight_mask_rate, use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy)
            # copy the masked parameters to the original model
            for param_name, param_value in finetuned_model.named_parameters():
                if param_name in masked_param_dict:
                    param_value.data.copy_(masked_param_dict[param_name])

        logger.info(f"get performance of {args.language_model_name}...")
        test_metrics = trainer.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"{args.language_model_name} test performance on dataset {dataset_name}: {test_metrics}")

        result_json = json.dumps(test_metrics, indent=4)
        save_result_dir = f"./save_model_results/{dataset_name}/{save_model_name}"
        os.makedirs(save_result_dir, exist_ok=True)
        save_result_path = os.path.join(save_result_dir, f"{save_model_name}.json")
        with open(save_result_path, "w") as file:
            file.write(result_json)

        datasets_test_metrics.append((save_model_name, dataset_name, test_metrics))

        # avoid the overlap of logs
        if dataset_name != dataset_names[-1]:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    for dataset_test_metrics in datasets_test_metrics:
        logger.info(f"{dataset_test_metrics[0]} test performance on dataset {dataset_test_metrics[1]}: {dataset_test_metrics[2]}")

    sys.exit()
