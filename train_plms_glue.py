import os
import sys
import time
import json
import copy
import logging
import argparse
from functools import partial
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, TrainingArguments

from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.utils import save_state_and_model_for_hf_trainer, load_state_and_model_for_hf_trainer
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from models.models import MultiTaskModel
from utils.load_config import cache_dir


parser = argparse.ArgumentParser("Interface for training PLMs on glue")
parser.add_argument("--dataset_name", type=str, help="dataset to be used", default="cola", choices=["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"])
parser.add_argument("--auxiliary_dataset_name", type=str, help="auxiliary dataset to be used", default="cola", choices=["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"])
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["bert-base-uncased", "roberta-base"])
parser.add_argument("--multitask_training", action="store_true", default=False, help="whether to use multitask training")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_epochs", type=int, default=10, help="number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
parser.add_argument("--num_runs", type=int, default=5, help="number of runs")

try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()


if __name__ == "__main__":

    test_metrics_all_runs = []
    if args.multitask_training:
        target_dataset_name = args.dataset_name
    for run in range(args.num_runs):

        if run == 0:
            save_model_name = f"{args.language_model_name}_lr{args.learning_rate}"
        else:
            save_model_name = f"{args.language_model_name}_lr{args.learning_rate}_run{run}"
        if args.multitask_training:
            args.dataset_name = target_dataset_name
            assert args.dataset_name != args.auxiliary_dataset_name, "names of target dataset and auxiliary dataset should be different!"
            args.target_dataset_name = args.dataset_name
            # put the target dataset name at end
            args.dataset_name = f"{args.auxiliary_dataset_name}_{args.target_dataset_name}"
        args.save_model_dir = f"./save_models/{args.dataset_name}/{save_model_name}"

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

        run_start_time = time.time()
        logger.info(f"********** Run starts. **********")

        logger.info(f"configuration is {args}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
        except:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)

        glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

        if args.multitask_training:
            dataset_names = [args.auxiliary_dataset_name, args.target_dataset_name]
            multi_train_datasets, multi_val_datasets, multi_test_datasets, datasets_num_labels = \
                glue_data_loader.load_multitask_datasets(dataset_names=dataset_names, train_split_ratio_for_val=0.1, max_seq_length=128)

            # base_model = AutoModel.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)
            base_model = AutoModel.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
            headers = nn.ModuleDict()
            for dataset_name, dataset_num_labels in zip(dataset_names, datasets_num_labels):
                headers[dataset_name] = nn.Linear(in_features=base_model.config.hidden_size, out_features=dataset_num_labels)
            model = MultiTaskModel(base_model=base_model, headers=headers)

            if os.path.exists(os.path.join(args.save_model_dir, "trainer_state.json")):
                model, _ = load_state_and_model_for_hf_trainer(model=model, load_model_dir=args.save_model_dir, map_location="cpu")
                model = model.to(args.device)
            training_args = TrainingArguments(
                output_dir=args.save_model_dir,                     # save model directory
                learning_rate=args.learning_rate,                   # learning rate
                num_train_epochs=args.num_epochs,                   # total # of training epochs
                per_device_train_batch_size=args.batch_size,        # batch size per device during training
                per_device_eval_batch_size=args.batch_size,         # batch size for evaluation
                warmup_ratio=0.06,                                  # warmup first 6% of training steps for learning rate scheduler
                weight_decay=0.01,                                  # strength of weight decay
                evaluation_strategy="epoch",                        # evaluate at per epoch
                load_best_model_at_end=True,                        # load the best model after training
                save_strategy="epoch",                              # strategy for saving models
                metric_for_best_model="averaged_scores",            # specify the metric to compare two different models
                greater_is_better=True,                             # better models should have a greater metric
                label_names=["dataset_ids", "labels"]               # set label names, we use dataset_ids to retrieve ids of datasets,
                                                                    # this makes Trainer's _signature_columns contain "dataset_ids"
            )
            logger.info(f"model -> {model}")

            trainer = CustomizedTrainer(
                model=model,                        # model to be trained
                args=training_args,                 # training arguments
                train_dataset=multi_train_datasets,        # training dataset
                eval_dataset=multi_val_datasets,           # evaluation dataset
                compute_metrics=partial(compute_metrics, dataset_names=dataset_names),   # function for computing metrics
                tokenizer=tokenizer,                # tokenizer
                use_multitask_setting=True   # use multitask setting
            )

            if not os.path.exists(os.path.join(args.save_model_dir, "trainer_state.json")):
                trainer.train()
                # save the state and model for trainer
                save_state_and_model_for_hf_trainer(trainer=trainer)

            logger.info(f"get final performance on datasets {dataset_names}...")
            test_metrics = trainer.evaluate(eval_dataset=multi_test_datasets)
            for key, value in copy.deepcopy(test_metrics).items():
                if isinstance(value, list):
                    for each_dataset_metric in value:
                        dataset_name = each_dataset_metric['dataset_name']
                        metric_name = glue_data_metrics_map[dataset_name]
                        metric_value = each_dataset_metric[metric_name]
                        test_metrics[f"{dataset_name}_{metric_name}"] = metric_value
                        logger.info(f"test performance on dataset {dataset_name}, {metric_name}: {float(f'{metric_value:.4f}')}")
            test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
            logger.info(f"averaged test performance: {test_metrics}")
        else:
            train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=args.dataset_name,
                                                                                                 train_split_ratio_for_val=0.1,
                                                                                                 max_seq_length=128)
            if os.path.exists(os.path.join(args.save_model_dir, "trainer_state.json")):
                model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.save_model_dir,
                                                                           num_labels=num_labels).to(args.device)

            else:
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name),
                                                                               num_labels=num_labels).to(args.device)
                except:
                    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir,
                                                                               num_labels=num_labels).to(args.device)

            training_args = TrainingArguments(
                output_dir=args.save_model_dir,                     # save model directory
                learning_rate=args.learning_rate,                   # learning rate
                num_train_epochs=args.num_epochs,                   # total # of training epochs
                per_device_train_batch_size=args.batch_size,        # batch size per device during training
                per_device_eval_batch_size=args.batch_size,         # batch size for evaluation
                warmup_ratio=0.06,                                  # warmup first 6% of training steps for learning rate scheduler
                weight_decay=0.01,                                  # strength of weight decay
                evaluation_strategy="epoch",                        # evaluate at per epoch
                load_best_model_at_end=True,                        # load the best model after training
                save_strategy="epoch",                              # strategy for saving models
                metric_for_best_model=glue_data_metrics_map[args.dataset_name],     # specify the metric to compare two different models
                greater_is_better=True                              # better models should have a greater metric
            )
            logger.info(f"model -> {model}")

            trainer = CustomizedTrainer(
                model=model,                        # model to be trained
                args=training_args,                 # training arguments
                train_dataset=train_dataset,        # training dataset
                eval_dataset=val_dataset,           # evaluation dataset
                compute_metrics=partial(compute_metrics, dataset_names=[args.dataset_name]),   # function for computing metrics
                tokenizer=tokenizer                 # tokenizer
            )

            if not os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")):
                trainer.train()
                # save the state and model for trainer
                save_state_and_model_for_hf_trainer(trainer=trainer)

            logger.info(f"get final performance on dataset {args.dataset_name}...")
            test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
            logger.info(f"test performance: {test_metrics}")

        result_json = json.dumps(test_metrics, indent=4)
        save_result_dir = f"./save_model_results/{args.dataset_name}/{save_model_name}"
        os.makedirs(save_result_dir, exist_ok=True)
        save_result_path = os.path.join(save_result_dir, f"{save_model_name}.json")
        with open(save_result_path, "w") as file:
            file.write(result_json)

        run_time = time.time() - run_start_time
        logger.info(f"run {run} cost {run_time:.2f} seconds.")

        test_metrics_all_runs.append(test_metrics)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    for run_idx, test_metrics in enumerate(test_metrics_all_runs):
        logger.info(f"test performance on run {run_idx}: {test_metrics}")

    sys.exit()
