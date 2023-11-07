import copy
import os
import sys
import argparse
from functools import partial
import time
import logging
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from utils.glue_data_loader import GLUEDataLoader, glue_data_metrics_map
from utils.metrics import compute_metrics
from utils.customized_trainers import CustomizedTrainer
from utils.utils import set_random_seed
from model_merging_methods.merging_methods import MergingMethod
from inference_plms_glue import dataset_model_learning_rate_mapping_dict
from utils.load_config import cache_dir


parser = argparse.ArgumentParser("Interface for merging PLMs on glue")
parser.add_argument("--language_model_name", type=str, default="roberta-base", help="name of the language model", choices=["bert-base-uncased", "roberta-base"])
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging", "mask_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--nums_fisher_examples", type=int, nargs="+", help="numbers of examples to compute fisher weights")
parser.add_argument("--fisher_scaling_coefficients", type=float, nargs="+", help="scaling coefficients to merge fisher weights")
parser.add_argument("--normalize_fisher_weight", action="store_true", default=False, help="whether to normalize fisher weights (L2 norm) or not")
parser.add_argument("--minimal_fisher_weight", type=float, default=1e-6, help="the minimal value in fisher weights, used for tackling the potential numerical issues")
parser.add_argument("--nums_regmean_examples", type=int, nargs="+", help="numbers of examples to compute regmean weights")
parser.add_argument("--reduce_non_diagonal_ratio", type=float, default=1.0, help="reduce non-diagonal elements in regmean weights by multiplying this scalar")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="mask rate of the smallest-magnitude parameter values")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging"])
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")

try:
    args = parser.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
except:
    parser.print_help()
    sys.exit()


def get_merge_performance(args: argparse.Namespace, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizer: transformers.AutoTokenizer):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizer: AutoTokenizer, tokenizer
    :return:
    """
    logger.info(f"configuration is {args}")

    try:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name)).to(args.device)
    except:
        merged_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir).to(args.device)

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    # exclude parameter whose name matches "classifier"
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[".*classifier.*"],
                                                   trainers=trainers,
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=args.nums_fisher_examples,
                                                   fisher_scaling_coefficients=args.fisher_scaling_coefficients,
                                                   normalize_fisher_weight=args.normalize_fisher_weight,
                                                   minimal_fisher_weight=args.minimal_fisher_weight,
                                                   nums_regmean_examples=args.nums_regmean_examples,
                                                   reduce_non_diagonal_ratio=args.reduce_non_diagonal_ratio,
                                                   param_value_mask_rate=args.param_value_mask_rate,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=[args.weight_mask_rate for _ in range(len(models_to_merge))],
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   models_use_deepcopy=True)

    merged_model_training_args = TrainingArguments(
        output_dir=args.save_merged_model_path,  # save model directory
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
    )

    for idx, (dataset_name, model_to_merge, trainer) in enumerate(zip(args.dataset_names, models_to_merge, trainers)):
        # only evaluate the target dataset to accelerate the search speed
        if idx < len(args.dataset_names) - 1:
            continue

        # since the classifier is not merged, we additionally set the classifier of merged_model for each model_to_merge
        merged_model.classifier = model_to_merge.classifier
        merged_model_evaluator = CustomizedTrainer(
            model=merged_model,  # final merged model
            args=merged_model_training_args,  # training arguments
            eval_dataset=trainer.eval_dataset,  # evaluation dataset
            compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),  # function for computing metrics
            tokenizer=tokenizer  # tokenizer
        )

        logger.info(f"perform model merging method {args.merging_method_name}:")
        logger.info(f"get performance...")
        test_metrics = merged_model_evaluator.evaluate()
        test_metrics = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in test_metrics.items()}
        logger.info(f"test performance on dataset {dataset_name}: {test_metrics}")

        return test_metrics


if __name__ == "__main__":

    for source_dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]:
        for target_dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]:
            # skip when source_dataset_name equals to target_dataset_name
            if source_dataset_name == target_dataset_name:
                continue

            args.dataset_names = [source_dataset_name, target_dataset_name]

            assert all([dataset_name in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"] for dataset_name in args.dataset_names]), \
                'name in dataset_names must be contained in ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]!'

            load_model_paths = []
            for dataset_name in args.dataset_names:
                # best checkpoint setting
                learning_rate = dataset_model_learning_rate_mapping_dict[f"{dataset_name}_{args.language_model_name}"]
                load_model_paths.append(f"./save_models/{dataset_name}/{args.language_model_name}_lr{learning_rate}")

            # put the target dataset name at end
            if args.merging_method_name == "mask_merging":
                args.save_merged_model_path = f"./save_merge_models/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}/{args.language_model_name}"
            else:
                args.save_merged_model_path = f"./save_merge_models/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.language_model_name}"
            try:
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.language_model_name))
            except:
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.language_model_name, cache_dir=cache_dir)
            glue_data_loader = GLUEDataLoader(tokenizer=tokenizer)

            # load the checkpoint of each individual model that needs to be merged
            models_to_merge, trainers, = [], []
            for dataset_name, load_model_path in zip(args.dataset_names, load_model_paths):

                train_dataset, val_dataset, test_dataset, num_labels = glue_data_loader.load_dataset(dataset_name=dataset_name,
                                                                                                     train_split_ratio_for_val=0.1,
                                                                                                     max_seq_length=128)
                training_args = TrainingArguments(
                    output_dir=load_model_path,                        # load model directory
                    per_device_train_batch_size=args.batch_size,       # batch size per device during training
                    per_device_eval_batch_size=args.batch_size,        # batch size for evaluation
                )

                assert os.path.exists(os.path.join(training_args.output_dir, "trainer_state.json")), "cannot find file trainer_state.json!"
                model_to_merge = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=training_args.output_dir,
                                                                                    num_labels=num_labels).to(args.device)
                trainer = CustomizedTrainer(
                    model=model_to_merge,               # model to be merged
                    args=training_args,                 # training arguments
                    train_dataset=train_dataset,        # training dataset
                    eval_dataset=test_dataset,          # evaluation dataset
                    compute_metrics=partial(compute_metrics, dataset_names=[dataset_name]),   # function for computing metrics
                    tokenizer=tokenizer                 # tokenizer
                )
                models_to_merge.append(model_to_merge)
                trainers.append(trainer)

            merging_method = MergingMethod(merging_method_name=args.merging_method_name)

            # set up logger
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            # put the target dataset name at end
            if args.merging_method_name == "mask_merging":
                save_merge_log_path = f"./save_merge_logs/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}/{args.language_model_name}"
            else:
                save_merge_log_path = f"./save_merge_logs/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.language_model_name}"
            os.makedirs(save_merge_log_path, exist_ok=True)
            # create file handler that logs debug and higher level messages
            fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
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

            best_target_performance = {}
            # search for average_merging
            if args.merging_method_name == "average_merging":
                target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                for metric_name in target_performance.keys():
                    if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                        if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                            logger.info(f"a better model is saved")
                            best_target_performance = copy.deepcopy(target_performance)
            # search for task_arithmetic
            elif args.merging_method_name == "task_arithmetic":
                scaling_coefficient_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                for scaling_coefficient in scaling_coefficient_range:
                    args.scaling_coefficient = scaling_coefficient
                    # dictionary
                    target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                    for metric_name in target_performance.keys():
                        if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                            if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                                logger.info(f"a better model is saved")
                                best_target_performance = copy.deepcopy(target_performance)
                                best_target_performance["scaling_coefficient"] = args.scaling_coefficient
            # search for fisher_merging
            elif args.merging_method_name == "fisher_merging":
                fisher_scaling_coefficient_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                num_fisher_examples_range = [256, 512, 1024, 2048]
                for fisher_scaling_coefficient in fisher_scaling_coefficient_range:
                    for num_fisher_examples in num_fisher_examples_range:
                        args.fisher_scaling_coefficients = [fisher_scaling_coefficient] * len(args.dataset_names)
                        args.nums_fisher_examples = [num_fisher_examples] * len(args.dataset_names)
                        # dictionary
                        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                        for metric_name in target_performance.keys():
                            if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                                if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                                    logger.info(f"a better model is saved")
                                    best_target_performance = copy.deepcopy(target_performance)
                                    best_target_performance["fisher_scaling_coefficients"] = args.fisher_scaling_coefficients
                                    best_target_performance["nums_fisher_examples"] = args.nums_fisher_examples
            # search for regmean_merging
            elif args.merging_method_name == "regmean_merging":
                num_regmean_examples_range = [256, 512, 1024, 2048]
                reduce_non_diagonal_ratio_range = [0.7, 0.8, 0.9, 1.0]
                for num_regmean_examples in num_regmean_examples_range:
                    for reduce_non_diagonal_ratio in reduce_non_diagonal_ratio_range:
                        args.nums_regmean_examples = [num_regmean_examples] * len(args.dataset_names)
                        args.reduce_non_diagonal_ratio = reduce_non_diagonal_ratio
                        # dictionary
                        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                        for metric_name in target_performance.keys():
                            if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                                if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                                    logger.info(f"a better model is saved")
                                    best_target_performance = copy.deepcopy(target_performance)
                                    best_target_performance["nums_regmean_examples"] = args.nums_regmean_examples
                                    best_target_performance["reduce_non_diagonal_ratio"] = args.reduce_non_diagonal_ratio
            # search for ties_merging
            elif args.merging_method_name == "ties_merging":
                scaling_coefficient_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
                param_value_mask_rate_range = [0.7, 0.8, 0.9]
                for scaling_coefficient in scaling_coefficient_range:
                    for param_value_mask_rate in param_value_mask_rate_range:
                        args.scaling_coefficient = scaling_coefficient
                        args.param_value_mask_rate = param_value_mask_rate
                        # dictionary
                        target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                        for metric_name in target_performance.keys():
                            if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                                if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                                    logger.info(f"a better model is saved")
                                    best_target_performance = copy.deepcopy(target_performance)
                                    best_target_performance["scaling_coefficient"] = args.scaling_coefficient
                                    best_target_performance["param_value_mask_rate"] = args.param_value_mask_rate
           # search for mask_merging
            elif args.merging_method_name == "mask_merging":
                with open(f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.mask_apply_method}/{args.language_model_name}.json", "r") as file:
                    # key is evaluate metric or model hyperparameters
                    results_dict = json.load(file)
                if args.mask_apply_method == "task_arithmetic":
                    args.scaling_coefficient = results_dict["scaling_coefficient"]
                elif args.mask_apply_method == "fisher_merging":
                    args.fisher_scaling_coefficients = results_dict["fisher_scaling_coefficients"]
                    args.nums_fisher_examples = results_dict["nums_fisher_examples"]
                elif args.mask_apply_method == "regmean_merging":
                    args.nums_regmean_examples = results_dict["nums_regmean_examples"]
                    args.reduce_non_diagonal_ratio = results_dict["reduce_non_diagonal_ratio"]
                elif args.mask_apply_method == "ties_merging":
                    args.scaling_coefficient = results_dict["scaling_coefficient"]
                    args.param_value_mask_rate = results_dict["param_value_mask_rate"]
                weight_mask_rate_range = [0.1, 0.3, 0.5, 0.7, 0.9]
                for weight_mask_rate in weight_mask_rate_range:
                    args.weight_mask_rate = weight_mask_rate
                    # dictionary
                    target_performance = get_merge_performance(args=args, models_to_merge=models_to_merge, trainers=trainers, logger=logger, merging_method=merging_method, tokenizer=tokenizer)
                    for metric_name in target_performance.keys():
                        if glue_data_metrics_map[args.dataset_names[-1]] in metric_name:
                            if len(best_target_performance.keys()) == 0 or best_target_performance[metric_name] < target_performance[metric_name]:
                                logger.info(f"a better model is saved")
                                best_target_performance = copy.deepcopy(target_performance)
                                best_target_performance["weight_mask_rate"] = args.weight_mask_rate
            else:
                raise NotImplementedError(f"unsupported for merging_method_name {args.merging_method_name}!")
            best_target_performance = {k: float(f"{v:.4f}") if isinstance(v, float) else v for k, v in best_target_performance.items()}
            logger.info(f"best performance and configurations on datasets {args.dataset_names}: {best_target_performance}")
            result_json = json.dumps(best_target_performance, indent=4)
            if args.merging_method_name == "mask_merging":
                save_result_dir = f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}/{args.mask_apply_method}"
            else:
                save_result_dir = f"./save_merge_results/{args.dataset_names[0]}_{args.dataset_names[-1]}/{args.merging_method_name}"
            os.makedirs(save_result_dir, exist_ok=True)
            save_result_path = os.path.join(save_result_dir, f"{args.language_model_name}.json")
            with open(save_result_path, "w") as file:
                file.write(result_json)

            # avoid the overlap of logs
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    sys.exit()
