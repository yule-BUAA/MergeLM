import argparse
import sys
import logging
import os
import time
from vllm import LLM, SamplingParams

from inference_llms_instruct_math_code import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp


task_model_mapping_dict = {
    "instruct": "WizardLM-13B-V1.2",
    "math": "WizardMath-13B-V1.0",
    "code": "llama-2-13b-code-alpaca"
}
finetuned_model_backbone_mapping_dict = {
    "WizardLM-13B-V1.2": "Llama-2-13b-hf",
    "WizardMath-13B-V1.0": "Llama-2-13b-hf",
    "llama-2-13b-code-alpaca": "Llama-2-13b-hf"
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for direct inference merged LLMs")
    parser.add_argument("--merge_instruct", action="store_true", default=False, help="whether to merge instruct model")
    parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
    parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge code model")
    parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                        choices=["average_merging", "task_arithmetic", "mask_merging"])
    parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
    parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
    parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
    parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
    parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
    parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                        choices=["average_merging", "task_arithmetic"])
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--evaluate_task", type=str, default="instruct", choices=["instruct", "math", "code"], help="task to be evaluated")
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    assert sum([args.merge_instruct, args.merge_math, args.merge_code]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_instruct, args.merge_math, args.merge_code], ["instruct", "math", "code"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        else:
            assert args.mask_apply_method == "task_arithmetic"
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}"

    save_merge_log_path = f"./save_merge_llm_logs/{'_'.join(merge_task_names)}/{args.save_model_name}"
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

    logger.info(f"configuration is {args}")

    save_instruct_model_path = save_math_model_path = save_code_model_path = None
    load_model_path = None
    num_evaluate_tasks = 0
    if args.merge_instruct and args.evaluate_task == "instruct":
        save_instruct_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/instruct/{args.save_model_name}"
        load_model_path = save_instruct_model_path
        num_evaluate_tasks += 1
    if args.merge_math and args.evaluate_task == "math":
        save_math_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math/{args.save_model_name}"
        load_model_path = save_math_model_path
        num_evaluate_tasks += 1
    if args.merge_code and args.evaluate_task == "code":
        save_code_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/code/{args.save_model_name}"
        load_model_path = save_code_model_path
        num_evaluate_tasks += 1

    assert load_model_path is not None and num_evaluate_tasks == 1

    llm = LLM(model=load_model_path, tensor_parallel_size=args.tensor_parallel_size)

    if save_instruct_model_path is not None:
        logger.info(f"evaluating merged model on instruct task...")
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{'_'.join(merge_task_names)}/alpaca_eval/{args.save_model_name}"
        test_alpaca_eval(llm=llm, finetuned_model_name=save_instruct_model_path,
                         args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    if save_math_model_path is not None:
        logger.info(f"evaluating merged model on math task...")
        test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                   start_index=args.start_index, end_index=args.end_index, save_model_path=None)
        test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, save_model_path=None)

    if save_code_model_path is not None:
        logger.info(f"evaluating merged model on code task...")
        save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/human_eval/{args.save_model_name}"
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_model_path=None, save_gen_results_folder=save_gen_results_folder)
        save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/mbpp/{args.save_model_name}"
        test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                  start_index=args.start_index, end_index=args.end_index,
                  save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    logger.info(f"inference of merging method {args.merging_method_name} is completed")

    sys.exit()
