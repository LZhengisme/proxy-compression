"""
ReCode: Robustness Evaluation of Code Generation Models
https://arxiv.org/abs/2212.10264
Recode is a benchmark evaluating the robustness of code generation models to code and natural language perturbations.
This task allows to run the released perturbed HumanEval benchmark, and compute the robust-pass-at-k metric.
"""
from collections import defaultdict
from gen_eval.base import Task
from gen_eval.tasks.custom_metrics.code_eval import compute_code_eval
from datasets import concatenate_datasets, load_dataset
import numpy as np
import os

_CITATION = """
@article{wang2022recode,
  title={ReCode: Robustness Evaluation of Code Generation Models},
  author={Wang, Shiqi and Li, Zheng and Qian, Haifeng and Yang, Chenghao and Wang, Zijian and Shang, Mingyue and Kumar, Varun and Tan, Samson and Ray, Baishakhi and Bhatia, Parminder and others},
  journal={arXiv preprint arXiv:2212.10264},
  year={2022}
}
"""

# typical tasks to run:
#     "perturbed-humaneval-func_name-num_seeds_5"
#     "perturbed-humaneval-format-num_seeds_5"
#     "perturbed-humaneval-syntax-num_seeds_5"
#     "perturbed-humaneval-docstrings-num_seeds_5"


def calculate_passatk(data):
    length = len(data)
    cnt = 0
    for d in data:
        if d["passed"]:
            cnt += 1
    return cnt / length

def get_worst_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_worst = {}
    for pdata in perturbed_data_list[0]:
        passatk_worst[pdata["task_id"]] = True
    for perturbed_data in perturbed_data_list:
        for pdata in perturbed_data:
            assert pdata["task_id"] in passatk_worst
            passatk_worst[pdata["task_id"]] = passatk_worst[pdata["task_id"]] and pdata["passed"]
    return passatk_worst

def get_best_passatk_dict(perturbed_data_list):
    assert len(perturbed_data_list) >= 1
    passatk_best = {}
    for pdata in perturbed_data_list[0]:
        passatk_best[pdata["task_id"]] = False
    for perturbed_data in perturbed_data_list:
        for pdata in perturbed_data:
            assert pdata["task_id"] in passatk_best
            passatk_best[pdata["task_id"]] = passatk_best[pdata["task_id"]] or pdata["passed"]
    return passatk_best


def calculate_metric(perturbed_data_list, metric, nominal_data):
    """ Get targeted metric numbers
    perturbed_data_list: a list of perturbed data completions, each element is the completion of one seed dataset
                         
    """
    length = len(nominal_data)
    # init worst dict
    # passatk_worst = {}
    # for ndata in nominal_data:
    #     passatk_worst[ndata["task_id"]] = True
    passatk_worst = get_worst_passatk_dict(perturbed_data_list)
    passatk_best = get_best_passatk_dict(perturbed_data_list)
    if metric == "passatk":
        # perturbed pass@k
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            passatk_list.append(calculate_passatk(perturbed_data))
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        return passatk_list, worst_cnt / length if passatk_list else " ", passatk_worst

    if metric == "drop":
        # (nominal pass@k - perturbed pass@k) / nominal pass@k
        nominal_passatk = calculate_passatk(nominal_data)
        passatk_list = []
        for perturbed_data in perturbed_data_list:
            perturbed_passatk = calculate_passatk(perturbed_data)
            passatk_list.append((nominal_passatk - perturbed_passatk) / nominal_passatk)
        worst_cnt = 0
        for key in passatk_worst:
            if passatk_worst[key]: 
                worst_cnt += 1
        perturbed_passatk_worst = worst_cnt / length
        return passatk_list, (nominal_passatk - perturbed_passatk_worst) / nominal_passatk if passatk_list else " ", passatk_worst, nominal_passatk

    if metric == "relative":
        # (nominal != perturbed) / total prompts
        diffset = []
        nominal_dict = {}
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
        relative_list = []
        for perturbed_data in perturbed_data_list:
            relative_cnt = 0
            for pdata in perturbed_data:
                if nominal_dict[pdata["task_id"]] != pdata["passed"]:
                    relative_cnt += 1
                    diffset.append(pdata["task_id"])
            relative_list.append(relative_cnt / length)
        diffset = set(diffset)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] != passatk_worst[key]:
                worst_cnt += 1
            elif nominal_dict[key] != passatk_best[key]:
                worst_cnt += 1
        assert len(diffset) == worst_cnt
        return relative_list, worst_cnt / length  if relative_list else " ", passatk_worst

    if metric == "attack_success":
        # (nominal correct & perturbed incorrect) / nominal correct
        nominal_dict = {}
        correct_cnt = 0
        for ndata in nominal_data:
            nominal_dict[ndata["task_id"]] = ndata["passed"]
            if ndata["passed"]:
                correct_cnt += 1
        success_list = []
        for perturbed_data in perturbed_data_list:
            success_cnt = 0
            for pdata in perturbed_data:
                if nominal_dict[pdata["task_id"]] and not pdata["passed"]:
                    success_cnt += 1
            success_list.append(success_cnt / correct_cnt)
        worst_cnt = 0
        for key in passatk_worst:
            if nominal_dict[key] and not passatk_worst[key]:
                worst_cnt += 1
        return success_list, worst_cnt / correct_cnt  if success_list else " ", passatk_worst


TRANSFORMATION_CATEGORIES = ["format", "func_name", "syntax", "docstrings"]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {
        f"perturbed-humaneval-{category}-num_seeds_{num_seeds}": create_task(
            category, num_seeds
        )
        for category in TRANSFORMATION_CATEGORIES
        for num_seeds in range(1, 11)
    }


def create_task(category, num_seeds):
    if category == "docstrings":
        _category = "nlaugmenter"
    elif category == "syntax":
        _category = "natgen"
    else:
        _category = category
    class PerturbedHumanEval(GeneralPerturbedHumanEval):
        DATASET_NAME = _category

        def __init__(self):
            super().__init__(category, num_seeds)

    return PerturbedHumanEval


class GeneralPerturbedHumanEval(Task):
    DATASET_PATH = "RaymondLi/perturbed_humaneval"

    def __init__(self, category, num_seeds):
        super().__init__(
            stop_words=["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```"],
            requires_execution=True,
        )
        # Transformation category
        self.category = category
        self.num_seeds = num_seeds
        self.filtered_dataset = self.dataset["test"].filter(
            lambda x: x["seed"] < num_seeds
        )
        if self.category in ["format", "syntax"]:
            self.original_dataset = load_dataset("json", data_files="gen_eval/tasks/recode_data/humaneval_partial.jsonl")["train"]
        else:
            self.original_dataset = load_dataset("openai_humaneval")["test"]
        self.original_dataset = self.original_dataset.add_column("seed", [None] * len(self.original_dataset))
        self.original_dataset = self.original_dataset.add_column("perturbation_name", [None] * len(self.original_dataset))
        # combine the original dataset with the perturbed dataset
        self.dataset = concatenate_datasets([self.original_dataset, self.filtered_dataset])

    def get_dataset(self):
        """
        Returns dataset for the task or an iterable of any object, that get_prompt can handle
        Only keep the first NUM_SEEDS seeds
        """
        return self.dataset

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc["prompt"]

    def get_reference(self, doc):
        """
        Builds the reference solution for the doc (sample from the test dataset).
        Will be passed to the `process_results` function, and potentially saved.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: dict
        """
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        test_code = "\n" + test_func + "\n" + entry_point
        return {
            "task_id": doc["task_id"],
            "seed": doc["seed"],
            "perturbation_name": doc["perturbation_name"],
            "test_code": test_code,
        }

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]

    def trim_generation(self, generation, idx):
        """Intermediate Removal of any code beyond the current completion scope.
        :param generation: str
            code generation from LM (w/o the prompt)
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        return self._stop_at_stop_token(generation, self.stop_words)

    def postprocess_generation(self, generation, idx):
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        trimmed_gen_code = prompt + self._stop_at_stop_token(generation, self.stop_words)
        tmp_gen_code = ""
        for line in trimmed_gen_code.splitlines():
            lspace = len(line) - len(line.lstrip())
            if lspace == 3:
                tmp_gen_code += " "
            tmp_gen_code += line + "\n"
        return tmp_gen_code

    def process_results(self, generations, references):
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(dict)
            list of dict containing refrences
        :return: dict[str: float]
        """

        _, detailed_results = compute_code_eval(
            references=[ref["test_code"] for ref in references],
            predictions=generations,
            num_workers=int(os.getenv("HF_CODE_EVAL_NUM_PROC", "1")),
        )

        # Calculate metrics using the author's script.
        # perturbed_data_list = [list of [results for each problem] for each seed]
        perturbed_data_dict = defaultdict(lambda: defaultdict(list))
        nominal_data = []
        for i, ref in enumerate(references):
            result = [{"task_id": ref["task_id"], "passed": x[1]["passed"]} for x in detailed_results[i]]
            if ref["seed"] is None or ref["perturbation_name"] is None:
                nominal_data.extend(result)
            else:
                perturbed_data_dict[ref["perturbation_name"]][ref["seed"]].extend(result)
        
        perturbed_data_lists = defaultdict(list)
        for perturbation_name, seed_results in perturbed_data_dict.items():
            for seed, results in seed_results.items():
                perturbed_data_lists[perturbation_name].append(results)
        
                # add all perturbed data to the overall category
                perturbed_data_lists["OVERALL"].append(results)
                
        
        orig_metrics = {}
        for perturbation_name, perturbed_data_list in perturbed_data_lists.items():
            _, worst_passatk, __ = calculate_metric(perturbed_data_list, "passatk", nominal_data)
            _, worst_drop, __, regular_passatk = calculate_metric(perturbed_data_list, "drop", nominal_data)
            _, worst_relative, __ = calculate_metric(perturbed_data_list, "relative", nominal_data)
            _, worst_attack_success, __ = calculate_metric(perturbed_data_list, "attack_success", nominal_data)
            orig_metrics[perturbation_name] = {
                "nominal_passatk": regular_passatk,
                "robust_passatk": worst_passatk,
                "drop": worst_drop,
                "relative": worst_relative,
                "attack_success": worst_attack_success,
            }

        # Compute robust-pass-at-1. For each transformation and each prompt, we have s=5 randomly perturbed prompts.
        # With a single sample per prompt, RP@1 on a given transformation is the fraction of examples where completions
        # for all the perturbed prompts are correct.
        # With n samples per prompt, https://arxiv.org/abs/2212.10264 defines RP@1 as the average of the
        # 1/n * sum_{i=1}^n I(all s correct for generation-seed i) over all prompts.
        # An alternate could be the average of the
        # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i) over all prompts.

        # We compute RP@1 for each transformation
        # transformation -> problem -> seed -> [n results]
        transformation_problem_results = defaultdict(lambda: defaultdict(dict))
        for i, ref in enumerate(references):
            result = detailed_results[i]
            result = [x[1]["passed"] for x in result]
            assert (
                ref["seed"]
                not in transformation_problem_results[ref["perturbation_name"]][
                    ref["task_id"]
                ]
            )
            transformation_problem_results[ref["perturbation_name"]][ref["task_id"]][
                ref["seed"]
            ] = result

        rp1 = {}
        for transformation, problem_results in transformation_problem_results.items():
            res = {}
            res["robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # 1/n * sum_{i=1}^n I(all s correct for generation-seed i)
                float(all(results_)) / len(list(results.values())[0])
                for results in problem_results.values() # [key:seed-value:results for all problems]
                for results_ in zip(*results.values()) # [key:seed-value:results for all problems]
            ) / len(problem_results)

            res["alt-robust-pass-at-1"] = sum(
                # results = {seed -> [n results]}
                # prod_{j=1}^s 1/n * sum_{i=1}^n I(j-th prompt correct for generation-seed i)
                np.prod([np.mean(results[j]) for j in results])
                for results in problem_results.values()
            ) / len(problem_results)
            rp1[transformation] = res

        # TODO: for overall-performance, a prompt is solved if correct over the s prompts for all transformation categories.
        return {"orig_metrics": orig_metrics, "rp1": rp1}
