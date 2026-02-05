from pprint import pprint

from . import (
    humaneval,
    humaneval_mbpp_plus,
    recode,
)

TASK_REGISTRY = {
    **humaneval.create_all_tasks(),
    **humaneval_mbpp_plus.create_all_tasks(),
    **recode.create_all_tasks(),
}

ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name):
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
