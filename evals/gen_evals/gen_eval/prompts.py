class BaseIOProcessor:
    def __init__(self, task, task_name, tokenizer, prompt_healing=None):
        self.task = task
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.prompt_healing = prompt_healing
        self.prompt_buffer = {}
        self.heal_func = None

    def process_input(self, doc):
        if self.prompt_healing is None:
            return self.task.get_prompt(doc)
        elif self.prompt_healing == "strip":
            return self.task.get_prompt(doc).rstrip()
        elif self.prompt_healing == "heal":
            orig_prompt = self.task.get_prompt(doc)
            if orig_prompt in self.prompt_buffer:
                return self.prompt_buffer[orig_prompt]
            transformed_prompt = orig_prompt.rstrip() + "\n    "
            new_prompt = self.heal_func(transformed_prompt)
            self.prompt_buffer[orig_prompt] = new_prompt
            return self.prompt_buffer[orig_prompt]
        elif self.prompt_healing == "pad_to_even":
            orig_prompt = self.task.get_prompt(doc)
            if len(orig_prompt) % 2 == 0:
                return orig_prompt
            else:
                return "\n" + orig_prompt
        else:
            raise ValueError(f"Unknown prompt_healing method {self.args.prompt_healing}")

    def process_output(self, output, task_id):
        """
            clean up the generation and extract the appropriate snippet
            for subsequent evaluation
            @output: the full solution (include both prompt and code)
        """
        if "insertion" in self.task_name:
            gen_code = self.task.postprocess_generation(output, int(task_id))
            return gen_code
        dataset = self.task.get_dataset()
        # prompt = self.process_input(dataset[task_id])
        # NOTE: we should use orig prompt to get code
        prompt = self.task.get_prompt(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.postprocess_generation(gen_code, int(task_id))
        return gen_code

    def trim_output(self, output, task_id):
        """
            remove any code beyond the current completion scope
            @output: the full solution (include both prompt and code)
        """
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        gen_code = self.task.trim_generation(gen_code, int(task_id))
        return prompt + gen_code

class ByteLMBaseIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)

    def process_input(self, doc):
        prompt = self.task.get_prompt(doc)
        return prompt

class BaseInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)

    def process_input(self, doc):
        return self.task.get_instruct_prompt(doc, self.tokenizer)

class ByteLMInstructIOProcessor(BaseIOProcessor):
    def __init__(self, task, task_name, tokenizer):
        super().__init__(task, task_name, tokenizer)
        if self.task_name in [
            "humaneval",
            "humaneval_plus",
            "mbpp",
            "mbpp_plus"
        ]:
            self.task.stop_words = [
                "\nclass", 
                "\nprint(", 
                "\nif __name__", 
                "\ndef main(", 
                "\n```", 
                '\n"""', 
                "\nassert", 
                "\n#"
            ]

        if self.task_name.startswith("bigcodebench"):
            self.task.stop_words = [
                "\nif __name__",
                "\ndef main(",
                "\nprint(",
                "\n```\n",
            ]
            # TODO: hacky. will refactor
            self.task._mode = "instruct"
        self.task.stop_words.append("<|eot_id|>")
        # eos tokens are added in evaluator.py

    def process_input(self, doc):
        prompt = self.task.get_instruct_prompt(doc, self.tokenizer)
        if self.task_name in ["mmlu"] or self.task_name.startswith("cute"):
            # we add an extra space for byte models so that the prediction of choices
            # is much more natural
            prompt = prompt + " "
        # print("Input prompt =====>", [prompt], flush=True)
        return prompt

    def process_output(self, output, task_id):
        dataset = self.task.get_dataset()
        prompt = self.process_input(dataset[task_id])
        gen_code = output[len(prompt) :]
        if self.task_name.startswith("bigcodebench"):
            gen_code = self.task.trim_generation(gen_code, int(task_id))
            print("Input prompt =====>", [prompt], flush=True)
            print("Output code =====>", [output[len(prompt) :]], [gen_code], flush=True)
            return gen_code
        gen_code = self.task.postprocess_generation(gen_code, int(task_id))
        if self.task_name.startswith("math") or self.task_name.startswith("gsm"):
            print("Input prompt =====>", [prompt], flush=True)
            print("Output code =====>", [output[len(prompt) :]], [gen_code], flush=True)
        return gen_code
