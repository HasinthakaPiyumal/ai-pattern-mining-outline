# Cluster 77

class Workflow:

    def __init__(self, name: str, llm_config: LLMConfig, benchmark: Benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.answer_generate = operator.AnswerGenerate(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.answer_generate(input=problem)
        return solution['answer']

def create_llm_instance(llm_config: LLMConfig) -> BaseLLM:
    llm_cls = MODEL_REGISTRY.get_model(llm_config.llm_type)
    llm = llm_cls(config=llm_config)
    return llm

class Workflow:

    def __init__(self, name: str, llm_config: LLMConfig, benchmark: Benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_MATH_PROBLEM_PROMPT)
        return solution['response']

class Workflow:

    def __init__(self, name: str, llm_config: LLMConfig, benchmark: Benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        return solution['response']

class Workflow:

    def __init__(self, name: str, llm_config: LLMConfig, benchmark: Benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        工作流的实现
        Custom 操作符可以生成任何你想要的内容。
        但当你想获取标准代码时，应该使用 custom_code_generate 操作符。
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        return solution['response']

class Workflow:

    def __init__(self, name: str, llm_config: LLMConfig, benchmark: Benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.benchmark = benchmark
        self.custom = operator.Custom(self.llm)
        self.custom_code_generate = operator.CustomCodeGenerate(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        Custom operator to generate anything you want.
        But when you want to get standard code, you should use custom_code_generate operator.
        """
        solution = await self.custom_code_generate(problem=problem, entry_point=entry_point, instruction=prompt_custom.GENERATE_PYTHON_CODE_PROMPT)
        return solution['response']

