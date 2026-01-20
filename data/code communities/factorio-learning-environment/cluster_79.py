# Cluster 79

class ParallelPlanningMCTS:

    def __init__(self, instances: List[FactorioInstance], db_client: DBClient, api_factory: Any, config: ParallelMCTSConfig, version=26, version_description='', formatter: ConversationFormatter=StructurePreservingFormatter()):
        """
        Initialize parallel planning MCTS

        Args:
            instances: List of Factorio instances to distribute
            db_client: Database client
            api_factory: Factory for creating language models
            config: Configuration parameters including model paths and prompts
        """
        self.console = Console()
        self.config = config
        self.sampler = config.sampler
        self.db_client = db_client
        self.llm = api_factory
        self.version = version
        self.version_description = version_description
        self.formatter = formatter
        self._validate_instance_count(len(instances), config.n_parallel)
        instances_per_group = floor(len(instances) / config.n_parallel)
        self.logger = GroupedFactorioLogger(n_groups=config.n_parallel, instances_per_group=instances_per_group)
        self.logger.start()
        self.max_steps_per_objective = config.max_steps_per_objective
        self.number_of_steps_for_judge = config.number_of_steps_for_judge
        self.planning_model = config.mcts_kwargs['planning_model']
        self.executor_model = config.mcts_kwargs['executor_model']
        self.objective_model = config.mcts_kwargs['objective_model']
        self.step_executor_prompt_path = config.mcts_kwargs['step_executor_prompt_path']
        self.step_generator_prompt_path = config.mcts_kwargs['step_generator_prompt_path']
        self.step_judge_prompt_path = config.mcts_kwargs['step_judge_prompt_path']
        self.example_plan_prompt_path = config.mcts_kwargs['example_plan_prompt_path']
        self.step_executor_system_prompt, self.step_executor_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_executor_prompt_path'])
        self.step_generator_system_prompt, self.step_generator_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_generator_prompt_path'])
        self.step_judge_system_prompt, self.step_judge_user_prompt = self.read_in_prompts(config.mcts_kwargs['step_judge_prompt_path'])
        self.example_plan_system_prompt, self.example_plan_user_prompt = self.read_in_prompts(config.mcts_kwargs['example_plan_prompt_path'])
        self.instance_groups = self._create_instance_groups(instances)
        self.api_description = self.instance_groups[0].evaluator.instances[0].get_system_prompt()
        self.step_executor_system_prompt = self.step_executor_system_prompt.format(schema=self.api_description)
        self.example_plan_system_prompt = self.example_plan_system_prompt.format(schema=self.api_description)

    def read_in_prompts(self, path):
        system_prompt_path = os.path.join(path, 'system_prompt.md')
        user_prompt_path = os.path.join(path, 'user_prompt.md')
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            user_prompt = f.read()
        return (system_prompt, user_prompt)

    def _create_instance_groups(self, instances: List['FactorioInstance']) -> List[PlanningGroup]:
        """Create instance groups for parallel execution"""
        instances_per_group = floor(len(instances) / self.config.n_parallel)
        groups = []
        for group_id in range(self.config.n_parallel):
            start_idx = group_id * instances_per_group
            end_idx = start_idx + instances_per_group
            group_instances = instances[start_idx:end_idx]
            active_instances = group_instances[:-1]
            holdout_instance = group_instances[-1]
            evaluator = Evaluator(db_client=self.db_client, instances=group_instances, value_accrual_time=3, logger=self.logger, error_penalty=self.config.mcts_kwargs['error_penalty'])
            mcts = self.config.mcts_class(api_factory=self.llm, db_client=self.db_client, sampler=self.sampler, evaluator=evaluator, **self.config.mcts_kwargs)
            groups.append(PlanningGroup(group_id=group_id, mcts=mcts, evaluator=evaluator, active_instances=active_instances, holdout_instance=holdout_instance))
        return groups

    def _validate_instance_count(self, total_instances: int, n_parallel: int):
        min_required = n_parallel * 2
        if total_instances < min_required:
            raise ValueError(f'Need at least {min_required} instances for {n_parallel} parallel searches (got {total_instances})')
        instances_per_group = floor(total_instances / n_parallel)
        if instances_per_group < 2:
            raise ValueError(f'Not enough instances per group (need at least 2, got {instances_per_group})')

    async def search(self, n_iterations: int, skip_failures: bool=False):
        """
        Run truly parallel MCTS search across all groups

        Args:
            n_iterations: Number of iterations to run
            skip_failures: Whether to skip failed program generations
        """
        try:
            search_tasks = []
            for group in self.instance_groups:
                task = asyncio.create_task(self._run_group_search(group=group, n_iterations=n_iterations, skip_failures=skip_failures))
                search_tasks.append(task)
            await asyncio.gather(*search_tasks)
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()

    async def _run_group_search(self, group: PlanningGroup, n_iterations: int, skip_failures: bool=False):
        """Run parallel planning search across all groups"""
        try:
            for iteration in range(n_iterations):
                parent = await self.sampler.sample_parent(version=self.version)
                group.evaluator.set_status('Generating tasks')
                tasks, start_state = await self._get_tasks(group, parent)
                group.evaluator.set_status('Generating plans')
                group.plans = await self.generate_plans(tasks)
                saved_step_ids = []
                for step_idx in range(self.max_steps_per_objective):
                    if step_idx == 0:
                        for instance_id, instance in enumerate(group.evaluator.instances):
                            instance.reset(start_state)
                    plans = await self._process_group_step(group, step_idx, skip_failures, start_state, parent)
                    for plan in plans:
                        try:
                            step_to_save = plan.steps[-1]
                            if step_to_save.program.id not in saved_step_ids:
                                await self.save_step(plan, step_to_save)
                                saved_step_ids.append(step_to_save.program.id)
                        except Exception:
                            print('Could not save step - possibly missing (in case of skipping errors)')
                    group.evaluator.logger.update_progress()
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()

    async def _process_group_step(self, group: PlanningGroup, step_idx: int, skip_failures: bool, start_state: GameState, parent: Program) -> List[PlanOutput]:
        """Process a single step for a group"""
        try:
            group.evaluator.set_status(f'Getting candidates for step {step_idx}')
            group.plans = await self.generate_next_step_candidates(group)
            group.evaluator.set_status(f'Judging candidates for step {step_idx}')
            group.plans = await self.get_next_step_with_judge(group)
            group.evaluator.set_status(f'Generating programs for step {step_idx}')
            group.plans = await self.get_next_step_programs(group)
            eval_futures = []
            completed_plans = []
            for instance_id, (instance, plan) in enumerate(zip(group.active_instances, group.plans.values())):
                if plan.success:
                    if plan.steps[-1].program is None:
                        plan.steps[-1].program = self._create_output_completed_program(plan, parent.id if parent else None)
                    completed_plans.append(plan)
                    continue
                latest_program = plan.steps[-1].program
                group.evaluator.logger.update_instance(instance_id, program_id=latest_program.id, status='evaluating')
                parent_id = parent.id if parent else None
                eval_futures.append(self._process_last_step(plan=plan, start_state=start_state, group=group, instance_id=instance_id, parent_id=parent_id, skip_failures=skip_failures))
            return await asyncio.gather(*eval_futures) + completed_plans
        except Exception as e:
            print(f'Error in group {group.group_id}, step {step_idx}: {str(e)}')
            raise

    def cleanup(self):
        """Clean up resources"""
        self.logger.stop()
        for group in self.instance_groups:
            if hasattr(group.evaluator, 'logger'):
                group.evaluator.logger.stop()

    def get_group_metrics(self, group_id: int) -> Dict[str, Any]:
        """Get metrics for a specific group"""
        if 0 <= group_id < len(self.instance_groups):
            group = self.instance_groups[group_id]
            return {'active_instances': len(group.active_instances), 'total_programs': sum((inst.total_programs for inst in group.evaluator.logger.groups[group_id].instances.values())), 'error_count': sum((inst.error_count for inst in group.evaluator.logger.groups[group_id].instances.values()))}
        return {}

    async def _evaluate_step(self, step: Step, start_state: GameState, group: PlanningGroup, instance_id: int, parent_id) -> Tuple[Step, float, List]:
        """Modified to work with instance groups"""
        group.holdout_instance.reset(start_state)
        entity_list = []
        try:
            instance = group.active_instances[instance_id]
            step.start_state = GameState.from_instance(instance)
            group.evaluator.logger.update_instance(instance_id, status='executing')
            reward, state, response, entities, achievements, ticks = await group.evaluator._evaluate_single(instance_id, step.program, instance)
            entity_list.append(entities)
            step.end_state = state
            step.reward = reward
        except Exception as e:
            print(f'Error during evaluation in group {group.group_id}, instance {instance_id}: {e}')
            raise e
        step.program.value = step.reward
        step.program.raw_reward = step.reward
        step.program.holdout_value = step.reward
        step.program.state = step.end_state
        step.program.response = response
        step.program.parent_id = parent_id
        step.program.achievements = achievements
        return (step, step.reward, entity_list)

    async def save_step(self, plan: PlanOutput, step: Step):
        candidate_step_meta = []
        if step.judge_step_str == '':
            for candidate_step in step.candidate_language_outputs:
                try:
                    messages = candidate_step.conversation.model_dump()['messages']
                except:
                    messages = candidate_step.conversation.dict()['messages']
                output = candidate_step.response
                candidate_step_meta.append({'output': output, 'messages': messages})
            step.program.meta['candidate_steps'] = candidate_step_meta
            await self.db_client.create_program(step.program)
            return
        objective = plan.task.task
        initial_plan = plan.initial_plan.initial_plan
        parent_id = None
        for current_step, next_step in zip(plan.steps[:-1], plan.steps[1:]):
            if next_step.final_step == step.final_step:
                parent_id = current_step.program.id
        for candidate_step in step.candidate_language_outputs:
            try:
                messages = candidate_step.conversation.model_dump()['messages']
            except:
                messages = candidate_step.conversation.dict()['messages']
            output = candidate_step.response
            candidate_step_meta.append({'output': output, 'messages': messages})
            mining_setup = candidate_step.meta['mining_setup']
            starting_inventory = candidate_step.meta['starting_inventory']
        try:
            judge_messages = step.judge_language_output_step.conversation.model_dump()['messages']
        except:
            judge_messages = step.judge_language_output_step.conversation.dict()['messages']
        judge_output = step.judge_step_str
        executor_step = step.final_step
        meta = {'objective': objective, 'initial_plan': initial_plan, 'candidate_steps': candidate_step_meta, 'judge_step': {'messages': judge_messages, 'output': judge_output}, 'executor_step': {'input_step': executor_step, 'natural_language_plan': step.program.meta['text_response'], 'model': step.program.meta['model']}, 'mining_setup': mining_setup, 'starting_inventory': starting_inventory, 'final_output': plan.final_output}
        program = step.program
        program.meta = meta
        program.parent_id = parent_id
        await self.db_client.create_program(program)
        parent_id = program.id

    async def save_plan(self, plan: PlanOutput):
        for step in plan.steps:
            await self.save_step(plan, step)

    async def _process_last_step(self, plan: PlanOutput, start_state: GameState, group: PlanningGroup, instance_id: int, parent_id: Optional[int], skip_failures: bool) -> PlanOutput:
        try:
            step_to_process = plan.steps[-1]
            step_to_process, _, entity_list = await self._evaluate_step(step_to_process, start_state, group, instance_id, parent_id)
            if skip_failures and 'error' in step_to_process.program.response.lower():
                raise Exception('Found error in response. Skipping step.')
            plan.steps[-1] = step_to_process
            log_str = f'Step {len(plan.steps)}: {step_to_process.final_step}\n{step_to_process.program.response}'
            plan.logs.append(log_str)
            return plan
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}: {str(e)}')
            plan.steps.pop()
            return plan

    def _create_output_completed_program(self, plan: PlanOutput, parent_id: Optional[int]) -> PlanOutput:
        objective = f"'{plan.task.task}'"
        python_code = f"print('Objective {objective} has been completed. Now lets prepare the next objective.')"
        program_parent_id = plan.steps[-2].program.id if len(plan.steps) > 1 else parent_id
        program = Program(id=hash((python_code, plan.task.task, program_parent_id)), code=python_code, conversation=Conversation(messages=[]), parent_id=program_parent_id, version=self.version, version_description=self.version_description, meta={'objective': plan.task.task})
        return program

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_natural_language_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta: dict) -> List[LanguageOutput]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model)
            outputs = []
            try:
                messages = conversation.model_dump()['messages']
            except:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                str_output = response.content[0].text
                outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.output_tokens + response.usage.input_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            else:
                for choice in response.choices:
                    str_output = choice.message.content
                    outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            return outputs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def get_inventory_dict(self, inventory):
        inventory_dict = {}
        for item in inventory:
            if isinstance(item, tuple):
                inventory_dict[item[0]] = inventory[item]
            else:
                inventory_dict[item] = inventory[item]
        return inventory_dict

    async def _get_tasks(self, group: PlanningGroup, parent: Program=None) -> Tuple[List[TaskOutput], GameState]:
        """Modified to work with instance groups"""
        start_state = parent.state if parent else self.config.initial_state
        first_instance = group.active_instances[0]
        first_instance.reset(start_state)
        mining_setup = get_mining_setup(first_instance)
        starting_inventory = first_instance.inspect_inventory()
        conversation = Conversation(messages=[Message(role='system', content=self.config.system_prompt), Message(role='user', content=f"Your starting inventory is {starting_inventory}. {mining_setup}. Create an incrementally useful task that you can carry out in the current game, in order to grow your factory's _automatic_ throughput.")])
        generation_params = GenerationParameters(model=self.config.mcts_kwargs['objective_model'], n=len(group.active_instances), stop_sequences=['\n'])
        inventory_dict = self.get_inventory_dict(starting_inventory)
        game_state_str = GameState.from_instance(first_instance).entities
        tasks = await self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'objective_generation', 'inventory': inventory_dict, 'mining_setup': mining_setup, 'game_state': game_state_str, 'group_id': group.group_id})
        task_outputs = []
        for task in tasks:
            task_string = task.response.split('\n')[0].strip()
            task_string = task_string.lower().replace('sure! the task i will carry out is', '').strip()
            if '.' in task_string:
                task_string = task_string.split('.')[0]
            task_outputs.append(TaskOutput(task=task_string, language_output=task))
        return (task_outputs, start_state)

    async def generate_plans(self, task_outputs: List[TaskOutput]) -> List[InitialPlanOutput]:
        generation_params = GenerationParameters(model=self.executor_model, stop_sequences=['```'], logits={'7032': -100})
        conversations_to_process = [Conversation(messages=[Message(role='system', content=self.example_plan_system_prompt), Message(role='user', content=self.example_plan_user_prompt.format(task=task_output.task))]) for task_output in task_outputs]
        initial_plans = [asyncio.ensure_future(self._generate_natural_language_batch(conversation, generation_params, meta={'type': 'initial_plan_generation'})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*initial_plans)
        plan_outputs = {}
        for idx, response in enumerate(responses):
            initial_plan = response[0].response.strip()
            new_line_idx = initial_plan.rfind('\n')
            if new_line_idx != -1:
                initial_plan = initial_plan[:new_line_idx].strip()
            initial_plan_output = InitialPlanOutput(initial_plan=initial_plan, language_output=response[0])
            plan_output = PlanOutput(task=task_outputs[idx], initial_plan=initial_plan_output, meta={'plan_id': idx})
            plan_outputs[idx] = plan_output
        return plan_outputs

    def format_log_string(self, plan_output: PlanOutput):
        return '\n\n'.join(plan_output.logs) if plan_output.logs else 'The agent has not yet interacted with the world'

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_next_step_candidates(self, group) -> List[PlanOutput]:
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            starting_inventory_dict = self.get_inventory_dict(starting_inventory)
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            user_message = self.step_generator_user_prompt.format(mining_setup=mining_setup, starting_inventory=starting_inventory, logs=log_str, objective=objective, plan=initial_plan)
            conversations_to_process += [(Conversation(messages=[Message(role='system', content=self.step_generator_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id'])] * self.number_of_steps_for_judge
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_candidates', 'plan_id': conversation[1], 'mining_setup': mining_setup, 'starting_inventory': starting_inventory_dict})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        step_output_objects = {}
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            if plan_id not in step_output_objects:
                step_output_objects[plan_id] = Step(candidate_language_outputs=[])
            step_output_objects[plan_id].candidate_language_outputs.append(output)
            if '#output' in step_output.lower() and '#step' not in step_output.lower():
                step_output = step_output.lower().split('#output')[-2].strip()
                plan_outputs[plan_id].success = True
                plan_outputs[plan_id].final_output = step_output
                step_output_objects[plan_id].final_step = step_output
        for plan_id, step_output in step_output_objects.items():
            plan_outputs[plan_id].steps.append(step_output)
        return plan_outputs

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_step_with_judge(self, group) -> List[PlanOutput]:
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.planning_model, max_tokens=4096)
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.evaluator.instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            log_str = self.format_log_string(plan_output)
            objective = plan_output.task.task
            initial_plan = plan_output.initial_plan.initial_plan
            step_to_process = plan_output.steps[-1].candidate_language_outputs
            step_candidate_str = ''
            for step_idx, step_candidate in enumerate(step_to_process):
                step_candidate_str += f'Step {step_idx}\n{step_candidate.response}\n\n'
            user_message = self.step_judge_user_prompt.format(objective=objective, starting_inventory=starting_inventory, mining_setup=mining_setup, logs=log_str, plan=initial_plan, analysis_step_str=step_candidate_str)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_judge_system_prompt), Message(role='user', content=user_message)]), plan_output.meta['plan_id']))
        step_outputs = [asyncio.ensure_future(self._generate_natural_language_batch(conversation[0], generation_params, meta={'type': 'next_step_judge', 'plan_id': conversation[1]})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            step_output = output.response.strip()
            plan_outputs[plan_id].steps[-1].judge_language_output_step = output
            plan_outputs[plan_id].steps[-1].judge_step_str = step_output
            if '#STEP' in step_output:
                step = step_output.split('#STEP')[-2].strip()
            elif 'OUTPUT' in step_output:
                step = step_output.split('OUTPUT')[-1].strip()
            else:
                step = None
            if step:
                plan_outputs[plan_id].steps[-1].final_step = step
        return plan_outputs

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get_next_step_programs(self, group: PlanningGroup) -> List[PlanOutput]:
        """Generate programs for the next step in a specific group"""
        plan_outputs = group.plans
        generation_params = GenerationParameters(model=self.config.mcts_kwargs['executor_model'], temperature=0.5, max_tokens=4096, logits={'7032': -100})
        conversations_to_process = []
        for instance_id, plan_output in plan_outputs.items():
            if plan_output.success:
                continue
            instance = group.active_instances[instance_id]
            mining_setup = get_mining_setup(instance)
            starting_inventory = instance.inspect_inventory()
            executor_objective = plan_output.steps[-1].final_step
            user_message = self.step_executor_user_prompt.format(task=executor_objective, starting_inventory=starting_inventory, mining_setup=mining_setup)
            conversations_to_process.append((Conversation(messages=[Message(role='system', content=self.step_executor_system_prompt), Message(role='user', content=user_message)]), {'plan_id': plan_output.meta['plan_id'], 'group_id': group.group_id}))
        step_outputs = [asyncio.ensure_future(self._generate_programs_batch(conversation[0], generation_params, meta={'type': 'next_step_program', 'plan_id': conversation[1]['plan_id'], 'group_id': conversation[1]['group_id']})) for conversation in conversations_to_process]
        responses = await asyncio.gather(*step_outputs)
        for idx, response in enumerate(responses):
            output = response[0]
            plan_id = output.meta['plan_id']
            plan_outputs[plan_id].steps[-1].program = output
        return plan_outputs

    def _verify_response_is_python(self, content):
        code = content
        try:
            compile(code, filename='<ast>', mode='exec')
        except SyntaxError:
            code = code.rsplit('\n', 1)[0] + '\n'
            compile(code, filename='<ast>', mode='exec')
        return code

    def _extract_code_from_choice(self, choice) -> Optional[str]:
        """Extract code from a single completion choice"""
        code = ''
        try:
            content = choice.message.content
            code = self._verify_response_is_python(content)
            code = code.replace('from factorio_instance import *', '')
            return (code, None)
        except Exception:
            try:
                content = choice.message.content
                content_split = content.split('```python')
                code = content_split[1].split('```')[0].strip()
                text_response = content_split[0].strip()
                code = self._verify_response_is_python(code)
                code = code.replace('from factorio_instance import *', '')
                return (code, text_response)
            except Exception as e1:
                content = '\n'.join(choice.message.content.split('\n')[1:])
                try:
                    code = self._verify_response_is_python(content)
                    code = code.replace('from factorio_instance import *', '')
                    return (code, None)
                except Exception:
                    try:
                        content_split = content.split('from factorio_instance import *')
                        code = content_split[1].strip()
                        text_response = content_split[0].strip()
                        code = self._verify_response_is_python(code)
                        return (code, text_response)
                    except Exception:
                        chain_of_thoughts = '"""\n' + content.strip().strip('"') + '\n"""'
                        return (chain_of_thoughts, content.strip())
                print(f'Failed to extract code from choice: {str(e1)}')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=generation_params.presence_penalty)
            programs = []
            try:
                messages = conversation.model_dump()['messages']
            except Exception:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                code, text_response = self._extract_code_from_choice(response)
                if code:
                    programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=response.message.content, token_usage=response.usage.total_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                    if meta:
                        programs[0].meta.update(meta)
            else:
                for choice in response.choices:
                    code, text_response = self._extract_code_from_choice(choice)
                    if code:
                        programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=choice.message.content, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                        if meta:
                            programs[-1].meta.update(meta)
            return programs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

class SupervisedTaskExecutorABC(ABC):

    def __init__(self, instances: List[FactorioInstance], db_client: DBClient, api_factory: Any, config: SupervisedExecutorConfig, version=None, version_description=''):
        """
        Initialize parallel planning MCTS

        Args:
            instances: List of Factorio instances to distribute
            db_client: Database client
            api_factory: Factory for creating language models
            config: Configuration parameters including model paths and prompts
        """
        self.console = Console()
        self.config = config
        self.db_client = db_client
        self.llm = api_factory
        self.version = version
        self.version_description = version_description
        self.model_to_evaluate = config.model_to_evaluate
        self.formatter = DefaultFormatter()
        self._validate_instance_count(len(instances), config.n_parallel)
        instances_per_group = floor(len(instances) / config.n_parallel)
        self.logger = GroupedFactorioLogger(n_groups=config.n_parallel, instances_per_group=instances_per_group)
        self.logger.start()
        self.instance_groups = self._create_instance_groups(instances)

    def read_in_prompts(self, path):
        system_prompt_path = os.path.join(path, 'system_prompt.md')
        user_prompt_path = os.path.join(path, 'user_prompt.md')
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        with open(user_prompt_path, 'r') as f:
            user_prompt = f.read()
        return (system_prompt, user_prompt)

    def _create_instance_groups(self, instances: List['FactorioInstance']) -> List[PlanningGroupV2]:
        """Create instance groups for parallel execution"""
        instances_per_group = floor(len(instances) / self.config.n_parallel)
        groups = []
        for group_id in range(self.config.n_parallel):
            start_idx = group_id * instances_per_group
            end_idx = start_idx + instances_per_group
            group_instances = instances[start_idx:end_idx]
            active_instances = group_instances
            evaluator = Evaluator(db_client=self.db_client, instances=group_instances, value_accrual_time=3, logger=self.logger)
            groups.append(PlanningGroupV2(group_id=group_id, evaluator=evaluator, active_instances=active_instances))
        return groups

    def _validate_instance_count(self, total_instances: int, n_parallel: int):
        min_required = n_parallel * 2
        if total_instances < min_required:
            raise ValueError(f'Need at least {min_required} instances for {n_parallel} parallel searches (got {total_instances})')
        instances_per_group = floor(total_instances / n_parallel)
        if instances_per_group < 2:
            raise ValueError(f'Not enough instances per group (need at least 2, got {instances_per_group})')

    async def search_supervised(self, n_iterations: int, task: TaskABC, skip_failures: bool=False, run_id: str=''):
        """
        Run truly parallel MCTS search across all groups

        Args:
            n_iterations: Number of iterations to run
            skip_failures: Whether to skip failed program generations
        """
        try:
            search_tasks = []
            for group in self.instance_groups:
                search_task = asyncio.create_task(self._run_group_search(group=group, n_iterations=n_iterations, skip_failures=skip_failures, task=task, run_id=run_id))
                search_tasks.append(search_task)
            results = await asyncio.gather(*search_tasks)
        except Exception as e:
            print(f'Error during parallel search: {str(e)}')
            raise
        finally:
            self.cleanup()
            results = [item for sublist in results for item in sublist]
            return results

    async def generate_plans(self, task: TaskABC, nr_of_beams: int) -> List[InitialPlanOutput]:
        plan_outputs = {}
        for idx in range(nr_of_beams):
            plan_output = PlanOutput(task=TaskOutput(task=task.task), meta={'plan_id': idx})
            plan_outputs[idx] = plan_output
        return plan_outputs

    @abstractmethod
    async def _run_group_search(self, group: PlanningGroupV2, task: TaskABC, n_iterations: int, skip_failures: bool=False):
        """Run parallel planning search across all groups"""
        '\n        Need to check again over what to do mcts exactly\n        '
        pass

    def cleanup(self):
        """Clean up resources"""
        self.logger.stop()
        for group in self.instance_groups:
            if hasattr(group.evaluator, 'logger'):
                group.evaluator.logger.stop()

    def get_group_metrics(self, group_id: int) -> Dict[str, Any]:
        """Get metrics for a specific group"""
        if 0 <= group_id < len(self.instance_groups):
            group = self.instance_groups[group_id]
            return {'active_instances': len(group.active_instances), 'total_programs': sum((inst.total_programs for inst in group.evaluator.logger.groups[group_id].instances.values())), 'error_count': sum((inst.error_count for inst in group.evaluator.logger.groups[group_id].instances.values()))}
        return {}

    async def _evaluate_step(self, step: Step, start_state: GameState, group: PlanningGroupV2, instance_id: int, parent_id) -> Tuple[Step, float, List]:
        """Modified to work with instance groups"""
        entity_list = []
        try:
            instance = group.active_instances[instance_id]
            group.evaluator.logger.update_instance(instance_id, status='executing')
            for program in step.sampled_programs:
                instance.reset(step.start_state)
                if not isinstance(program, Program):
                    print(f'Weird program 1: {program}')
                instance.reset(step.start_state)
                reward, state, response, entities, achievements, profits, error = await group.evaluator._evaluate_single(instance_id, program, instance)
                if not isinstance(program, Program):
                    print(f'Weird program 2: {program}')
                if error:
                    print(f'Error in group {group.group_id}, instance {instance_id}: {error}')
                step.program = program
                if not error:
                    break
            entity_list.append(entities)
            step.end_state = state
            step.reward = reward
            post_production_flows = instance.get_production_stats()
            step.program.meta['post_production_flows'] = post_production_flows
            step.program.meta['profits'] = profits
        except Exception as e:
            print(f'Error during evaluation in group {group.group_id}, instance {instance_id}: {e}')
            raise e
        step.program.value = step.reward
        step.program.raw_reward = step.reward
        step.program.state = step.end_state
        step.program.response = response
        step.program.parent_id = parent_id
        step.program.achievements = achievements
        return (step, entity_list)

    async def _process_last_step(self, plan: PlanOutput, start_state: GameState, group: PlanningGroupV2, instance_id: int, parent_id: Optional[int], skip_failures: bool) -> PlanOutput:
        try:
            step_to_process = plan.steps[-1]
            step_to_process, entity_list = await self._evaluate_step(step_to_process, start_state, group, instance_id, parent_id)
            if skip_failures and 'error' in step_to_process.program.response.lower():
                raise Exception('Found error in response. Skipping step.')
            plan.steps[-1] = step_to_process
            log_str = f'Step {len(plan.steps)}: {step_to_process.final_step}\n{step_to_process.program.response}'
            plan.logs.append(log_str)
            return plan
        except Exception as e:
            print(f'Failed to evaluate program on instance {instance_id}: {str(e)}')
            plan.steps.pop()
            return plan

    def check_for_task_completion(self, task: TaskABC, plan: PlanOutput, group: PlanningGroupV2) -> bool:
        sleep_seconds = 60
        instance_id = plan.meta['plan_id']
        instance = group.evaluator.instances[instance_id]
        start_state = plan.steps[-1].start_state
        instance.reset(start_state)
        instance_inventory = instance.inspect_inventory()
        result, achievements, post_production_flows = group.evaluator._evaluate_for_achievements(code=f'sleep({sleep_seconds})', instance=instance)
        for check_dict in task.check_dicts:
            if check_dict['task_type'] == 'craft':
                item = check_dict['item']
                quantity = check_dict['quantity']
                if instance_inventory[item] < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'dynamic':
                item = check_dict['item']
                quantity = check_dict['quantity']
                item_dynamic_value = achievements['dynamic'].get(item, 0)
                item_dynamic_value_per_second = item_dynamic_value
                if item_dynamic_value_per_second < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'production_flows_output':
                item = check_dict['item']
                quantity = check_dict['quantity']
                production_flows_output_item_value = post_production_flows['output'].get(item, 0)
                if production_flows_output_item_value < quantity:
                    return (False, post_production_flows)
            elif check_dict['task_type'] == 'production_flows_input':
                item = check_dict['item']
                quantity = check_dict['quantity']
                production_flows_output_item_value = post_production_flows['input'].get(item, 0)
                if production_flows_output_item_value < quantity:
                    return (False, post_production_flows)
        return (True, post_production_flows)

    def _create_output_completed_program(self, plan: PlanOutput, parent_id: Optional[int], task: TaskABC, group: PlanningGroupV2) -> PlanOutput:
        if task.check_for_completion:
            check_result, post_production_flows = self.check_for_task_completion(task, plan, group)
            post_production_flows.pop('price_list', None)
        else:
            check_result = None
            post_production_flows = None
        objective = f"'{plan.task.task}'"
        python_code = f"print('Objective {objective} has been completed. Now lets prepare the next objective.')"
        program_parent_id = plan.steps[-2].program.id if len(plan.steps) > 1 else parent_id
        program = Program(id=hash((python_code, plan.task.task, program_parent_id)), code=python_code, conversation=Conversation(messages=[]), parent_id=program_parent_id, version=self.version, version_description=self.version_description, model=self.model_to_evaluate, meta={'objective': plan.task.task, 'type': 'completed_objective', 'checked_result_correct': check_result, 'nr_of_steps': len(plan.steps), 'post_production_flows': post_production_flows})
        return program

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_natural_language_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta: dict) -> List[LanguageOutput]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model)
            outputs = []
            try:
                messages = conversation.model_dump()['messages']
            except:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                str_output = response.content[0].text
                outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.output_tokens + response.usage.input_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            else:
                for choice in response.choices:
                    str_output = choice.message.content
                    outputs.append(LanguageOutput(id=hash((str_output, json.dumps(messages))), response=str_output, conversation=conversation, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta=meta))
            return outputs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def get_inventory_dict(self, inventory):
        inventory_dict = {}
        for item in inventory:
            if isinstance(item, tuple):
                inventory_dict[item[0]] = inventory[item]
            else:
                inventory_dict[item] = inventory[item]
        return inventory_dict

    def format_log_string(self, logs: list):
        return '\n\n'.join(logs) if logs else 'The agent has not yet interacted with the world'

    def _verify_response_is_python(self, content):
        code = content
        try:
            compile(code, filename='<ast>', mode='exec')
        except SyntaxError:
            code = code.rsplit('\n', 1)[0] + '\n'
            compile(code, filename='<ast>', mode='exec')
        return code

    def _extract_code_from_choice(self, input_str) -> Optional[str]:
        """Extract code from a single completion choice"""
        code = ''
        try:
            content = input_str
            code = self._verify_response_is_python(content)
            code = code.replace('from factorio_instance import *', '')
            return (code, None)
        except Exception:
            try:
                content = input_str
                content_split = content.split('```python')
                code = content_split[1].split('```')[0].strip()
                text_response = content_split[0].strip()
                code = self._verify_response_is_python(code)
                code = code.replace('from factorio_instance import *', '')
                code = code.strip()
                return (code, text_response)
            except Exception as e1:
                content = '\n'.join(input_str.split('\n')[1:])
                try:
                    code = self._verify_response_is_python(content)
                    code = code.replace('from factorio_instance import *', '')
                    code = code.strip()
                    return (code, None)
                except Exception:
                    try:
                        content_split = content.split('from factorio_instance import *')
                        code = content_split[1].strip()
                        text_response = content_split[0].strip()
                        code = self._verify_response_is_python(code)
                        code = code.strip()
                        return (code, text_response)
                    except Exception:
                        return ('', content.strip())
                print(f'Failed to extract code from choice: {str(e1)}')

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        conversation = copy.deepcopy(conversation)
        formatted = await self.formatter.format_conversation(conversation)
        formatted_messages = self.formatter.to_llm_messages(formatted)
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=generation_params.presence_penalty)
            programs = []
            try:
                messages = conversation.model_dump()['messages']
            except Exception:
                messages = conversation.dict()['messages']
            if 'claude' in generation_params.model:
                code, text_response = self._extract_code_from_choice(response.content[0].text)
                programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=response.content[0].text, token_usage=response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None, completion_token_usage=response.usage.output_tokens if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.input_tokens if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                if meta:
                    programs[0].meta.update(meta)
            else:
                for choice in response.choices:
                    code, text_response = self._extract_code_from_choice(choice.message.content)
                    programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=choice.message.content, token_usage=response.usage.total_tokens // generation_params.n if hasattr(response, 'usage') else None, completion_token_usage=response.usage.completion_tokens // generation_params.n if hasattr(response, 'usage') else None, prompt_token_usage=response.usage.prompt_tokens // generation_params.n if hasattr(response, 'usage') else None, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                    if meta:
                        programs[-1].meta.update(meta)
            return programs
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

class ParallelMCTS:
    """
    Manages multiple parallel MCTS instances for distributed search
    """

    def __init__(self, instances: List['FactorioInstance'], db_client: DBClient, api_factory: 'APIFactory', config: ParallelMCTSConfig, version: int, version_description: str):
        """
        Initialize parallel MCTS with configuration

        Args:
            instances: List of Factorio instances to distribute across groups
            db_client: Database client for persistence
            api_factory: Factory for creating language models
            config: Configuration parameters
        """
        self.console = Console()
        self.config = config
        self.db_client = db_client
        self.api_factory = api_factory
        self.version = version
        self.version_description = version_description
        self._validate_instance_count(len(instances), config.n_parallel)
        instances_per_group = floor(len(instances) / config.n_parallel)
        self.logger = GroupedFactorioLogger(n_groups=config.n_parallel, instances_per_group=instances_per_group)
        self.logger.start()
        self.instance_groups = self._create_instance_groups(instances)

    def _validate_instance_count(self, total_instances: int, n_parallel: int):
        """Validate that we have enough instances for the requested parallelism"""
        min_required = n_parallel * 2
        if total_instances < min_required:
            raise ValueError(f'Need at least {min_required} instances for {n_parallel} parallel searches (got {total_instances})')
        instances_per_group = floor(total_instances / n_parallel)
        if instances_per_group < 2:
            raise ValueError(f'Not enough instances to allocate at least one active and one holdout instance per group (need {n_parallel * 2}, got {total_instances})')

    def _create_instance_groups(self, instances: List['FactorioInstance']) -> List[InstanceGroup]:
        """Create instance groups for parallel execution"""
        instances_per_group = floor(len(instances) / self.config.n_parallel)
        groups = []
        for group_id in range(self.config.n_parallel):
            start_idx = group_id * instances_per_group
            end_idx = start_idx + instances_per_group
            group_instances = instances[start_idx:end_idx]
            evaluator = Evaluator(db_client=self.db_client, instances=group_instances, value_accrual_time=3, logger=self.logger, error_penalty=self.config.mcts_kwargs['error_penalty'])
            mcts = self.config.mcts_class(api_factory=self.api_factory, db_client=self.db_client, evaluator=evaluator, sampler=self.config.sampler, system_prompt=self.config.system_prompt, initial_state=self.config.initial_state, **self.config.mcts_kwargs)
            groups.append(InstanceGroup(group_id=group_id, mcts=mcts, evaluator=evaluator, active_instances=group_instances))
        return groups

    async def search(self, n_iterations: int, skip_failures: bool=False):
        """
        Run parallel MCTS search across all groups

        Args:
            n_iterations: Number of iterations to run
            skip_failures: Whether to skip failed program generations
        """
        try:
            search_tasks = [self._run_group_search(group, n_iterations, skip_failures) for group in self.instance_groups]
            await asyncio.gather(*search_tasks)
        except Exception as e:
            logger.error(f'Error during parallel search: {str(e)}', exc_info=True)
            raise
        finally:
            self.cleanup()

    async def _run_group_search(self, group: InstanceGroup, n_iterations: int, skip_failures: bool):
        """Run search iterations for a single group"""
        try:
            logger.info(f'Starting search for Group {group.group_id}')
            for iteration in range(n_iterations):
                await group.mcts.run_iteration(len(group.active_instances), skip_failures, iteration, n_iterations)
                self.logger.update_progress()
        except Exception as e:
            logger.error(f'Error in group {group.group_id}: {str(e)}', exc_info=True)
            raise

    def cleanup(self):
        """Clean up resources"""
        self.logger.stop()
        for group in self.instance_groups:
            if hasattr(group.evaluator, 'logger'):
                group.evaluator.logger.stop()

    def get_group_metrics(self, group_id: int) -> Dict[str, Any]:
        """Get metrics for a specific group"""
        if 0 <= group_id < len(self.instance_groups):
            group = self.instance_groups[group_id]
            return {'active_instances': len(group.active_instances), 'total_programs': sum((inst.total_programs for inst in group.evaluator.logger.groups[group_id].instances.values())), 'error_count': sum((inst.error_count for inst in group.evaluator.logger.groups[group_id].instances.values()))}
        return {}

class MCTS:

    def __init__(self, api_factory: 'APIFactory', db_client: DBClient, evaluator: Evaluator, sampler: DBSampler, system_prompt: str, initial_state: GameState, formatter: ConversationFormatter=DefaultFormatter(), version=1, version_description='', presence_penalty=0, frequency_penalty=0, error_penalty=0, maximum_lookback=20):
        self.llm = api_factory
        self.db = db_client
        self.evaluator = evaluator
        self.system_prompt = system_prompt
        self.initial_state = initial_state
        self.sampler = sampler
        self.version = version
        self.version_description = version_description
        self.formatter = formatter
        self.retry_count = 0
        self.max_retries = 3
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.error_penalty = error_penalty
        self.maximum_lookback = maximum_lookback
        self.parser = PythonParser()

    def _is_model_compatible_with_n_samples(self, model):
        """Check if the model is compatible with generating n samples in a single call"""
        return 'gpt' in model or 'o1' in model or 'gemini' in model

    @retry(wait=wait_exponential(multiplier=1, min=3, max=30))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs either through OpenAI's n parameter or parallel calls"""
        conversation = copy.deepcopy(conversation)
        formatted = await self.formatter.format_conversation(conversation)
        formatted_messages = self.formatter.to_llm_messages(formatted)
        try:
            messages = conversation.model_dump()['messages']
        except Exception:
            messages = conversation.dict()['messages']
        try:
            if self._is_model_compatible_with_n_samples(generation_params.model) and hasattr(self.llm, 'acall'):
                response = await self.llm.acall(messages=formatted_messages, n_samples=generation_params.n, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty)
                return await self._process_openai_response(response, conversation, generation_params, messages, meta)
            else:
                return await self._generate_parallel(conversation, generation_params, formatted_messages, messages, meta)
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            raise e

    async def _generate_parallel(self, conversation, generation_params, formatted_messages, messages, meta) -> List[Program]:
        """Generate n programs in parallel for providers that don't support batch generation"""

        async def single_generation():
            try:
                response = await self.llm.acall(messages=formatted_messages, n_samples=1, temperature=generation_params.temperature, max_tokens=generation_params.max_tokens, logit_bias=generation_params.logit_bias, stop_sequences=generation_params.stop_sequences, model=generation_params.model, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty)
                if 'sonnet' in generation_params.model or ('gemini' in generation_params.model and len(formatted_messages) > 32):
                    await sleep(2 + random() * 2)
                return response
            except Exception as e:
                print(f'Single generation failed: {str(e)}')
                return None
        responses = await asyncio.gather(*[single_generation() for _ in range(generation_params.n)], return_exceptions=True)
        programs = []
        for response in responses:
            if response is not None and (not isinstance(response, Exception)):
                program = await self._create_program(response, conversation, messages, generation_params.model, meta)
                if program:
                    programs.append(program)
        return programs

    async def _create_program(self, response, conversation, messages, model, meta) -> Program:
        """Create a Program instance from a single response"""
        if hasattr(response, 'choices'):
            choice = response.choices[0]
            input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') else 0
            total_tokens = input_tokens + output_tokens
        else:
            choice = response.content[0]
            input_tokens = response.usage.input_tokens if hasattr(response, 'usage') else 0
            output_tokens = response.usage.output_tokens if hasattr(response, 'usage') else 0
            total_tokens = input_tokens + output_tokens
        try:
            code, text_response = self.parser.extract_code(choice)
        except Exception as e:
            print(f'Failed to extract code from choice: {str(e)}')
            code = None
        if not code:
            return None
        program = Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=code, token_usage=total_tokens, completion_token_usage=output_tokens, prompt_token_usage=input_tokens, version=self.version, model=model, version_description=self.version_description, meta={'text_response': text_response, 'model': model}, depth=len(messages) - 2)
        if meta:
            program.meta.update(meta)
        return program

    async def _process_openai_response(self, response, conversation, generation_params, messages, meta) -> List[Program]:
        """Process OpenAI's response with multiple choices"""
        programs = []
        total_tokens = completion_tokens = prompt_tokens = 0
        if hasattr(response, 'usage'):
            total_tokens = response.usage.total_tokens if response.usage.total_tokens else response.usage.totalTokens
            completion_tokens = response.usage.completion_tokens if response.usage.completion_tokens else response.usage.completionTokens
            prompt_tokens = response.usage.prompt_tokens if response.usage.prompt_tokens else response.usage.promptTokens
        for choice in response.choices:
            code, text_response = self.parser.extract_code(choice)
            if code:
                programs.append(Program(id=hash((code, json.dumps(messages))), code=code, conversation=conversation, response=choice.message.content, token_usage=total_tokens // generation_params.n, completion_token_usage=completion_tokens // generation_params.n, prompt_token_usage=prompt_tokens // generation_params.n, version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': generation_params.model}))
                if meta:
                    programs[-1].meta.update(meta)
        return programs

    async def search(self, n_iterations: int, samples_per_iteration: int, skip_failures: bool=False):
        """
        Search for the best program using Monte Carlo Tree Search (MCTS).
        :param n_iterations: Number of iterations to perform.
        :param samples_per_iteration: Number of programs to sample per iteration.
        :param skip_failures: Whether to skip saving failed program generations.
        """
        for iteration in range(n_iterations):
            print(f'Starting iteration {iteration}')
            await self.run_iteration(samples_per_iteration, skip_failures, iteration, n_iterations)
            self.evaluator.logger.update_progress()

    @tenacity.retry(retry=retry_if_exception_type(psycopg2.Error), wait=wait_exponential(multiplier=1, min=1, max=4), stop=tenacity.stop_after_attempt(3))
    async def run_iteration(self, samples_per_iteration, skip_failures, iteration, n_iterations):
        """Run a single MCTS iteration with retries for concurrent operations"""
        try:
            parent = await self.sampler.sample_parent(version=self.version, maximum_lookback=self.maximum_lookback)
            if parent:
                start_state = parent.state
                conversation = parent.conversation
            else:
                start_state = self.initial_state
                self.evaluator.instances[0].reset(start_state)
                entities = self.evaluator.instances[0].get_entities()
                conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='assistant', content="print(f'Inventory: {inspect_inventory()}')\nprint(f'Entities: {get_entities()}')\n"), Message(role='user', content=f"1: ('Inventory: {start_state.inventories[0].__dict__}')\n2: ('Entities: {entities}')")])
            self.evaluator.set_sampling_status()
            self.evaluator.set_iteration(iteration, n_iterations)
            generation_parameters = GenerationParameters(n=samples_per_iteration, model=self.llm.model, presence_penalty=0.7)
            programs = await self._generate_programs_batch(conversation, generation_parameters)
            if not programs:
                return
            programs = [p for p in programs if p is not None]
            for program in programs:
                program.parent_id = parent.id if parent else None
            evaluated_programs = await self.evaluator.evaluate_batch(programs, start_state)
            save_tasks = []
            for program in evaluated_programs:
                if program.state is not None:
                    if not skip_failures or program.value is not None:
                        save_tasks.append(self.db.create_program(program))
            if save_tasks:
                await asyncio.gather(*save_tasks)
                await self.sampler.visit(parent.id, len(save_tasks))
        except Exception as e:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                print(f'Max retries ({self.max_retries}) reached. Error: {str(e)}')
                self.retry_count = 0
                raise e
            raise e

class ObjectiveMCTS(MCTS):

    def __init__(self, api_factory: 'APIFactory', db_client: DBClient, evaluator: Evaluator, sampler: DBSampler, system_prompt: str, initial_state: GameState, formatter: ConversationFormatter=DefaultFormatter(), version=1, version_description='', logit_bias=[], presence_penalty=0, frequency_penalty=0, objective_model: str='ft:gpt-4o-mini-2024-07-18:paperplane-ai:plans-tree:AcZ8gHSo'):
        self.logit_bias = logit_bias
        self.objective_tree_sampler = ObjectiveTreeSampler(APIFactory(model=objective_model))
        super().__init__(api_factory, db_client, evaluator, sampler, system_prompt, initial_state, formatter, version, version_description, presence_penalty, frequency_penalty)

    async def _get_objectives(self, conversation: Conversation) -> List[str]:
        if len(conversation.messages) == 0:
            previous_objectives = []
        elif 'objectives' not in conversation.messages[-1].metadata:
            previous_objectives = []
        elif not conversation.messages[-1].metadata['objectives']:
            previous_objectives = []
        else:
            previous_objectives = conversation.messages[-1].metadata['objectives']
        objective = await self.objective_tree_sampler.sample_tree(previous_objectives, number=1)
        return previous_objectives + objective

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _generate_programs_batch(self, conversation: Conversation, generation_params: GenerationParameters, meta={}) -> List[Program]:
        """Generate multiple programs in a single API call using 'n' parameter"""
        if 'objectives' not in meta:
            objectives = await self._get_objectives(conversation)
            meta['objectives'] = objectives
            objective = objectives[-1]
            self._append_inventory_check_messages(conversation, objective)
            conversation.messages[-1].metadata = {**conversation.messages[-1].metadata, 'objectives': objectives}
        formatted_messages = self.formatter.to_llm_messages(self.formatter.format_conversation(conversation))
        if 'claude' in generation_params.model:
            assert generation_params.n == 1, 'Number of samples must be 1 for Claude'
        try:
            response = await self._generate_llm_response(formatted_messages, generation_params)
            return await self._process_llm_response(response, conversation, generation_params, meta)
        except Exception as e:
            print(f'Batch program generation failed: {str(e)}')
            return []

    def _append_inventory_check_messages(self, conversation: Conversation, objective: str):
        """Append inventory check messages to the conversation"""
        conversation.messages.extend([Message(role='assistant', content=f'"""\nObjective: {objective}\n"""\nprint("Inventory: ", inspect_inventory())\nprint("Entities: ", get_entities())\n', metadata={'objectives': [objective]}), Message(role='user', content="Execution Result: \n0: ('Inventory: ', {})\n1: ('Entities: ': {})", metadata={'objectives': [objective]})])

    async def _generate_llm_response(self, formatted_messages: list, params: GenerationParameters):
        """Generate response from LLM with given parameters"""
        return await self.llm.acall(messages=formatted_messages, n_samples=params.n, temperature=params.temperature, max_tokens=params.max_tokens, logit_bias=params.logit_bias, stop_sequences=params.stop_sequences, model=params.model, presence_penalty=self.presence_penalty, frequency_penalty=self.frequency_penalty)

    async def _process_llm_response(self, response, conversation: Conversation, params: GenerationParameters, meta: dict) -> List[Program]:
        """Process LLM response and create Program objects"""
        programs = []
        try:
            messages = conversation.model_dump()['messages']
        except Exception:
            messages = conversation.dict()['messages']
        if 'claude' in params.model:
            programs = await self._handle_claude_response(response, messages, meta, params.model)
        else:
            programs = await self._handle_openai_response(response, messages, meta, params)
        return programs

    async def _handle_claude_response(self, response, messages, meta, model):
        """Handle Claude-specific response format"""
        programs = []
        code, text_response = self._extract_code_from_choice(response)
        if not code:
            objectives = await self._get_objectives(Conversation(messages=[Message(**msg.dict()) for msg in messages]))
            code = f'"""\nObjective: {objectives[-1]}\n"""'
        if code:
            new_conversation = Conversation(messages=[Message(**msg.dict()) for msg in messages])
            objectives = await self._get_objectives(new_conversation)
            self._append_inventory_check_messages(new_conversation, objectives[-1])
            program = self._create_program(code=code, messages=new_conversation.messages, conversation=new_conversation, response_content=response.message.content, token_usage=self._get_token_usage(response), model=model, text_response=text_response, meta={**meta, 'objectives': objectives})
            programs.append(program)
        return programs

    async def _handle_openai_response(self, response, messages, meta, params):
        """Handle OpenAI response format with multiple choices"""
        programs = []
        for choice in response.choices:
            code, text_response = self._extract_code_from_choice(choice)
            objectives = messages[-1]['metadata']['objectives']
            if not code:
                new_conversation = Conversation(messages=[Message(**msg) for msg in messages])
                objectives = await self._get_objectives(new_conversation)
                code = f'"""\nObjective: {objectives[-1]}\n"""'
            if code:
                new_conversation = Conversation(messages=[Message(**msg) for msg in messages])
                program = self._create_program(code=code, messages=new_conversation.messages, conversation=new_conversation, response_content=choice.message.content, token_usage=self._get_token_usage(response, divide_by_n=params.n), model=params.model, text_response=text_response, meta={**meta, 'objectives': objectives})
                programs.append(program)
        return programs

    def _create_program(self, code, messages, conversation, response_content, token_usage, model, text_response, meta):
        """Create a Program object with given parameters"""
        program = Program(id=hash((code, json.dumps([msg.__dict__ if isinstance(msg, Message) else msg for msg in messages]))), code=code, conversation=conversation, response=response_content, token_usage=token_usage.get('total'), completion_token_usage=token_usage.get('completion'), prompt_token_usage=token_usage.get('prompt'), version=self.version, version_description=self.version_description, meta={'text_response': text_response, 'model': model})
        if meta:
            program.meta.update(meta)
        return program

    def _get_token_usage(self, response, divide_by_n=1):
        """Extract token usage from response"""
        if not hasattr(response, 'usage'):
            return {'total': None, 'completion': None, 'prompt': None}
        return {'total': response.usage.total_tokens // divide_by_n, 'completion': response.usage.completion_tokens // divide_by_n, 'prompt': response.usage.prompt_tokens // divide_by_n}

    async def search(self, n_iterations: int, samples_per_iteration: int, skip_failures: bool=False):
        """
        Search for the best program using Monte Carlo Tree Search (MCTS).
        :param n_iterations: Number of iterations to perform.
        :param samples_per_iteration: Number of programs to sample per iteration.
        :param skip_failures: Whether to skip saving failed program generations.
        """
        for iteration in range(n_iterations):
            print(f'Starting iteration {iteration}')
            await self.run_iteration(samples_per_iteration, skip_failures)
            self.evaluator.logger.update_progress()

    @tenacity.retry(retry=retry_if_exception_type(psycopg2.Error), wait=wait_exponential(multiplier=1, min=1, max=4), stop=tenacity.stop_after_attempt(3))
    async def run_iteration(self, samples_per_iteration, skip_failures):
        """Run a single MCTS iteration with retries for concurrent operations"""
        try:
            parent = await self.sampler.sample_parent(version=self.version)
            if parent:
                start_state = parent.state
                conversation = parent.conversation
            else:
                start_state = self.initial_state
                conversation = Conversation(messages=[Message(role='system', content=self.system_prompt), Message(role='user', content=OBJECTIVE_PLANNING_PROMPT, metadata={'objectives': ['1. Automate resource production']})])
            self.evaluator.set_sampling_status()
            generation_parameters = GenerationParameters(n=samples_per_iteration, model=self.llm.model, stop_sequences=['Objective:'], logit_bias=self.logit_bias, presence_penalty=0.7)
            programs = await self._generate_programs_batch(conversation, generation_parameters)
            if not programs:
                return
            programs = [p for p in programs if p is not None]
            for program in programs:
                program.parent_id = parent.id if parent else None
            evaluated_programs = await self.evaluator.evaluate_batch(programs, start_state)
            save_tasks = []
            for program in evaluated_programs:
                if program.state is not None:
                    if not skip_failures or program.value is not None:
                        save_tasks.append(self.db.create_program(program))
            if save_tasks:
                await asyncio.gather(*save_tasks)
        except Exception as e:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                print(f'Max retries ({self.max_retries}) reached. Error: {str(e)}')
                self.retry_count = 0
                raise e
            raise e

class TestChunkBasedPythonParser(unittest.TestCase):

    def setUp(self):
        self.parser = PythonParser()

    def test_valid_python_chunks(self):
        """Test handling of valid Python code chunks."""
        test_content = 'print("Hello")\n\nx = 5\ny = 10\nprint(x + y)\n\ndef test_func():\n    return 42'

        class MockResponse:

            def __init__(self, content):
                self.message = type('Message', (), {'content': content})
        result = self.parser.extract_code(MockResponse(test_content))
        self.assertIsNotNone(result)
        code, original = result
        self.assertIn('print("Hello")', code)
        self.assertIn('x = 5', code)
        self.assertIn('def test_func():', code)
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            self.fail('Resulting code is not valid Python')

    def test_markdown_chunks(self):
        """Test handling of markdown-style documentation chunks."""
        test_content = '# Step 1: Initialize variables\n\nx = 5\ny = 10\n\n## Processing steps:\n1. Add numbers\n2. Print result\n\nresult = x + y\nprint(result)'

        class MockResponse:

            def __init__(self, content):
                self.message = type('Message', (), {'content': content})
        result = self.parser.extract_code(MockResponse(test_content))
        self.assertIsNotNone(result)
        code, original = result
        self.assertIn('"""', code)
        self.assertIn('# Step 1:', code)
        self.assertIn('x = 5', code)
        self.assertIn('result = x + y', code)
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            self.fail('Resulting code is not valid Python')

    def test_mixed_content(self):
        """Test handling of mixed markdown and code content."""
        test_content = '# First we need to check inventory\n\ncurrent_inventory = inspect_inventory()\nrequired_stone = 5\n\n## Next steps:\n* Get more resources\n* Build furnace\n\nif current_inventory["stone"] < required_stone:\n    gather_stone()'

        class MockResponse:

            def __init__(self, content):
                self.message = type('Message', (), {'content': content})
        result = self.parser.extract_code(MockResponse(test_content))
        self.assertIsNotNone(result)
        code, original = result
        code_lines = code.split('\n')
        docstring_count = sum((1 for line in code_lines if '"""' in line))
        self.assertEqual(docstring_count % 2, 0, 'Unmatched docstring delimiters')
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            self.fail('Resulting code is not valid Python')

    def test_empty_chunks(self):
        """Test handling of empty chunks and whitespace."""
        test_content = '\n\nprint("First")\n\n\nprint("Second")\n\n'

        class MockResponse:

            def __init__(self, content):
                self.message = type('Message', (), {'content': content})
        result = self.parser.extract_code(MockResponse(test_content))
        self.assertIsNotNone(result)
        code, original = result
        self.assertIn('print("First")', code)
        self.assertIn('print("Second")', code)
        self.assertNotIn('"""\n"""', code)
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            self.fail('Resulting code is not valid Python')

    def test_code_block_markers(self):
        """Test handling of markdown code block markers."""
        test_content = 'Some explanation here\n\n```python\ndef test_func():\n    return 42\n```\n\nMore explanation\n\n```\nx = 5\nprint(x)\n```'

        class MockResponse:

            def __init__(self, content):
                self.message = type('Message', (), {'content': content})
        result = self.parser.extract_code(MockResponse(test_content))
        self.assertIsNotNone(result)
        code, original = result
        self.assertIn('def test_func():', code)
        self.assertIn('x = 5', code)
        self.assertNotIn('```', code)
        self.assertIn('"""Some explanation here"""', code.replace('\n', ''))
        import ast
        try:
            ast.parse(code)
        except SyntaxError:
            self.fail('Resulting code is not valid Python')

