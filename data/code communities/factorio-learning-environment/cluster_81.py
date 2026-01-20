# Cluster 81

class Evaluator:

    def __init__(self, db_client: DBClient, instances: List[FactorioInstance], value_accrual_time=10, error_penalty=10, logger=None):
        self.db = db_client
        self.instances = instances
        self.value_accrual_time = value_accrual_time
        self.error_penalty = error_penalty
        if not logger:
            self.logger = FactorioLogger(len(instances))
            self.logger.start()
        else:
            self.logger = logger
        self.instance_to_port = {i: instance.tcp_port for i, instance in enumerate(self.instances)}
        if logger:
            self.port_to_group = logger.port_to_group

    def set_status(self, status):
        for instance in self.instances:
            self.logger.update_instance(instance.tcp_port, status=status)

    def set_sampling_status(self):
        """Update status for all instances in this evaluator's group"""
        if self.logger:
            for instance in self.instances:
                self.logger.update_instance(instance.tcp_port, status='sampling')

    def set_iteration(self, iteration, n_iterations):
        """Update iteration number for all instances in this evaluator's group"""
        if self.logger:
            for instance in self.instances:
                self.logger.update_instance(instance.tcp_port, iteration=iteration, n_iterations=n_iterations)

    async def evaluate_batch(self, programs: List[Program], start_state: GameState) -> List[Program]:
        try:
            eval_futures = []
            for i, (prog, inst) in enumerate(zip(programs, self.instances)):
                inst.reset(start_state)
                if self.logger:
                    self.logger.update_instance(inst.tcp_port, program_id=prog.id, status='resetting')
                eval_futures.append(self._evaluate_single(inst.tcp_port, prog, inst))
            eval_results = await asyncio.gather(*eval_futures)
            for i, (program, (raw_reward, state, response, entities, achievements, ticks)) in enumerate(zip(programs, eval_results)):
                relative_reward = raw_reward
                if self.logger:
                    self.logger.update_instance(self.instances[i].tcp_port, status='completed', raw_reward=raw_reward, holdout_value=raw_reward, relative_reward=relative_reward, total_programs=self.logger.groups[self.port_to_group[self.instances[i].tcp_port]].instances[self.instances[i].tcp_port].total_programs + 1)
                program.value = relative_reward
                program.state = state
                program.raw_reward = raw_reward
                program.ticks = ticks
                conversation = copy.deepcopy(program.conversation)
                conversation.add_result(program.code, response, score=raw_reward, advantage=relative_reward, objectives=program.meta['objectives'] if 'objectives' in program.meta else [])
                program.conversation = conversation
                program.response = response
                program.achievements = achievements
            return programs
        except Exception as e:
            if self.logger:
                for instance in self.instances:
                    self.logger.update_instance(instance.tcp_port, status='error', error_count=self.logger.groups[self.port_to_group[instance.tcp_port]].instances[instance.tcp_port].error_count + 1)
            raise e

    def _evaluate_for_achievements(self, code: str, instance: FactorioInstance) -> Tuple[float, GameState, str, List[Union[Entity, EntityGroup]], Dict[str, Dict[str, int]]]:
        start_production_flows = instance.namespace._get_production_stats()
        reward, time, result = instance.eval(code, timeout=120)
        post_production_flows = instance.namespace._get_production_stats()
        achievements = get_achievements(start_production_flows, copy.deepcopy(post_production_flows))
        return (result, achievements, post_production_flows)

    async def _evaluate_single(self, instance_id: int, program: Program, instance: FactorioInstance) -> Tuple[float, GameState, str, List[Union[Entity, EntityGroup]], Dict[str, Dict[str, int]], int]:
        try:
            tcp_port = self.instance_to_port[instance_id]
        except:
            tcp_port = instance_id
        try:
            self.logger.update_instance(tcp_port, status='starting value')
            start_entities = instance.namespace.get_entities()
            start_inventory = instance.namespace.inspect_inventory()
            start_production_flows = instance.namespace._get_production_stats()
            initial_value, start_time = instance.namespace.score()
            self.logger.update_instance(tcp_port, status='executing')
            reward, time, result = instance.eval(program.code, timeout=60)
            self.logger.update_instance(tcp_port, status='capturing state')
            state = GameState.from_instance(instance)
            self.logger.update_instance(tcp_port, status=f'accruing value ({self.value_accrual_time}s)')
            await asyncio.sleep(self.value_accrual_time)
            entities = instance.namespace.get_entities()
            final_inventory = instance.namespace.inspect_inventory()
            get_inventory_code = 'print(f"Current inventory {inspect_inventory()}")'
            if start_inventory.__dict__ != final_inventory.__dict__ and 'error' not in result.lower() and (get_inventory_code not in program.code) and ('inspect_inventory()' not in program.code):
                program.code += f'\n{get_inventory_code}'
                result += '\n' + str(len(program.code.split('\n'))) + f": ('Current inventory {final_inventory}',)"
            get_entities_code = 'print(f"Entities on the map: {get_entities()}")'
            if start_entities != entities and 'error' not in result.lower() and (get_entities_code not in program.code) and ('get_entities()' not in program.code):
                program.code += f'\n{get_entities_code}\n'
                result += '\n' + str(len(program.code.split('\n'))) + f": ('Entities on the map: {entities}',)"
            result = result.rstrip() + '\n'
            if 'error' in result.lower():
                result += f"('Current inventory: {final_inventory}',)\n"
                result += f"('Entities on the map after the current step: {entities}',)"
            score, _ = instance.namespace.score()
            final_reward = score - initial_value
            ticks = instance.get_elapsed_ticks()
            post_production_flows = instance.namespace._get_production_stats()
            achievements = get_achievements(start_production_flows, post_production_flows)
            group_id = self.port_to_group[tcp_port]
            group = self.logger.groups[group_id]
            instance_metrics = group.instances[tcp_port]
            self.logger.update_instance(tcp_port, status='accrued value', current_reward=final_reward, raw_reward=final_reward, final_entities=len(entities), start_entities=len(start_entities), total_programs=instance_metrics.total_programs + 1, start_inventory_count=sum([v for k, v in start_inventory.__dict__.items() if v > 0]), final_inventory_count=sum([v for k, v in final_inventory.__dict__.items() if v > 0]))
            if 'error' in result.lower() and self.logger:
                group_id = self.port_to_group[tcp_port]
                group = self.logger.groups[group_id]
                instance_metrics = group.instances[tcp_port]
                self.logger.update_instance(tcp_port, status='error', error_count=instance_metrics.error_count + 1)
            return (final_reward, state, result, entities, achievements, ticks)
        except Exception as e:
            print('Error in _evaluate_single:')
            print(f'Instance ID: {instance_id}')
            print(f'TCP Port: {self.instance_to_port.get(instance_id, 'Unknown')}')
            print(f'Error: {str(e)}')
            import traceback
            traceback.print_exc()
            if self.logger:
                tcp_port = self.instance_to_port[instance_id]
                group_id = self.port_to_group[tcp_port]
                group = self.logger.groups[group_id]
                instance_metrics = group.instances[tcp_port]
                self.logger.update_instance(tcp_port, status='error', error_count=instance_metrics.error_count + 1)
            raise e

    def __del__(self):
        """Clean up logger on deletion"""
        self.logger.stop()

