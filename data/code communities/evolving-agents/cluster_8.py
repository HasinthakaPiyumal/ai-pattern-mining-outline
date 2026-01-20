# Cluster 8

class ABTestOpenAIAgentTool(Tool[ABTestInput, None, StringToolOutput]):
    """
    Tool for A/B testing two OpenAI agents to compare their performance.
    This tool can help determine if an evolved agent performs better than its predecessor.
    """
    name = 'ABTestOpenAIAgentTool'
    description = 'Compare two OpenAI agents on the same tasks to measure performance differences'
    input_schema = ABTestInput

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, agent_factory: AgentFactory, agent_logger: Optional[OpenAIAgentLogger]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.llm = llm_service
        self.agent_factory = agent_factory
        self.agent_logger = agent_logger or OpenAIAgentLogger()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'openai', 'ab_test'], creator=self)

    async def _run(self, input: ABTestInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Run an A/B test comparing two OpenAI agents.
        
        Args:
            input: The test parameters
        
        Returns:
            StringToolOutput containing the test results in JSON format
        """
        try:
            agent_a_record = await self.library.find_record_by_id(input.agent_a_id)
            agent_b_record = await self.library.find_record_by_id(input.agent_b_id)
            if not agent_a_record or not agent_b_record:
                return StringToolOutput(json.dumps({'status': 'error', 'message': 'One or both agents not found'}, indent=2))
            agent_a = await self.agent_factory.create_agent(agent_a_record)
            agent_b = await self.agent_factory.create_agent(agent_b_record)
            criteria = input.evaluation_criteria or ['accuracy', 'response_time', 'completeness']
            results_a = []
            results_b = []
            for test_input in input.test_inputs:
                start_time_a = time.time()
                result_a = await self.agent_factory.execute_agent(agent_a, test_input)
                response_time_a = time.time() - start_time_a
                start_time_b = time.time()
                result_b = await self.agent_factory.execute_agent(agent_b, test_input)
                response_time_b = time.time() - start_time_b
                evaluation = await self._evaluate_responses(test_input, result_a['result'], result_b['result'], criteria, input.domain)
                results_a.append({'input': test_input, 'output': result_a['result'], 'response_time': response_time_a, 'scores': evaluation['scores_a']})
                results_b.append({'input': test_input, 'output': result_b['result'], 'response_time': response_time_b, 'scores': evaluation['scores_b']})
                if self.agent_logger:
                    self.agent_logger.record_invocation(agent_a_record['id'], agent_a_record['name'], input.domain, test_input, evaluation['winner'] == 'A' or evaluation['winner'] == 'Tie', response_time_a)
                    self.agent_logger.record_invocation(agent_b_record['id'], agent_b_record['name'], input.domain, test_input, evaluation['winner'] == 'B' or evaluation['winner'] == 'Tie', response_time_b)
            agent_a_scores = {criterion: statistics.mean([result['scores'].get(criterion, 0) for result in results_a]) for criterion in criteria}
            agent_a_scores['average_response_time'] = statistics.mean([result['response_time'] for result in results_a])
            agent_b_scores = {criterion: statistics.mean([result['scores'].get(criterion, 0) for result in results_b]) for criterion in criteria}
            agent_b_scores['average_response_time'] = statistics.mean([result['response_time'] for result in results_b])
            a_wins = sum((1 for a, b in zip(results_a, results_b) if sum(a['scores'].values()) > sum(b['scores'].values())))
            b_wins = sum((1 for a, b in zip(results_a, results_b) if sum(a['scores'].values()) < sum(b['scores'].values())))
            ties = len(results_a) - a_wins - b_wins
            total_a_score = sum(agent_a_scores.values())
            total_b_score = sum(agent_b_scores.values())
            percentage_difference = (total_b_score - total_a_score) / total_a_score * 100
            recommendations = await self._generate_improvement_recommendations(agent_a_record, agent_b_record, agent_a_scores, agent_b_scores, results_a, results_b)
            return StringToolOutput(json.dumps({'status': 'success', 'test_summary': {'agent_a': {'id': agent_a_record['id'], 'name': agent_a_record['name'], 'version': agent_a_record['version']}, 'agent_b': {'id': agent_b_record['id'], 'name': agent_b_record['name'], 'version': agent_b_record['version']}, 'total_tests': len(input.test_inputs), 'agent_a_wins': a_wins, 'agent_b_wins': b_wins, 'ties': ties, 'percentage_difference': f'{percentage_difference:.2f}%', 'overall_winner': 'Agent A' if a_wins > b_wins else 'Agent B' if b_wins > a_wins else 'Tie'}, 'agent_a_scores': agent_a_scores, 'agent_b_scores': agent_b_scores, 'detailed_results': {'agent_a': results_a, 'agent_b': results_b}, 'improvement_recommendations': recommendations}, indent=2))
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({'status': 'error', 'message': f'Error running A/B test: {str(e)}', 'details': traceback.format_exc()}, indent=2))

    async def _evaluate_responses(self, test_input: str, response_a: str, response_b: str, criteria: List[str], domain: str) -> Dict[str, Any]:
        """Evaluate and compare two responses based on the given criteria."""
        evaluation_prompt = f'\n        You are an impartial judge evaluating the performance of two AI assistants.\n        \n        TASK INPUT:\n        {test_input}\n        \n        RESPONSE FROM ASSISTANT A:\n        {response_a}\n        \n        RESPONSE FROM ASSISTANT B:\n        {response_b}\n        \n        DOMAIN: {domain}\n        \n        Please evaluate both responses on the following criteria (score from 0-10):\n        {', '.join(criteria)}\n        \n        For each criterion, provide:\n        1. A score for Assistant A\n        2. A score for Assistant B\n        3. A brief explanation for the scores\n        \n        Finally, declare a winner or a tie based on overall performance.\n        \n        Return your evaluation in JSON format with this structure:\n        {{\n          "scores_a": {{"criterion1": score, "criterion2": score, ...}},\n          "scores_b": {{"criterion1": score, "criterion2": score, ...}},\n          "explanations": {{"criterion1": "explanation", ...}},\n          "winner": "A", "B", or "Tie",\n          "reasoning": "explanation for the overall winner decision"\n        }}\n        '
        evaluation_response = await self.llm.generate(evaluation_prompt)
        try:
            evaluation = json.loads(evaluation_response)
            return evaluation
        except json.JSONDecodeError:
            default_scores = {criterion: 5.0 for criterion in criteria}
            return {'scores_a': default_scores, 'scores_b': default_scores, 'explanations': {criterion: 'Evaluation parsing failed' for criterion in criteria}, 'winner': 'Tie', 'reasoning': 'Could not determine a winner due to evaluation parsing failure'}

    async def _generate_improvement_recommendations(self, agent_a_record: Dict[str, Any], agent_b_record: Dict[str, Any], agent_a_scores: Dict[str, float], agent_b_scores: Dict[str, float], results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for further improvements based on test results."""
        a_strengths = [criterion for criterion, score in agent_a_scores.items() if score > agent_b_scores.get(criterion, 0)]
        b_strengths = [criterion for criterion, score in agent_b_scores.items() if score > agent_a_scores.get(criterion, 0)]
        a_weaknesses = [criterion for criterion, score in agent_a_scores.items() if score < agent_b_scores.get(criterion, 0)]
        b_weaknesses = [criterion for criterion, score in agent_b_scores.items() if score < agent_a_scores.get(criterion, 0)]
        is_a_evolved_from_b = agent_a_record.get('parent_id') == agent_b_record['id']
        is_b_evolved_from_a = agent_b_record.get('parent_id') == agent_a_record['id']
        evolution_relationship = None
        if is_a_evolved_from_b:
            evolution_relationship = f'{agent_a_record['name']} is evolved from {agent_b_record['name']}'
        elif is_b_evolved_from_a:
            evolution_relationship = f'{agent_b_record['name']} is evolved from {agent_a_record['name']}'
        recommendation_prompt = f'\n        Based on A/B testing results between two AI agents, generate specific recommendations for further improvements.\n        \n        AGENT A: {agent_a_record['name']} (Version: {agent_a_record['version']})\n        AGENT B: {agent_b_record['name']} (Version: {agent_b_record['version']})\n        \n        {(evolution_relationship if evolution_relationship else 'The agents are separate implementations.')}\n        \n        STRENGTHS:\n        - Agent A excels in: {(', '.join(a_strengths) if a_strengths else 'No clear strengths')}\n        - Agent B excels in: {(', '.join(b_strengths) if b_strengths else 'No clear strengths')}\n        \n        WEAKNESSES:\n        - Agent A is weaker in: {(', '.join(a_weaknesses) if a_weaknesses else 'No clear weaknesses')}\n        - Agent B is weaker in: {(', '.join(b_weaknesses) if b_weaknesses else 'No clear weaknesses')}\n        \n        SAMPLE TEST ITEMS THAT SHOWED LARGEST DIFFERENCES:\n        {chr(10).join([f'- Input: {a['input'][:100]}...' for a, b in zip(results_a[:3], results_b[:3]) if abs(sum(a['scores'].values()) - sum(b['scores'].values())) > 3])}\n        \n        Provide 3-5 specific, actionable recommendations for further agent evolution that would:\n        1. Combine strengths of both agents\n        2. Address key weaknesses\n        3. Represent a clear improvement over both existing agents\n        \n        Format each recommendation as a clear instruction that could be used in an evolution strategy.\n        '
        recommendation_response = await self.llm.generate(recommendation_prompt)
        recommendations = [line.strip().lstrip('-').strip() for line in recommendation_response.split('\n') if line.strip() and line.strip().startswith('-')]
        if not recommendations:
            recommendations = [recommendation_response.strip()]
        return recommendations

class EvolveOpenAIAgentTool(Tool[EvolveOpenAIAgentInput, None, StringToolOutput]):
    """
    Tool for evolving OpenAI agents based on experiences and requirements.
    This tool uses the agent's performance history and user-specified changes to
    create an improved version of the agent.
    """
    name = 'EvolveOpenAIAgentTool'
    description = 'Evolves OpenAI agents through different strategies to adapt to new requirements or improve performance'
    input_schema = EvolveOpenAIAgentInput

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, firmware: Optional[Firmware]=None, agent_logger: Optional[OpenAIAgentLogger]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.library = smart_library
        self.llm = llm_service
        self.firmware = firmware or Firmware()
        self.agent_logger = agent_logger or OpenAIAgentLogger()
        self.evolution_strategies = {'standard': {'description': 'Balanced evolution that enhances instructions while maintaining core behavior', 'instruction_preservation': 0.7, 'prompt_modifier': "Enhance the agent's instructions with new capabilities while preserving its core behavior."}, 'conservative': {'description': 'Minimal changes focused on fixing specific issues while maintaining compatibility', 'instruction_preservation': 0.9, 'prompt_modifier': "Make minor adjustments to the agent's instructions to address specific issues while preserving its behavior."}, 'aggressive': {'description': 'Significant changes to optimize for new requirements', 'instruction_preservation': 0.4, 'prompt_modifier': "Substantially revise the agent's instructions to optimize for the new requirements."}, 'domain_adaptation': {'description': 'Specialized adaptation to a new domain', 'instruction_preservation': 0.6, 'prompt_modifier': "Adapt the agent's instructions for the new domain while maintaining its core capabilities."}}

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'openai', 'evolve_agent'], creator=self)

    async def _run(self, input: EvolveOpenAIAgentInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        """
        Evolve an OpenAI agent based on the specified strategy and changes.
        
        Args:
            input: The evolution parameters
        
        Returns:
            StringToolOutput containing the evolution result in JSON format
        """
        try:
            agent_record = await self._get_agent_record(input.agent_id_or_name)
            if not agent_record:
                return StringToolOutput(json.dumps({'status': 'error', 'message': f"Agent '{input.agent_id_or_name}' not found"}, indent=2))
            metadata = agent_record.get('metadata', {})
            framework = metadata.get('framework', '').lower()
            if framework != 'openai-agents':
                return StringToolOutput(json.dumps({'status': 'error', 'message': f"Agent '{input.agent_id_or_name}' is not an OpenAI agent"}, indent=2))
            strategy = input.evolution_type.lower()
            if strategy not in self.evolution_strategies:
                strategy = 'standard'
            strategy_details = self.evolution_strategies[strategy]
            current_instructions = self._extract_instructions(agent_record['code_snippet'])
            experience_data = {}
            if input.learning_from_experience and self.agent_logger:
                experience = self.agent_logger.get_experience(agent_record['id'])
                if experience:
                    experience_data = {'total_invocations': experience.total_invocations, 'success_rate': experience.successful_invocations / max(1, experience.total_invocations), 'average_response_time': experience.average_response_time, 'domain_performance': experience.domain_performance, 'common_inputs': sorted(experience.input_patterns.items(), key=lambda x: x[1], reverse=True)[:5], 'recent_failures': experience.recent_failures}
            target_domain = input.target_domain or agent_record.get('domain', 'general')
            firmware_content = self.firmware.get_firmware_prompt(target_domain)
            evolved_instructions = await self._generate_evolved_instructions(current_instructions, input.changes, strategy_details, experience_data, firmware_content, target_domain)
            evolved_code = self._generate_evolved_code(agent_record['code_snippet'], evolved_instructions)
            evolved_record = await self.library.evolve_record(parent_id=agent_record['id'], new_code_snippet=evolved_code, description=f'{agent_record['description']} (Evolved with {strategy} strategy)', status='active')
            if 'metadata' not in evolved_record:
                evolved_record['metadata'] = {}
            evolved_record['metadata'].update({'framework': 'openai-agents', 'evolution_strategy': strategy, 'evolution_timestamp': self._get_current_timestamp()})
            if input.target_domain:
                evolved_record['domain'] = input.target_domain
                evolved_record['metadata']['previous_domain'] = agent_record.get('domain', 'general')
            await self.library.save_record(evolved_record)
            if self.agent_logger:
                self.agent_logger.record_evolution(agent_record['id'], strategy, {'new_agent_id': evolved_record['id'], 'changes': input.changes, 'target_domain': target_domain})
            return StringToolOutput(json.dumps({'status': 'success', 'message': f"Successfully evolved OpenAI agent '{agent_record['name']}' using '{strategy}' strategy", 'original_agent_id': agent_record['id'], 'evolved_agent_id': evolved_record['id'], 'evolution_strategy': {'name': strategy, 'description': strategy_details['description']}, 'evolved_agent': {'name': evolved_record['name'], 'version': evolved_record['version'], 'domain': evolved_record['domain']}}, indent=2))
        except Exception as e:
            import traceback
            return StringToolOutput(json.dumps({'status': 'error', 'message': f'Error evolving OpenAI agent: {str(e)}', 'details': traceback.format_exc()}, indent=2))

    async def _get_agent_record(self, agent_id_or_name: str) -> Optional[Dict[str, Any]]:
        """Get an agent record by ID or name."""
        record = await self.library.find_record_by_id(agent_id_or_name)
        if record:
            return record
        return await self.library.find_record_by_name(agent_id_or_name, 'AGENT')

    def _extract_instructions(self, code_snippet: str) -> str:
        """Extract the instructions from an OpenAI agent's code snippet."""
        import re
        instruction_pattern = 'instructions=(?:"""|\\\'\\\'\\\')(.*?)(?:"""|\\\'\\\'\\\')'
        match = re.search(instruction_pattern, code_snippet, re.DOTALL)
        if match:
            return match.group(1).strip()
        alt_pattern = 'instructions=[\\\'"](.*?)[\\\'"]'
        match = re.search(alt_pattern, code_snippet, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ''

    async def _generate_evolved_instructions(self, current_instructions: str, changes: str, strategy_details: Dict[str, Any], experience_data: Dict[str, Any], firmware_content: str, target_domain: str) -> str:
        """Generate evolved instructions for the OpenAI agent."""
        prompt_modifier = strategy_details.get('prompt_modifier', '')
        instruction_preservation = strategy_details.get('instruction_preservation', 0.7)
        experience_text = ''
        if experience_data:
            experience_text = f'\nAGENT EXPERIENCE DATA:\n- Total Invocations: {experience_data.get('total_invocations', 0)}\n- Success Rate: {experience_data.get('success_rate', 0) * 100:.1f}%\n- Average Response Time: {experience_data.get('average_response_time', 0):.2f} seconds\n- Domain Performance: {', '.join([f'{domain}: {score:.2f}' for domain, score in experience_data.get('domain_performance', {}).items()])}\n\nCommon Input Patterns:\n{chr(10).join([f'- {pattern}: {count} times' for pattern, count in experience_data.get('common_inputs', [])])}\n\nRecent Failures:\n{chr(10).join([f'- Domain: {failure.get('domain')}, Pattern: {failure.get('input_pattern')}' for failure in experience_data.get('recent_failures', [])])}\n'
        evolution_prompt = f"\n{firmware_content}\n\nCURRENT OPENAI AGENT INSTRUCTIONS:\n```\n{current_instructions}\n```\n\nREQUESTED CHANGES:\n{changes}\n\nTARGET DOMAIN: {target_domain}\n\n{experience_text}\n\nEVOLUTION STRATEGY:\n{strategy_details.get('description', 'Standard evolution')}\nInstruction Preservation Level: {instruction_preservation * 100:.0f}%\n\nINSTRUCTIONS:\n1. {prompt_modifier}\n2. Maintain the agent's role and primary purpose\n3. Follow all firmware guidelines for the target domain\n4. Focus on improving clarity, effectiveness, and addressing the requested changes\n5. Create a clear and well-structured set of instructions for the OpenAI agent\n\nEVOLVED INSTRUCTIONS:\n"
        return await self.llm.generate(evolution_prompt)

    def _generate_evolved_code(self, original_code: str, evolved_instructions: str) -> str:
        """Generate the evolved code by replacing the instructions in the original code."""
        import re
        instruction_pattern = '(instructions=)(?:"""|\\\'\\\'\\\')(.*?)(?:"""|\\\'\\\'\\\')'
        if re.search(instruction_pattern, original_code, re.DOTALL):
            return re.sub(instruction_pattern, f'\\1"""\n{evolved_instructions}\n"""', original_code, flags=re.DOTALL)
        alt_pattern = '(instructions=)[\\\'"](.*?)[\\\'"]'
        if re.search(alt_pattern, original_code, re.DOTALL):
            escaped_instructions = evolved_instructions.replace('"', '\\"').replace("'", "\\'")
            return re.sub(alt_pattern, f'\\1"{escaped_instructions}"', original_code, flags=re.DOTALL)
        return f"\n# WARNING: Could not update instructions automatically\n# Please manually replace the instructions with the following:\n'''\n{evolved_instructions}\n'''\n\n{original_code}\n"

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.utcnow().isoformat()

class OpenAIAgentsEvolutionManager:
    """
    Manager for OpenAI agent evolution processes.
    This class integrates the evolution tools with the core framework.
    """

    def __init__(self, smart_library: SmartLibrary, llm_service: LLMService, agent_factory: AgentFactory):
        self.library = smart_library
        self.llm = llm_service
        self.agent_factory = agent_factory
        self.agent_logger = OpenAIAgentLogger()
        self.evolve_tool = EvolveOpenAIAgentTool(smart_library=smart_library, llm_service=llm_service, agent_logger=self.agent_logger)
        self.ab_test_tool = ABTestOpenAIAgentTool(smart_library=smart_library, llm_service=llm_service, agent_factory=agent_factory, agent_logger=self.agent_logger)
        logger.info('OpenAI Agents Evolution Manager initialized')

    async def analyze_evolution_candidates(self, domain: Optional[str]=None) -> List[Dict[str, Any]]:
        """
        Analyze agents in the library to identify candidates for evolution.
        
        Args:
            domain: Optional domain to filter agents
            
        Returns:
            List of agents with evolution recommendations
        """
        if domain:
            domain_records = await self.library.find_records_by_domain(domain, 'AGENT')
        else:
            domain_records = []
            for record in self.library.records:
                if record['record_type'] == 'AGENT':
                    domain_records.append(record)
        openai_agents = [record for record in domain_records if record.get('metadata', {}).get('framework', '').lower() == 'openai-agents']
        candidates = []
        for agent in openai_agents:
            experience = self.agent_logger.get_experience(agent['id'])
            if not experience:
                continue
            evolution_recommendation = await self._analyze_agent_for_evolution(agent, experience)
            if evolution_recommendation:
                candidates.append({'agent_id': agent['id'], 'agent_name': agent['name'], 'domain': agent['domain'], 'version': agent['version'], 'evolution_recommendation': evolution_recommendation})
        return candidates

    async def evolve_agent(self, agent_id: str, changes: str, evolution_type: str='standard', target_domain: Optional[str]=None) -> Dict[str, Any]:
        """
        Evolve an OpenAI agent using the EvolveOpenAIAgentTool.
        
        Args:
            agent_id: ID of the agent to evolve
            changes: Description of changes to make
            evolution_type: Type of evolution to perform
            target_domain: Optional target domain if adapting
            
        Returns:
            Result of the evolution operation
        """
        evolution_input = self.evolve_tool.input_schema(agent_id_or_name=agent_id, evolution_type=evolution_type, changes=changes, target_domain=target_domain, learning_from_experience=True)
        result = await self.evolve_tool._run(evolution_input)
        try:
            return json.loads(result.get_text_content())
        except json.JSONDecodeError:
            return {'status': 'error', 'message': 'Failed to parse evolution result', 'raw_result': result.get_text_content()}

    async def compare_agents(self, agent_a_id: str, agent_b_id: str, test_inputs: List[str], domain: str) -> Dict[str, Any]:
        """
        Compare two OpenAI agents using the ABTestOpenAIAgentTool.
        
        Args:
            agent_a_id: ID of the first agent
            agent_b_id: ID of the second agent
            test_inputs: List of test inputs
            domain: Domain for the test
            
        Returns:
            Result of the A/B test
        """
        test_input = self.ab_test_tool.input_schema(agent_a_id=agent_a_id, agent_b_id=agent_b_id, test_inputs=test_inputs, domain=domain)
        result = await self.ab_test_tool._run(test_input)
        try:
            return json.loads(result.get_text_content())
        except json.JSONDecodeError:
            return {'status': 'error', 'message': 'Failed to parse A/B test result', 'raw_result': result.get_text_content()}

    async def _analyze_agent_for_evolution(self, agent: Dict[str, Any], experience: OpenAIAgentExperience) -> Optional[Dict[str, Any]]:
        """
        Analyze an agent's experience to determine if it would benefit from evolution.
        
        Args:
            agent: The agent record
            experience: The agent's experience data
            
        Returns:
            Evolution recommendation if applicable, None otherwise
        """
        if experience.total_invocations < 10:
            return None
        success_rate = experience.successful_invocations / max(1, experience.total_invocations)
        if success_rate < 0.8:
            return {'reason': 'Low success rate', 'suggested_strategy': 'aggressive', 'details': f'Success rate is {success_rate:.2f}, below the 0.8 threshold', 'changes': "Improve the agent's instructions to handle more cases correctly."}
        if experience.domain_performance:
            worst_domain = min(experience.domain_performance.items(), key=lambda x: x[1])
            if worst_domain[1] < 0.7:
                return {'reason': 'Poor domain performance', 'suggested_strategy': 'domain_adaptation', 'target_domain': worst_domain[0], 'details': f"Performance in domain '{worst_domain[0]}' is {worst_domain[1]:.2f}, below the 0.7 threshold", 'changes': f"Adapt the agent to handle tasks in the '{worst_domain[0]}' domain more effectively."}
        if experience.recent_failures and len(experience.recent_failures) > 3:
            failure_pattern = self._identify_failure_pattern(experience.recent_failures)
            if failure_pattern:
                return {'reason': 'Recurring failure pattern', 'suggested_strategy': 'standard', 'details': f'Recurring failure pattern: {failure_pattern}', 'changes': f'Improve handling of inputs related to: {failure_pattern}'}
        if experience.evolution_history and experience.total_invocations > 100 and (len(experience.evolution_history) == 0):
            return {'reason': 'High usage without evolution', 'suggested_strategy': 'standard', 'details': f'Agent has {experience.total_invocations} invocations but has never been evolved', 'changes': 'General improvements based on usage patterns and common inputs.'}
        return None

    def _identify_failure_pattern(self, failures: List[Dict[str, Any]]) -> Optional[str]:
        """Identify common patterns in recent failures."""
        domains = [f.get('domain') for f in failures if 'domain' in f]
        patterns = [f.get('input_pattern') for f in failures if 'input_pattern' in f]
        domain_counts = {}
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        pattern_counts = {}
        for pattern in patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        if domain_counts:
            most_common_domain = max(domain_counts.items(), key=lambda x: x[1])
            if most_common_domain[1] >= 2:
                return f'domain: {most_common_domain[0]}'
        if pattern_counts:
            most_common_pattern = max(pattern_counts.items(), key=lambda x: x[1])
            if most_common_pattern[1] >= 2:
                return f'input pattern: {most_common_pattern[0]}'
        return None

