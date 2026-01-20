# Cluster 2

class ProcessWorkflowTool(Tool[ProcessWorkflowInput, None, StringToolOutput]):
    """
    Parses and prepares YAML-based workflows. Validates structure, substitutes parameters.
    If Intent Mode is active (globally and for 'intents' level), it converts steps
    into an IntentPlan, saves it to MongoDB, and returns its ID for review.
    Otherwise, it returns a structured list of steps for direct execution by SystemAgent.
    """
    name = 'ProcessWorkflowTool'
    description = 'Parse, validate, and prepare YAML workflows. Optionally generates an IntentPlan and saves it to MongoDB for review, or returns processed steps for direct execution.'
    input_schema = ProcessWorkflowInput

    def __init__(self, mongodb_client: Optional[MongoDBClient]=None, container: Optional[DependencyContainer]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.container = container
        if mongodb_client:
            self.mongodb_client = mongodb_client
        elif self.container and self.container.has('mongodb_client'):
            self.mongodb_client = self.container.get('mongodb_client')
        else:
            try:
                self.mongodb_client = MongoDBClient()
                if self.container and (not self.container.has('mongodb_client')):
                    self.container.register('mongodb_client', self.mongodb_client)
                logger.info('ProcessWorkflowTool: Default MongoDBClient instance created.')
            except ValueError as e:
                logger.warning(f'ProcessWorkflowTool: MongoDBClient FAILED to initialize: {e}. Intent plan saving to DB will be skipped if intent mode is active.')
                self.mongodb_client = None
        if self.mongodb_client:
            self.intent_plans_collection_name = 'eat_intent_plans'
            self.intent_plans_collection = self.mongodb_client.get_collection(self.intent_plans_collection_name)
            asyncio.create_task(self._ensure_intent_plan_indexes())
        else:
            self.intent_plans_collection = None
            self.intent_plans_collection_name = 'N/A (MongoDB unavailable)'
        logger.info(f'ProcessWorkflowTool initialized. MongoDB for intent plans: {self.intent_plans_collection_name}')

    async def _ensure_intent_plan_indexes(self):
        """Ensure MongoDB indexes for the intent_plans collection."""
        if self.intent_plans_collection is not None:
            try:
                await self.intent_plans_collection.create_index([('plan_id', pymongo.ASCENDING)], unique=True, background=True)
                await self.intent_plans_collection.create_index([('status', pymongo.ASCENDING)], background=True)
                await self.intent_plans_collection.create_index([('created_at', pymongo.DESCENDING)], background=True)
                logger.info(f"Ensured indexes on '{self.intent_plans_collection_name}' collection.")
            except Exception as e:
                logger.error(f"Error creating indexes for '{self.intent_plans_collection_name}': {e}", exc_info=True)
        else:
            logger.warning(f"Cannot ensure indexes on '{self.intent_plans_collection_name}' as collection is None.")

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'workflow', 'process'], creator=self)

    async def _run(self, tool_input: ProcessWorkflowInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        logger.info('Processing workflow YAML...')
        try:
            cleaned_yaml_string = html.unescape(tool_input.workflow_yaml)
            cleaned_yaml_string = re.sub('```yaml\\s*', '', cleaned_yaml_string)
            cleaned_yaml_string = re.sub('\\s*```', '', cleaned_yaml_string)
            lines = cleaned_yaml_string.strip().split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if stripped_line and (stripped_line.split(':')[0] in ['scenario_name', 'domain', 'description', 'steps'] or stripped_line.startswith('- type:')):
                    start_idx = i
                    break
            cleaned_yaml_string = '\n'.join(lines[start_idx:])
            logger.debug('Cleaned YAML string (first 500 chars):\n%s', cleaned_yaml_string[:500])
            try:
                workflow = yaml.safe_load(cleaned_yaml_string)
                if not isinstance(workflow, dict):
                    if isinstance(workflow, list) and all((isinstance(s, dict) and 'type' in s for s in workflow)):
                        logger.warning('YAML loaded as a list of steps. Wrapping with default scenario info.')
                        workflow = {'scenario_name': f'Recovered_Workflow_{uuid.uuid4().hex[:4]}', 'domain': 'general', 'description': 'Workflow recovered from a list of steps.', 'steps': workflow}
                    else:
                        raise ValueError(f'Workflow YAML must load as a dictionary or a list of valid step dictionaries. Got type: {type(workflow)}')
                if 'steps' not in workflow or not isinstance(workflow.get('steps'), list):
                    raise ValueError("Invalid workflow: Missing 'steps' list or 'steps' is not a list.")
            except yaml.YAMLError as e:
                error_mark = getattr(e, 'problem_mark', None)
                line_num_str = str(error_mark.line + 1) if error_mark else 'unknown'
                problem = getattr(e, 'problem', str(e))
                context_msg = getattr(e, 'context', '')
                snippet_start = max(0, error_mark.index - 40) if error_mark else 0
                snippet_end = min(len(cleaned_yaml_string), error_mark.index + 40 if error_mark else 80)
                snippet = cleaned_yaml_string[snippet_start:snippet_end]
                raise ValueError(f'Error parsing YAML near line {line_num_str}: {problem}. {context_msg}. Snippet: ...{snippet}...') from e
            logger.info(f"Successfully parsed YAML for workflow '{workflow.get('scenario_name', 'N/A')}'")
            params_dict = tool_input.params if isinstance(tool_input.params, dict) else {}
            processed_workflow_steps = self._substitute_params_recursive(workflow['steps'], params_dict)
            logger.info('Parameter substitution complete.')
            intent_mode_active_for_intents = False
            if context and hasattr(context, 'get_value') and context.get_value('intent_review_mode_override', False):
                intent_mode_active_for_intents = True
                logger.debug('Intent mode FORCED by context override for this ProcessWorkflowTool run.')
            elif config.INTENT_REVIEW_ENABLED:
                intent_mode_active_for_intents = 'intents' in getattr(config, 'INTENT_REVIEW_LEVELS', [])
            logger.debug(f"Intent mode active for 'intents' level: {intent_mode_active_for_intents}")
            objective = tool_input.objective or params_dict.get('objective', params_dict.get('user_request', workflow.get('description', 'Objective not explicitly specified.')))
            logger.debug(f"Determined objective for IntentPlan: '{objective}'")
            validated_output = self._validate_steps(processed_workflow_steps, intent_mode=intent_mode_active_for_intents, objective=objective, workflow_name=workflow.get('scenario_name', 'Unnamed Workflow'), plan_id_override=tool_input.plan_id_override)
            log_msg_suffix = ''
            if intent_mode_active_for_intents and isinstance(validated_output, IntentPlan):
                log_msg_suffix = f" (IntentPlan '{validated_output.plan_id}' generated)"
            logger.info(f'Workflow steps processed.{log_msg_suffix}')
            if isinstance(validated_output, IntentPlan):
                intent_plan_obj = validated_output
                intent_plan_dict = intent_plan_obj.to_dict()
                if self.intent_plans_collection is not None:
                    try:
                        await self.intent_plans_collection.replace_one({'plan_id': intent_plan_obj.plan_id}, intent_plan_dict, upsert=True)
                        logger.info(f"IntentPlan '{intent_plan_obj.plan_id}' saved/updated in MongoDB.")
                        return_payload = {'status': 'intent_plan_created', 'message': 'Intent plan created and saved to MongoDB. Review is required before execution.', 'plan_id': intent_plan_obj.plan_id}
                    except Exception as db_err:
                        logger.error(f"Failed to save IntentPlan '{intent_plan_obj.plan_id}' to MongoDB: {db_err}", exc_info=True)
                        return_payload = {'status': 'intent_plan_created_db_error', 'message': f'Intent plan created but FAILED to save to MongoDB: {db_err}. Pass this full plan to review tool.', 'plan_id': intent_plan_obj.plan_id, 'intent_plan': intent_plan_dict}
                else:
                    logger.warning('MongoDB client/collection not available. IntentPlan was generated but not saved to database.')
                    return_payload = {'status': 'intent_plan_created_no_db', 'message': 'Intent plan created (MongoDB unavailable). Pass this full plan to review tool.', 'plan_id': intent_plan_obj.plan_id, 'intent_plan': intent_plan_dict}
                if context and hasattr(context, 'set_value'):
                    try:
                        context.set_value('intent_plan_json_output', safe_json_dumps(intent_plan_dict))
                        logger.debug(f"Full IntentPlan '{intent_plan_obj.plan_id}' also saved to run context.")
                    except Exception as ctx_err:
                        logger.warning(f'Failed to save full IntentPlan to context: {ctx_err}')
                return StringToolOutput(safe_json_dumps(return_payload))
            elif isinstance(validated_output, list):
                plan_for_direct_execution = {'status': 'success', 'scenario_name': workflow.get('scenario_name', 'Unnamed Workflow'), 'domain': workflow.get('domain', 'general'), 'description': workflow.get('description', 'Workflow for direct execution'), 'steps': validated_output, 'execution_guidance': 'The SystemAgent should now execute these steps sequentially.'}
                return StringToolOutput(safe_json_dumps(plan_for_direct_execution))
            else:
                raise TypeError(f'Unexpected output type from _validate_steps: {type(validated_output)}. Expected list or IntentPlan.')
        except Exception as e:
            import traceback
            logger.error(f'Error processing workflow in _run: {e}', exc_info=True)
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f'Error processing workflow: {str(e)}', 'details': traceback.format_exc()}))

    def _substitute_params_recursive(self, obj: Any, params: Dict[str, Any]) -> Any:
        placeholder_pattern = '{{\\s*params\\.([\\w_]+)\\s*}}'
        if isinstance(obj, dict):
            return {self._substitute_params_recursive(k, params) if isinstance(k, str) else k: self._substitute_params_recursive(v, params) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_params_recursive(item, params) for item in obj]
        elif isinstance(obj, str):
            exact_match = re.fullmatch(placeholder_pattern, obj.strip())
            if exact_match:
                key = exact_match.group(1)
                if key in params:
                    logger.debug(f"Substituting exact placeholder '{obj.strip()}' with param '{key}' (type: {type(params[key])}).")
                    return params[key]
                else:
                    logger.warning(f"Parameter '{key}' for exact match placeholder '{obj.strip()}' not found. Returning original placeholder string.")
                    return obj

            def replace_match_in_string(match_obj):
                key = match_obj.group(1)
                if key in params:
                    return str(params[key])
                else:
                    logger.warning(f"Parameter '{key}' for embedded placeholder '{match_obj.group(0)}' not found. Leaving placeholder.")
                    return match_obj.group(0)
            return re.sub(placeholder_pattern, replace_match_in_string, obj)
        else:
            return obj

    def _convert_steps_to_intent_plan_obj(self, workflow_steps: List[Dict[str, Any]], objective: str, workflow_name: str, plan_id_override: Optional[str]) -> IntentPlan:
        plan_id = plan_id_override or f'plan_{uuid.uuid4().hex[:10]}'
        logger.info(f"Converting {len(workflow_steps)} steps to IntentPlan object '{plan_id}' for workflow '{workflow_name}'")
        intents_list: List[Intent] = []
        step_index_to_intent_id: Dict[int, str] = {}
        current_time_iso = datetime.now(timezone.utc).isoformat()
        for i, step_dict in enumerate(workflow_steps):
            intent_id = f'intent_{plan_id}_{i + 1}_{uuid.uuid4().hex[:6]}'
            step_index_to_intent_id[i] = intent_id
            step_type_str = step_dict.get('type', 'UNKNOWN_STEP_TYPE')
            params_or_input_dict: Dict[str, Any] = {}
            if step_type_str == 'EXECUTE':
                params_or_input_dict = step_dict.get('input', {})
            elif step_type_str in ['DEFINE', 'CREATE']:
                core_keys = {'type', 'item_type', 'name', 'description', 'code_snippet', 'output_var', 'from_existing_snippet', 'config'}
                params_or_input_dict = {k: v for k, v in step_dict.items() if k not in core_keys}
                if 'config' in step_dict:
                    params_or_input_dict['config'] = step_dict['config']
            elif step_type_str == 'RETURN':
                params_or_input_dict = {'value': step_dict.get('value')}
            depends_on_ids: List[str] = []
            params_str_for_scan = safe_json_dumps(params_or_input_dict, indent=None)
            step_var_pattern = '{{\\s*([\\w_]+)\\s*}}'
            found_step_vars = set(re.findall(step_var_pattern, params_str_for_scan))
            for var_name in found_step_vars:
                for prev_step_idx, prev_step_dict in enumerate(workflow_steps[:i]):
                    if prev_step_dict.get('output_var') == var_name:
                        if prev_step_idx in step_index_to_intent_id:
                            depends_on_ids.append(step_index_to_intent_id[prev_step_idx])
                            logger.debug(f"Intent '{intent_id}' depends on Intent '{step_index_to_intent_id[prev_step_idx]}' via output_var '{var_name}'.")
                            break
                        else:
                            logger.warning(f"Could not find mapped Intent ID for previous step index {prev_step_idx} defining '{var_name}'.")
            intent_obj = Intent(intent_id=intent_id, step_type=step_type_str, component_type=step_dict.get('item_type', 'GENERIC_COMPONENT'), component_name=step_dict.get('name', 'UnnamedComponent'), action=step_dict.get('action', step_type_str.lower()), params=params_or_input_dict, justification=step_dict.get('description', f'Execute step {i + 1} of type {step_type_str}'), depends_on=sorted(list(set(depends_on_ids))), status=IntentStatus.PENDING)
            intents_list.append(intent_obj)
        return IntentPlan(plan_id=plan_id, title=f'Intent Plan: {workflow_name}', description=f'Generated for objective: {objective}', objective=objective, intents=intents_list, status=PlanStatus.PENDING_REVIEW, created_at=current_time_iso)

    def _validate_steps(self, steps: List[Dict[str, Any]], intent_mode: bool=False, objective: Optional[str]='Not specified.', workflow_name: Optional[str]='Unnamed Workflow', plan_id_override: Optional[str]=None) -> Union[List[Dict[str, Any]], IntentPlan]:
        if not isinstance(steps, list):
            raise ValueError("Workflow 'steps' must be a list.")
        validated_steps_list = []
        current_objective = objective or 'Objective not explicitly provided.'
        for i, step in enumerate(steps):
            step_num = i + 1
            if not isinstance(step, dict):
                raise ValueError(f'Step {step_num} is not a dictionary: {step}')
            step_type = step.get('type')
            if not step_type or not isinstance(step_type, str):
                raise ValueError(f"Step {step_num} missing or invalid 'type' field: {step}")
            valid_types = ['DEFINE', 'CREATE', 'EXECUTE', 'RETURN']
            if step_type not in valid_types:
                raise ValueError(f"Step {step_num} invalid type '{step_type}'. Must be one of {valid_types}.")
            required_keys = set()
            if step_type == 'DEFINE':
                required_keys = {'item_type', 'name', 'description'}
            elif step_type == 'CREATE':
                required_keys = {'item_type', 'name'}
            elif step_type == 'EXECUTE':
                required_keys = {'item_type', 'name'}
            elif step_type == 'RETURN':
                required_keys = {'value'}
            present_keys = set(step.keys())
            missing_keys = required_keys - present_keys
            if missing_keys:
                raise ValueError(f'Step {step_num} (type: {step_type}) missing required keys: {missing_keys}. Step: {step}')
            if step_type == 'EXECUTE' and 'input' in step and (not isinstance(step['input'], dict)):
                logger.error(f"Validation Error in Step {step_num}: 'input' field is not a dictionary. Found type: {type(step['input'])}. Step content: {step}")
                raise ValueError(f"Step {step_num} (EXECUTE): 'input' field MUST be a dictionary, but found type {type(step['input'])}.")
            if intent_mode and current_objective == 'Objective not explicitly provided.' and (step_type == 'EXECUTE') and isinstance(step.get('input'), dict):
                input_params = step['input']
                for key, value in input_params.items():
                    if isinstance(key, str) and key.lower() in ['objective', 'goal', 'user_request', 'task_description']:
                        if isinstance(value, str) and value:
                            current_objective = value
                            logger.debug(f"Objective updated from step {step_num} input: '{current_objective}'")
                            break
            validated_steps_list.append(step)
        if intent_mode:
            logger.info('Intent mode active, converting validated steps to IntentPlan object.')
            return self._convert_steps_to_intent_plan_obj(validated_steps_list, current_objective, workflow_name, plan_id_override)
        else:
            logger.info('Returning validated steps list for direct execution.')
            return validated_steps_list

class ApprovePlanTool(Tool[ApprovePlanInput, None, StringToolOutput]):
    """
    Tool for reviewing and approving intent plans.
    Loads plans from MongoDB using plan_id, or from context as a fallback.
    Updates the plan status in MongoDB after review.
    """
    name = 'ApprovePlanTool'
    description = 'Review and approve intent plans (from MongoDB or context) before execution in the SystemAgent'
    input_schema = ApprovePlanInput

    def __init__(self, llm_service: Optional[LLMService]=None, mongodb_client: Optional[MongoDBClient]=None, container: Optional[DependencyContainer]=None, options: Optional[Dict[str, Any]]=None):
        super().__init__(options=options or {})
        self.llm_service = llm_service
        self.container = container
        if mongodb_client:
            self.mongodb_client = mongodb_client
        elif self.container and self.container.has('mongodb_client'):
            self.mongodb_client = self.container.get('mongodb_client')
        else:
            try:
                self.mongodb_client = MongoDBClient()
                if self.container and (not self.container.has('mongodb_client')):
                    self.container.register('mongodb_client', self.mongodb_client)
                logger.info('ApprovePlanTool: Default MongoDBClient instance created.')
            except ValueError as e:
                logger.warning(f'ApprovePlanTool: MongoDBClient FAILED to initialize: {e}. Will rely on intent_plan_json_override or context for plan data.')
                self.mongodb_client = None
        if self.mongodb_client:
            self.intent_plans_collection_name = 'eat_intent_plans'
            self.intent_plans_collection = self.mongodb_client.get_collection(self.intent_plans_collection_name)
        else:
            self.intent_plans_collection = None
            self.intent_plans_collection_name = 'N/A (MongoDB unavailable)'
        logger.info(f'ApprovePlanTool initialized. MongoDB for intent plans: {self.intent_plans_collection_name}')

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=['tool', 'intent_review', 'approve_plan'], creator=self)

    async def _run(self, tool_input: ApprovePlanInput, options: Optional[Dict[str, Any]]=None, context: Optional[RunContext]=None) -> StringToolOutput:
        intent_plan_dict: Optional[Dict[str, Any]] = None
        if tool_input.intent_plan_json_override:
            try:
                intent_plan_dict = json.loads(tool_input.intent_plan_json_override)
                if intent_plan_dict.get('plan_id') != tool_input.plan_id:
                    logger.warning(f'Plan ID mismatch: Override plan ID {intent_plan_dict.get('plan_id')} vs requested {tool_input.plan_id}. Ignoring override.')
                    intent_plan_dict = None
                else:
                    logger.info(f"Loaded IntentPlan '{tool_input.plan_id}' from intent_plan_json_override input.")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse intent_plan_json_override for plan '{tool_input.plan_id}'.")
                intent_plan_dict = None
        if not intent_plan_dict and self.intent_plans_collection:
            try:
                db_doc = await self.intent_plans_collection.find_one({'plan_id': tool_input.plan_id})
                if db_doc:
                    db_doc.pop('_id', None)
                    intent_plan_dict = db_doc
                    logger.info(f"Loaded IntentPlan '{tool_input.plan_id}' from MongoDB.")
            except Exception as e:
                logger.error(f"Error loading IntentPlan '{tool_input.plan_id}' from MongoDB: {e}", exc_info=True)
        if not intent_plan_dict and context and hasattr(context, 'context') and isinstance(context.context, dict):
            intent_plan_json_from_context = context.context.get('intent_plan_json_output')
            if intent_plan_json_from_context:
                try:
                    parsed_from_context = json.loads(intent_plan_json_from_context)
                    if parsed_from_context.get('plan_id') == tool_input.plan_id:
                        intent_plan_dict = parsed_from_context
                        logger.info(f"Loaded IntentPlan '{tool_input.plan_id}' from RunContext.")
                    else:
                        logger.warning(f'Plan ID mismatch: Context plan ID {parsed_from_context.get('plan_id')} vs requested {tool_input.plan_id}.')
                except json.JSONDecodeError:
                    logger.error('Failed to parse intent plan from context.')
        if not intent_plan_dict:
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f"IntentPlan with ID '{tool_input.plan_id}' not found in any source (override, MongoDB, context)."}))
        try:
            intent_plan_obj = IntentPlan.from_dict(intent_plan_dict)
        except Exception as e:
            logger.error(f"Failed to instantiate IntentPlan object from loaded dictionary for plan '{tool_input.plan_id}': {e}", exc_info=True)
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f"Invalid IntentPlan data structure for plan '{tool_input.plan_id}': {e}"}))
        review_decision: Dict[str, Any] = {}
        modified_plan_for_saving = intent_plan_obj
        if tool_input.use_agent_reviewer and self.llm_service:
            agent_review_output_str = await self._agent_review_plan(modified_plan_for_saving, tool_input.agent_prompt)
            review_decision = json.loads(agent_review_output_str.get_text_content())
        elif tool_input.interactive_mode:
            interactive_output_str = await self._interactive_review_plan(modified_plan_for_saving)
            review_decision = json.loads(interactive_output_str.get_text_content())
            if review_decision.get('status') == PlanStatus.APPROVED.value and 'approved_plan' in review_decision:
                modified_plan_for_saving = IntentPlan.from_dict(review_decision['approved_plan'])
        else:
            cli_output_str = await self._cli_review_plan(modified_plan_for_saving)
            review_decision = json.loads(cli_output_str.get_text_content())
            if review_decision.get('status') == PlanStatus.APPROVED.value and 'approved_plan' in review_decision:
                modified_plan_for_saving = IntentPlan.from_dict(review_decision['approved_plan'])
        final_status_str = review_decision.get('status')
        if final_status_str == PlanStatus.APPROVED.value:
            modified_plan_for_saving.status = PlanStatus.APPROVED
            modified_plan_for_saving.reviewer_comments = review_decision.get('comments', modified_plan_for_saving.reviewer_comments)
            modified_plan_for_saving.review_timestamp = datetime.now(timezone.utc).isoformat()
            for intent_item in modified_plan_for_saving.intents:
                intent_item.status = IntentStatus.APPROVED
                intent_item.review_comments = review_decision.get('comments', '')
        elif final_status_str == PlanStatus.REJECTED.value:
            modified_plan_for_saving.status = PlanStatus.REJECTED
            modified_plan_for_saving.rejection_reason = review_decision.get('reason', 'No reason provided.')
            modified_plan_for_saving.review_timestamp = datetime.now(timezone.utc).isoformat()
            for intent_item in modified_plan_for_saving.intents:
                intent_item.status = IntentStatus.REJECTED
                intent_item.review_comments = review_decision.get('reason', 'Plan rejected')
        if modified_plan_for_saving.status in [PlanStatus.APPROVED, PlanStatus.REJECTED]:
            if self.intent_plans_collection:
                try:
                    updated_plan_dict_for_db = modified_plan_for_saving.to_dict()
                    await self.intent_plans_collection.replace_one({'plan_id': tool_input.plan_id}, updated_plan_dict_for_db, upsert=False)
                    logger.info(f"Updated IntentPlan '{tool_input.plan_id}' status to '{modified_plan_for_saving.status.value}' in MongoDB.")
                except Exception as db_err:
                    logger.error(f"Failed to update IntentPlan '{tool_input.plan_id}' in MongoDB: {db_err}", exc_info=True)
                    review_decision['db_update_status'] = f'failed: {db_err}'
            else:
                logger.warning(f"MongoDB client not available. IntentPlan '{tool_input.plan_id}' status updated in memory but not saved to DB.")
                review_decision['db_update_status'] = 'skipped_mongodb_unavailable'
        if tool_input.output_path:
            output_dir = os.path.dirname(tool_input.output_path)
            if output_dir and (not os.path.exists(output_dir)):
                os.makedirs(output_dir, exist_ok=True)
            try:
                with open(tool_input.output_path, 'w') as f:
                    f.write(safe_json_dumps(modified_plan_for_saving.to_dict()))
                logger.info(f'Reviewed intent plan copy saved to {tool_input.output_path}')
            except Exception as e:
                logger.error(f'Failed to save reviewed intent plan copy to {tool_input.output_path}: {e}')
        return StringToolOutput(safe_json_dumps(review_decision))

    async def _agent_review_plan(self, intent_plan_obj: IntentPlan, custom_prompt: Optional[str]=None) -> StringToolOutput:
        if not self.llm_service:
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': 'LLM Service not available for AI review'}))
        plan_dict_for_prompt = intent_plan_obj.to_dict()
        default_prompt = f'''\n        As an AI Safety and Efficacy Inspector, review the following IntentPlan.\n        Your goal is to ensure the plan is safe, aligned with its objective, and likely to succeed.\n\n        IntentPlan Details:\n        ```json\n        {safe_json_dumps(plan_dict_for_prompt)}\n        ```\n\n        Review Criteria:\n        1.  **Safety & Ethics:** Does any intent pose a risk of harm, data breach, or unethical action?\n        2.  **Objective Alignment:** Is each intent and the overall plan clearly aligned with the stated objective: "{intent_plan_obj.objective}"?\n        3.  **Parameter Validity:** Are the parameters for each 'EXECUTE' intent complete, correct, and make sense in context?\n        4.  **Dependency Correctness:** Are `depends_on` fields logical and correctly specified?\n        5.  **Likelihood of Success:** Based on the actions and components, how likely is this plan to achieve the objective?\n        6.  **Efficiency:** Could the objective be achieved more simply or with fewer steps? (Optional, if obvious)\n\n        Decision:\n        Based on your review, decide whether to:\n        -   `APPROVE` the plan if it's sound.\n        -   `REJECT` the plan if significant issues are found. Provide clear reasons and suggest critical modifications.\n\n        Output Format (Strict JSON):\n        Return your response as a single JSON object with the following fields:\n        -   `"status"`: Either "{PlanStatus.APPROVED.value}" or "{PlanStatus.REJECTED.value}".\n        -   `"reason"`: A concise overall reason for your decision.\n        -   `"overall_risk_assessment"`: (String) "Low", "Medium", or "High".\n        -   `"intent_reviews"`: (List of objects) For each intent in the original plan:\n            -   `"intent_id"`: (String) The ID of the intent being reviewed.\n            -   `"intent_status_assessment"`: (String) "{IntentStatus.APPROVED.value}" (if safe and good) or "{IntentStatus.REJECTED.value}" (if problematic).\n            -   `"concerns"`: (List of strings) Specific concerns or reasons for rejection for this intent. Empty if approved.\n            -   `"suggestions_for_intent"`: (String, Optional) Specific suggestions for improving this intent if it's problematic.\n        -   `"suggestions_for_plan"`: (String, Optional) Overall suggestions for improving the plan if rejected or if minor improvements are noted for an approved plan.\n        '''
        prompt_to_use = custom_prompt or default_prompt
        review_text = await self.llm_service.generate(prompt_to_use)
        try:
            match = re.search('\\{[\\s\\S]*\\}', review_text)
            if match:
                json_str = match.group(0)
                review_json = json.loads(json_str)
            else:
                review_json = json.loads(review_text)
            review_json['review_source'] = 'ai_agent'
            return StringToolOutput(safe_json_dumps(review_json))
        except json.JSONDecodeError as e:
            logger.error(f'AI reviewer response was not valid JSON: {e}. Raw response: {review_text}')
            return StringToolOutput(safe_json_dumps({'status': 'error', 'message': f'Failed to parse AI reviewer response: {e}', 'raw_response': review_text}))

    async def _interactive_review_plan(self, intent_plan_obj: IntentPlan) -> StringToolOutput:
        print('\n' + '=' * 60)
        print('🔍 INTENT PLAN REVIEW 🔍')
        print('=' * 60)
        print(f'\nObjective: {intent_plan_obj.objective}')
        print(f'Plan ID: {intent_plan_obj.plan_id}')
        print(f'\nThe plan contains {len(intent_plan_obj.intents)} intents:')
        for i, intent_obj in enumerate(intent_plan_obj.intents, 1):
            print(f'\n{i}. [{intent_obj.step_type}] {intent_obj.component_name}.{intent_obj.action}')
            if intent_obj.params:
                print(f'   Parameters: {safe_json_dumps(intent_obj.params, indent=6)}')
            print(f'   Justification: {intent_obj.justification}')
            if intent_obj.depends_on:
                print(f'   Depends on: {', '.join(intent_obj.depends_on)}')
        print('\nReview this plan. Safe to execute?')
        while True:
            choice = input('\nApprove? (y/n/d for details/i for intent details): ').strip().lower()
            if choice == 'd':
                print(safe_json_dumps(intent_plan_obj.to_dict()))
                continue
            if choice == 'i':
                try:
                    num_str = input('Enter intent number to inspect: ')
                    num = int(num_str)
                    if 1 <= num <= len(intent_plan_obj.intents):
                        print(safe_json_dumps(intent_plan_obj.intents[num - 1].to_dict()))
                    else:
                        print(f'Invalid number (1-{len(intent_plan_obj.intents)}).')
                except ValueError:
                    print('Invalid input.')
                continue
            if choice == 'y':
                comments = input('Optional comments: ').strip()
                approved_plan_dict = intent_plan_obj.to_dict()
                approved_plan_dict['status'] = PlanStatus.APPROVED.value
                for intent_d in approved_plan_dict.get('intents', []):
                    intent_d['status'] = IntentStatus.APPROVED.value
                return StringToolOutput(safe_json_dumps({'status': PlanStatus.APPROVED.value, 'message': 'Plan approved by human reviewer.', 'comments': comments, 'approved_plan': approved_plan_dict}))
            if choice == 'n':
                reason = input('Reason for rejection: ').strip()
                return StringToolOutput(safe_json_dumps({'status': PlanStatus.REJECTED.value, 'message': 'Plan rejected by human reviewer.', 'reason': reason}))
            print('Invalid choice. y/n/d/i.')

    async def _cli_review_plan(self, intent_plan_obj: IntentPlan) -> StringToolOutput:
        print('\n' + '=' * 60)
        print('INTENT PLAN REVIEW (CLI MODE)')
        print('=' * 60)
        print(f'Review Plan ID: {intent_plan_obj.plan_id}. Objective: {intent_plan_obj.objective}')
        print("Enter 'approve' or 'reject [reason]'")
        timeout_seconds = 600
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = input('Response: ').strip()
                if response.lower().startswith('approve'):
                    approved_plan_dict = intent_plan_obj.to_dict()
                    approved_plan_dict['status'] = PlanStatus.APPROVED.value
                    for intent_d in approved_plan_dict.get('intents', []):
                        intent_d['status'] = IntentStatus.APPROVED.value
                    return StringToolOutput(safe_json_dumps({'status': PlanStatus.APPROVED.value, 'approved_plan': approved_plan_dict}))
                elif response.lower().startswith('reject'):
                    reason = response[len('reject'):].strip() or 'No reason provided (CLI)'
                    return StringToolOutput(safe_json_dumps({'status': PlanStatus.REJECTED.value, 'reason': reason}))
                else:
                    print("Invalid. 'approve' or 'reject [reason]'.")
            except KeyboardInterrupt:
                return StringToolOutput(safe_json_dumps({'status': 'cancelled'}))
            await asyncio.sleep(0.1)
        return StringToolOutput(safe_json_dumps({'status': 'timeout'}))

class LLMService:
    """
    LLM service that interfaces with chat and embedding models.
    Now uses MongoDB-backed LLMCache if caching is enabled.
    """

    def __init__(self, provider: str='openai', api_key: Optional[str]=None, model: Optional[str]=None, embedding_model: Optional[str]=None, use_cache: bool=True, mongodb_client: Optional[MongoDBClient]=None, container: Optional[Any]=None):
        self.provider = provider
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.container = container
        self.chat_model = None
        self.embedding_model = None
        self.cache = None
        self.mongodb_client = None
        if mongodb_client:
            self.mongodb_client = mongodb_client
        elif self.container and hasattr(self.container, 'has') and self.container.has('mongodb_client'):
            self.mongodb_client = self.container.get('mongodb_client')
        else:
            try:
                self.mongodb_client = MongoDBClient()
                if self.container and hasattr(self.container, 'register'):
                    self.container.register('mongodb_client', self.mongodb_client)
            except ValueError as e:
                logger.warning(f'MongoDBClient could not be initialized for LLMService cache: {e}. Cache will be disabled.')
                self.mongodb_client = None
                use_cache = False
        self.use_cache = use_cache and self.mongodb_client is not None
        if self.use_cache and self.mongodb_client:
            self.cache = LLMCache(mongodb_client=self.mongodb_client)
        else:
            self.cache = None
            if use_cache and (not self.mongodb_client):
                logger.warning('LLM Cache was requested but is disabled due to missing MongoDB client.')
        if self.api_key and provider == 'openai':
            os.environ['OPENAI_API_KEY'] = self.api_key
        try:
            if provider == 'openai':
                self.chat_model = OpenAIChatModel(model_id=model, settings={'api_key': self.api_key} if self.api_key else None)
                self.embedding_model = OpenAIEmbeddingModel(model_id=embedding_model, settings={'api_key': self.api_key} if self.api_key else None)
            elif provider == 'ollama':
                self.chat_model = LiteLLMChatModel(model_id=model or 'llama3', provider_id='ollama', settings={})
                self.embedding_model = OllamaEmbeddingModel(model_id=embedding_model, settings={})
            else:
                logger.warning(f"Unknown LLM provider '{provider}'. Defaulting to OpenAI.")
                self.provider = 'openai'
                self.chat_model = OpenAIChatModel(model_id=model, settings={'api_key': self.api_key} if self.api_key else None)
                self.embedding_model = OpenAIEmbeddingModel(model_id=embedding_model, settings={'api_key': self.api_key} if self.api_key else None)
        except Exception as e:
            logger.error(f'Error initializing LLM models: {e}', exc_info=True)
        cache_status = 'enabled (MongoDB)' if self.cache else 'disabled'
        model_id = self.chat_model.model_id if self.chat_model else 'N/A'
        embedding_id = self.embedding_model.model_id if self.embedding_model else 'N/A'
        logger.info(f'Initialized LLM service with provider: {self.provider}, chat model: {model_id}, embedding model: {embedding_id}, cache: {cache_status}')

    async def generate(self, prompt: str) -> str:
        """Generate text based on a prompt."""
        if not self.chat_model:
            logger.error('Chat model not initialized. Cannot generate text.')
            return 'Error: Chat model not available.'
        logger.debug(f'Generating response for prompt: {prompt[:50]}...')
        messages = [UserMessage(prompt)]
        if self.cache:
            try:
                cached_response = await self.cache.get_completion(messages, self.chat_model.model_id)
                if cached_response is not None:
                    return cached_response
            except Exception as e:
                logger.warning(f'Error retrieving from cache: {e}. Proceeding without cache.')
        try:
            response = await self.chat_model.create(messages=messages)
            response_text = response.get_text_content()
            if self.cache:
                try:
                    await self.cache.save_completion(messages, self.chat_model.model_id, response_text)
                except Exception as e:
                    logger.warning(f'Error saving to cache: {e}')
            return response_text
        except Exception as e:
            logger.error(f'Error generating response: {e}', exc_info=True)
            return f'Error generating response: {str(e)}'

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        if not self.embedding_model:
            logger.error('Embedding model not initialized. Cannot generate embeddings.')
            return [0.0] * 1536
        logger.debug(f'Generating embedding for text: {text[:50]}...')
        if self.cache:
            try:
                cached_embedding = await self.cache.get_embedding(text, self.embedding_model.model_id)
                if cached_embedding is not None:
                    return cached_embedding
            except Exception as e:
                logger.warning(f'Error retrieving embedding from cache: {e}. Proceeding without cache.')
        try:
            response = await self.embedding_model.create(text)
            embedding = response.get('data', [])
            if not embedding and 'error' not in response:
                logger.warning(f'Embedding model returned no data for text: {text[:50]}')
            if embedding and all((isinstance(x, (int, float)) for x in embedding)):
                embedding = [float(x) for x in embedding]
                if self.cache:
                    try:
                        await self.cache.save_embedding(text, self.embedding_model.model_id, embedding)
                    except Exception as e:
                        logger.warning(f'Error saving embedding to cache: {e}')
                return embedding
            else:
                logger.warning(f'Invalid embedding format returned: {type(embedding)}')
                dim = 1536 if self.provider == 'openai' else 768
                return [0.0] * dim
        except Exception as e:
            logger.error(f'Error generating embedding: {e}', exc_info=True)
            dim = 1536 if self.provider == 'openai' else 768
            return [0.0] * dim

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self.embedding_model:
            logger.error('Embedding model not initialized. Cannot generate batch embeddings.')
            dim = 1536 if self.provider == 'openai' else 768
            return [[0.0] * dim for _ in texts]
        logger.debug(f'Generating batch embeddings for {len(texts)} texts...')
        if self.cache:
            try:
                cached_embeddings = await self.cache.get_batch_embeddings(texts, self.embedding_model.model_id)
                if cached_embeddings is not None:
                    return cached_embeddings
            except Exception as e:
                logger.warning(f'Error retrieving batch embeddings from cache: {e}. Proceeding without cache.')
        try:
            responses = await self.embedding_model.create_batch(texts)
            embeddings = []
            for i, response in enumerate(responses):
                current_embedding = response.get('data', [])
                if not current_embedding and 'error' not in response:
                    logger.warning(f'Embedding model returned no data for batch text item {i}: {texts[i][:50]}')
                    dim = 1536 if self.provider == 'openai' else 768
                    current_embedding = [0.0] * dim
                if current_embedding and all((isinstance(x, (int, float)) for x in current_embedding)):
                    current_embedding = [float(x) for x in current_embedding]
                    if self.cache:
                        try:
                            await self.cache.save_embedding(texts[i], self.embedding_model.model_id, current_embedding)
                        except Exception as e:
                            logger.warning(f'Error saving batch embedding to cache: {e}')
                else:
                    logger.warning(f'Invalid embedding format for batch item {i}')
                    dim = 1536 if self.provider == 'openai' else 768
                    current_embedding = [0.0] * dim
                embeddings.append(current_embedding)
            return embeddings
        except Exception as e:
            logger.error(f'Error generating batch embeddings: {e}', exc_info=True)
            dim = 1536 if self.provider == 'openai' else 768
            return [[0.0] * dim for _ in range(len(texts))]

    async def clear_cache(self, older_than_seconds: Optional[int]=None) -> int:
        """Clear the LLM cache. Delegates to LLMCache instance."""
        if not self.cache:
            logger.info('Cache is disabled, nothing to clear.')
            return 0
        try:
            return await self.cache.clear_cache(older_than_seconds)
        except Exception as e:
            logger.error(f'Error clearing cache: {e}', exc_info=True)
            return 0

    async def generate_applicability_text(self, text_chunk: str, component_type: str='', component_name: str='') -> str:
        """Generate applicability text (T_raz)."""
        prompt = f"\n        Analyze the following text chunk from a component ({component_type or 'unknown'}: {component_name or 'unknown'}):\n        '''\n        {text_chunk}\n        '''\n        Generate a concise description (T_raz) focusing ONLY on its potential applicability and relevance for different tasks. Describe:\n        1. **Relevant Tasks:** What specific developer/agent tasks might this chunk be useful for (e.g., 'code generation', 'API documentation', 'testing', 'requirements analysis', 'debugging', 'cost estimation', 'security review')?\n        2. **Key Concepts/Implications:** What are the non-obvious technical or functional implications derived from this text? (e.g., 'dependency on X', 'requires async handling', 'critical for user authentication flow').\n        3. **Target Audience/Context:** Who would find this most useful or in what situation? (e.g., 'backend developer implementing feature Y', 'project manager estimating effort', 'security auditor reviewing access control').\n\n        Be concise and focus on applicability *beyond* just restating the content. Output ONLY the generated description (T_raz).\n        "
        try:
            applicability_text = await self.generate(prompt)
            return applicability_text.strip()
        except Exception as e:
            logger.error(f'Error generating applicability text: {e}', exc_info=True)
            return f'Error generating applicability: {str(e)}'

