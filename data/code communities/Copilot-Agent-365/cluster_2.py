# Cluster 2

class ScriptedDemoAgent(BasicAgent):
    """
    Executes scripted demonstrations from JSON files with support for:
    - Canned responses
    - Rich content blocks (charts, tables, code, etc.)
    - Real-time agent orchestration with static/dynamic parameters
    - Automatic agent loading from GitHub repository
    - Rich data display with display_result field
    - Proper agent name tracking and display
    """
    GITHUB_REPO = 'kody-w/AI-Agent-Templates'
    GITHUB_BRANCH = 'main'
    GITHUB_RAW_BASE = f'https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}'
    GITHUB_API_BASE = f'https://api.github.com/repos/{GITHUB_REPO}'

    def __init__(self):
        self.name = 'ScriptedDemo'
        self.metadata = {'name': self.name, 'description': 'Executes scripted demonstrations from JSON files stored in Azure File Storage. This agent reads pre-written demo scenarios and returns the appropriate canned responses based on user input matching. Perfect for consistent, repeatable product demonstrations.', 'parameters': {'type': 'object', 'properties': {'demo_name': {'type': 'string', 'description': "The name of the demo JSON file to load from Azure File Storage (without .json extension). Example: 'Bot_342_Morning_Greeting_Demo'"}, 'user_input': {'type': 'string', 'description': "The user's message to match against the conversation flow and return the appropriate canned response"}, 'action': {'type': 'string', 'description': "The action to perform. Options: 'list_demos' (list available demo files), 'load_demo' (load a demo and show its structure), 'respond' (match user input and return canned response)", 'enum': ['list_demos', 'load_demo', 'respond']}, 'user_guid': {'type': 'string', 'description': 'Optional user GUID for context (used in demo responses that reference user data)'}}, 'required': ['action']}}
        self.storage_manager = AzureFileStorageManager()
        self.demo_directory = 'demos'
        self.loaded_demo_cache = {}
        if AGENT_MANAGER_AVAILABLE:
            self.agent_manager = AgentManager()
        else:
            self.agent_manager = None
        self.remote_agent_cache = {}
        self._agent_manifest_cache = None
        super().__init__(name=self.name, metadata=self.metadata)

    def perform(self, **kwargs):
        """
        Main entry point for the agent. Routes to appropriate handler based on action.
        """
        action = kwargs.get('action', 'list_demos')
        demo_name = kwargs.get('demo_name', '')
        user_input = kwargs.get('user_input', '')
        user_guid = kwargs.get('user_guid', 'c0p110t0-aaaa-bbbb-cccc-123456789abc')
        try:
            if action == 'list_demos':
                return self.list_available_demos()
            elif action == 'load_demo':
                if not demo_name:
                    return self.format_error_response('demo_name is required for load_demo action')
                return self.load_demo(demo_name)
            elif action == 'respond':
                if not demo_name or not user_input:
                    return self.format_error_response('demo_name and user_input are required for respond action')
                return self.get_response_for_user_input(demo_name, user_input, user_guid)
            else:
                return self.format_error_response(f'Unknown action: {action}')
        except Exception as e:
            logging.error(f'Error in ScriptedDemoAgent: {str(e)}')
            return self.format_error_response(f'Agent error: {str(e)}')

    def list_available_demos(self):
        """
        List all available demo JSON files in the Azure File Storage demos directory.
        """
        try:
            self.storage_manager.ensure_directory_exists(self.demo_directory)
            files = self.storage_manager.list_files(self.demo_directory)
            demo_files = []
            for file_info in files:
                if hasattr(file_info, 'name') and file_info.name.endswith('.json'):
                    demo_name = file_info.name.replace('.json', '')
                    demo_files.append(demo_name)
            if not demo_files:
                response = {'status': 'success', 'message': 'No demo files found in Azure File Storage', 'available_demos': [], 'instructions': "Upload demo JSON files to the 'demos' directory in Azure File Storage", 'demo_directory': self.demo_directory}
            else:
                response = {'status': 'success', 'message': f'Found {len(demo_files)} demo file(s)', 'available_demos': demo_files, 'demo_directory': self.demo_directory, 'next_steps': "Use 'load_demo' action to view demo structure, or 'respond' action to get canned responses"}
            return json.dumps(response, indent=2)
        except Exception as e:
            logging.error(f'Error listing demos: {str(e)}')
            return self.format_error_response(f'Failed to list demos: {str(e)}')

    def load_demo(self, demo_name):
        """
        Load a demo JSON file from Azure File Storage and return its structure.
        """
        try:
            demo_data = self._read_demo_file(demo_name)
            if not demo_data:
                return self.format_error_response(f"Demo file '{demo_name}.json' not found or empty")
            conversation_flow = demo_data.get('conversation_flow', [])
            flow_summary = []
            for step in conversation_flow:
                step_info = {'step_number': step.get('step_number', 0), 'description': step.get('description', ''), 'user_message': step.get('user_message', ''), 'has_response': 'agent_response' in step}
                flow_summary.append(step_info)
            response = {'status': 'success', 'demo_name': demo_data.get('demo_name', demo_name), 'description': demo_data.get('description', ''), 'trigger_phrases': demo_data.get('trigger_phrases', []), 'total_steps': len(conversation_flow), 'conversation_flow': flow_summary, 'instructions': "Use 'respond' action with user_input matching a step's user_message to get the canned agent_response"}
            return json.dumps(response, indent=2)
        except json.JSONDecodeError as e:
            logging.error(f'Invalid JSON in demo file: {str(e)}')
            return self.format_error_response(f'Invalid JSON in demo file: {str(e)}')
        except Exception as e:
            logging.error(f'Error loading demo: {str(e)}')
            return self.format_error_response(f'Failed to load demo: {str(e)}')

    def get_response_for_user_input(self, demo_name, user_input, user_guid):
        """
        Match user input against conversation flow and return the appropriate canned response.
        Uses fuzzy matching to find the best matching step.
        """
        try:
            demo_data = self._read_demo_file(demo_name)
            if not demo_data:
                return self.format_error_response(f"Demo file '{demo_name}.json' not found")
            conversation_flow = demo_data.get('conversation_flow', [])
            if not conversation_flow:
                return self.format_error_response('No conversation flow found in demo script')
            user_input_lower = user_input.lower().strip()
            for step in conversation_flow:
                step_message = step.get('user_message', '').lower().strip()
                if step_message == user_input_lower:
                    return self._format_agent_response(step, demo_data, user_guid)
            best_match = None
            best_match_score = 0
            for step in conversation_flow:
                step_message = step.get('user_message', '').lower().strip()
                score = 0
                user_words = set(user_input_lower.split())
                step_words = set(step_message.split())
                matching_words = user_words.intersection(step_words)
                score = len(matching_words)
                trigger_phrases = demo_data.get('trigger_phrases', [])
                for trigger in trigger_phrases:
                    if trigger.lower() in user_input_lower:
                        score += 10
                if score > best_match_score:
                    best_match_score = score
                    best_match = step
            if best_match and best_match_score >= 2:
                return self._format_agent_response(best_match, demo_data, user_guid)
            available_steps = [s.get('user_message', '') for s in conversation_flow]
            return self.format_error_response(f"No matching step found for input: '{user_input}'. Available user messages: {available_steps}")
        except json.JSONDecodeError as e:
            logging.error(f'Invalid JSON in demo file: {str(e)}')
            return self.format_error_response(f'Invalid JSON in demo file: {str(e)}')
        except Exception as e:
            logging.error(f'Error getting response: {str(e)}')
            return self.format_error_response(f'Failed to get response: {str(e)}')

    def _read_demo_file(self, demo_name):
        """
        Read and parse a demo file from Azure File Storage with caching.
        """
        if demo_name in self.loaded_demo_cache:
            return self.loaded_demo_cache[demo_name]
        file_name = f'{demo_name}.json'
        demo_content = self.storage_manager.read_file(self.demo_directory, file_name)
        if not demo_content:
            return None
        demo_data = json.loads(demo_content)
        self.loaded_demo_cache[demo_name] = demo_data
        return demo_data

    def _format_agent_response(self, step, demo_data, user_guid):
        """
        Format the agent response from a matched step.
        Supports:
        - Legacy string responses with template replacement
        - Enhanced array responses with content blocks
        - Agent call execution with static and dynamic parameters
        - Rich data display with display_result field
        """
        agent_response = step.get('agent_response', '')
        if not agent_response:
            return self.format_error_response('No agent_response found for this step')
        user_input = step.get('user_message', '')
        if isinstance(agent_response, str):
            return self._apply_template_variables(agent_response, demo_data, user_guid)
        if isinstance(agent_response, list):
            result_parts = []
            for content_block in agent_response:
                processed = self._process_agent_response_content(content_block, demo_data, user_guid, user_input)
                if processed:
                    result_parts.append(processed)
            return '\n\n'.join(result_parts)
        return str(agent_response)

    def _apply_template_variables(self, text, demo_data, user_guid):
        """Apply template variable replacement to text."""
        formatted_text = text
        formatted_text = formatted_text.replace('{user_guid}', user_guid)
        formatted_text = formatted_text.replace('{demo_name}', demo_data.get('demo_name', ''))
        formatted_text = formatted_text.replace('{demo_description}', demo_data.get('description', ''))
        return formatted_text

    def _process_agent_response_content(self, content_block, demo_data, user_guid, user_input):
        """
        Process a single content block from enhanced agent_response.
        Handles regular content blocks and agent_call type blocks with proper agent name extraction.
        
        **KEY FIX**: Now properly extracts agent name from the 'agent' field and displays it correctly.
        """
        if not isinstance(content_block, dict):
            return str(content_block)
        content_type = content_block.get('type', 'text')
        if content_type == 'agent_call':
            return self._process_agent_call_block(content_block, user_guid, user_input, demo_data)
        if content_type == 'text':
            text_content = content_block.get('content', '')
            return self._apply_template_variables(text_content, demo_data, user_guid)
        return json.dumps(content_block, indent=2)

    def _process_agent_call_block(self, agent_call_config, user_guid, user_input, demo_data):
        """
        Process an agent_call content block with proper agent name extraction and rich data support.
        
        **KEY FIX**: This method now:
        1. Extracts the correct agent name from the 'agent' field
        2. Checks for 'display_result' first (for demos with pre-rendered data)
        3. Falls back to actual agent execution if no display_result
        4. Shows the correct agent name in the response badge
        
        Args:
            agent_call_config: The agent_call content block from JSON
            user_guid: User GUID for context
            user_input: User's message for dynamic parameter extraction
            demo_data: Full demo data for additional context
            
        Returns:
            Formatted response with agent name badge
        """
        agent_name = agent_call_config.get('agent', 'UnknownAgent')
        description = agent_call_config.get('description', f'Calling {agent_name}')
        logging.info(f'Processing agent call: {agent_name} - {description}')
        if 'display_result' in agent_call_config:
            display_result = agent_call_config['display_result']
            response_parts = []
            intro_text = display_result.get('intro_text', '')
            if intro_text:
                response_parts.append(intro_text)
            data = display_result.get('data', {})
            data_format = display_result.get('format', 'generic')
            formatted_data = self._format_display_result(data, data_format)
            if formatted_data:
                response_parts.append(formatted_data)
            response_parts.append(f'🔧 Agent Call: {agent_name}')
            return '\n\n'.join(response_parts)
        else:
            result = self._execute_agent_call(agent_call_config, user_guid, user_input, demo_data)
            return f'{result}\n\n🔧 Agent Call: {agent_name}'

    def _format_display_result(self, data, data_format):
        """
        Format rich data for display based on format type.
        
        Supported formats:
        - priority_dashboard: Morning priorities with critical items
        - pipeline_breakdown: Sector analysis with metrics
        - at_risk_deals_grid: Deal cards with risk factors
        - recovery_playbook: Action plans and strategies
        - email_draft: Complete email with metadata
        - presentation_outline: Slide-by-slide breakdown
        - generic: Fallback JSON formatting
        
        Args:
            data: The data dict to format
            data_format: The format type string
            
        Returns:
            Formatted string for display
        """
        if data_format == 'priority_dashboard':
            return self._format_priority_dashboard(data)
        elif data_format == 'pipeline_breakdown':
            return self._format_pipeline_breakdown(data)
        elif data_format == 'at_risk_deals_grid':
            return self._format_deals_grid(data)
        elif data_format == 'recovery_playbook':
            return self._format_recovery_playbook(data)
        elif data_format == 'email_draft':
            return self._format_email_draft(data)
        elif data_format == 'presentation_outline':
            return self._format_presentation_outline(data)
        else:
            return json.dumps(data, indent=2)

    def _format_priority_dashboard(self, data):
        """Format morning priority dashboard with critical items and overnight changes."""
        output = []
        critical_items = data.get('critical_items', [])
        if critical_items:
            output.append("**🎯 Today's Priorities:**\n")
            for item in critical_items:
                output.append(f'{item.get('icon', '•')} **{item.get('title', 'Item')}**')
                output.append(f'   {item.get('value', '')} - {item.get('status', '')}')
                if 'description' in item:
                    output.append(f'   {item['description']}')
                output.append('')
        overnight_changes = data.get('overnight_changes', [])
        if overnight_changes:
            output.append('\n**🌙 Overnight Changes:**')
            for change in overnight_changes:
                output.append(f'  {change}')
        pipeline_summary = data.get('pipeline_summary', {})
        if pipeline_summary:
            output.append(f'\n**📊 Pipeline Summary:**')
            for key, value in pipeline_summary.items():
                label = key.replace('_', ' ').title()
                output.append(f'  {label}: {value}')
        return '\n'.join(output)

    def _format_pipeline_breakdown(self, data):
        """Format pipeline breakdown by sector with trends and metrics."""
        output = []
        sectors = data.get('sectors', [])
        for sector in sectors:
            output.append(f'\n{'=' * 60}')
            output.append(f'**{sector.get('name', 'Sector')}**')
            output.append(f'Total Value: {sector.get('total_value', 'N/A')} | Deals: {sector.get('deal_count', 0)} | Win Rate: {sector.get('win_rate', 'N/A')}')
            output.append(f'Avg Deal Size: {sector.get('average_deal_size', 'N/A')} | Trend: {sector.get('trend', 'N/A')}')
            top_deals = sector.get('top_deals', [])
            if top_deals:
                output.append(f'\nTop Deals:')
                for deal in top_deals:
                    output.append(f'  • {deal}')
            status = sector.get('status', '')
            if status:
                output.append(f'\n**Status:** {status}')
        health_metrics = data.get('pipeline_health_metrics', {})
        if health_metrics:
            output.append(f'\n{'=' * 60}')
            output.append(f'\n**Pipeline Health Metrics:**')
            for key, value in health_metrics.items():
                label = key.replace('_', ' ').title()
                output.append(f'  {label}: {value}')
        competitive = data.get('competitive_landscape', {})
        if competitive:
            output.append(f'\n**Competitive Landscape:**')
            if 'primary_competitors' in competitive:
                output.append(f'  Primary Competitors: {', '.join(competitive['primary_competitors'])}')
            if 'your_differentiators' in competitive:
                output.append(f'  Your Differentiators: {', '.join(competitive['your_differentiators'])}')
            if 'win_loss_trend' in competitive:
                output.append(f'  Win/Loss Trend: {competitive['win_loss_trend']}')
        return '\n'.join(output)

    def _format_deals_grid(self, data):
        """Format at-risk deals into a readable display with risk factors and links."""
        output = []
        deals = data.get('deals', [])
        for deal in deals:
            output.append(f'\n{'=' * 60}')
            output.append(f'**{deal.get('title', 'Deal')}** - {deal.get('company', 'Company')}')
            output.append(f'Value: {deal.get('value', 'N/A')} | Close: {deal.get('close_date', 'N/A')} | Risk: {deal.get('risk_level', 'N/A')} ({deal.get('risk_score', 'N/A')})')
            risk_factors = deal.get('risk_factors', [])
            if risk_factors:
                output.append(f'\n**Key Risk Factors:**')
                for factor in risk_factors:
                    output.append(f'  ⚠️ {factor}')
            stakeholders = deal.get('key_stakeholders', [])
            if stakeholders:
                output.append(f'\n**Key Stakeholders:**')
                for stakeholder in stakeholders:
                    output.append(f'  • {stakeholder}')
            links = []
            if 'dynamics_link' in deal:
                links.append(f'[View in Dynamics 365]({deal['dynamics_link']})')
            if 'teams_link' in deal:
                links.append(f'[Open in Teams]({deal['teams_link']})')
            if links:
                output.append(f'\n📊 {' | '.join(links)}')
            if 'last_activity' in deal:
                output.append(f'\nLast Activity: {deal['last_activity']}')
            if 'win_probability' in deal:
                output.append(f'Win Probability: {deal['win_probability']}')
            if 'competitive_threat' in deal:
                output.append(f'Competitive Threat: {deal['competitive_threat']}')
        summary_stats = data.get('summary_stats', {})
        if summary_stats:
            output.append(f'\n{'=' * 60}')
            output.append(f'\n**Summary Statistics:**')
            for key, value in summary_stats.items():
                label = key.replace('_', ' ').title()
                output.append(f'{label}: {value}')
        return '\n'.join(output)

    def _format_recovery_playbook(self, data):
        """Format comprehensive recovery playbook with action plans and strategies."""
        output = []
        deal_overview = data.get('deal_overview', {})
        if deal_overview:
            output.append('**Deal Overview:**')
            for key, value in deal_overview.items():
                label = key.replace('_', ' ').title()
                output.append(f'  {label}: {value}')
            output.append('')
        immediate_actions = data.get('immediate_actions', {})
        if immediate_actions:
            output.append(f'\n**{immediate_actions.get('title', 'Immediate Actions')}**')
            output.append(f'Priority: {immediate_actions.get('priority', 'HIGH')}\n')
            for item in immediate_actions.get('items', []):
                output.append(f'• **{item.get('action', 'Action')}**')
                output.append(f'  Owner: {item.get('owner', 'N/A')} | Timeline: {item.get('timeline', 'N/A')}')
                output.append(f'  {item.get('details', '')}')
                if item.get('template_available'):
                    output.append(f'  ✅ Template Available')
                output.append('')
        week_1 = data.get('week_1_strategy', {})
        if week_1:
            output.append(f'\n**{week_1.get('title', 'Week 1 Strategy')}**')
            for item in week_1.get('items', []):
                output.append(f'• **{item.get('action', 'Action')}**')
                output.append(f'  {item.get('details', '')}')
                if 'success_criteria' in item:
                    output.append(f'  ✓ Success: {item['success_criteria']}')
                output.append('')
        weeks_2_3 = data.get('weeks_2_3_strategy', {})
        if weeks_2_3:
            output.append(f'\n**{weeks_2_3.get('title', 'Weeks 2-3 Strategy')}**')
            for item in weeks_2_3.get('items', []):
                output.append(f'• **{item.get('action', 'Action')}**')
                output.append(f'  {item.get('details', '')}')
                if 'deliverable' in item:
                    output.append(f'  📋 Deliverable: {item['deliverable']}')
                output.append('')
        competitive = data.get('competitive_strategy', {})
        if competitive:
            output.append(f'\n**{competitive.get('title', 'Competitive Strategy')}**')
            output.append(f'Threat Level: {competitive.get('threat_level', 'Unknown')}\n')
            if 'their_strengths' in competitive:
                output.append(f'Their Strengths:')
                for strength in competitive['their_strengths']:
                    output.append(f'  • {strength}')
            if 'your_advantages' in competitive:
                output.append(f'\nYour Advantages:')
                for advantage in competitive['your_advantages']:
                    output.append(f'  ✓ {advantage}')
            if 'talking_points' in competitive:
                output.append(f'\nKey Talking Points:')
                for point in competitive['talking_points']:
                    output.append(f'  • {point}')
            if 'trap_setting' in competitive:
                output.append(f'\n💡 Trap Setting: {competitive['trap_setting']}')
            output.append('')
        stakeholder_plan = data.get('stakeholder_engagement_plan', {})
        if stakeholder_plan:
            output.append(f'\n**Stakeholder Engagement Plan:**\n')
            for stakeholder_key, stakeholder_data in stakeholder_plan.items():
                if isinstance(stakeholder_data, dict):
                    output.append(f'**{stakeholder_data.get('role', stakeholder_key)}**')
                    output.append(f'  Status: {stakeholder_data.get('status', 'N/A')}')
                    output.append(f'  Priority: {stakeholder_data.get('priority', 'N/A')}')
                    output.append(f'  Approach: {stakeholder_data.get('approach', 'N/A')}')
                    actions = stakeholder_data.get('actions', [])
                    if actions:
                        output.append(f'  Actions:')
                        for action in actions:
                            output.append(f'    • {action}')
                    win_signals = stakeholder_data.get('win_signals', '')
                    if win_signals:
                        output.append(f'  ✓ Win Signals: {win_signals}')
                    output.append('')
        probability = data.get('probability_improvement', {})
        if probability:
            output.append(f'\n**Probability Improvement Projection:**')
            output.append(f'  Current: {probability.get('current', 'N/A')} → With Playbook: {probability.get('with_playbook', 'N/A')}')
            output.append(f'  Expected Value Increase: {probability.get('expected_value_increase', 'N/A')}')
            output.append(f'  Time Investment: {probability.get('time_investment', 'N/A')}')
            output.append(f'  ROI: {probability.get('roi', 'N/A')}')
        return '\n'.join(output)

    def _format_email_draft(self, data):
        """Format executive email draft with metadata and full body."""
        output = []
        metadata = data.get('email_metadata', {})
        if metadata:
            output.append('**Email Details:**')
            output.append(f'To: {metadata.get('to', '')}')
            if 'cc' in metadata:
                output.append(f'Cc: {metadata['cc']}')
            output.append(f'Subject: {metadata.get('subject', '')}')
            output.append(f'Importance: {metadata.get('importance', 'Normal')}')
            output.append('\n' + '=' * 60 + '\n')
        body = data.get('email_body', {})
        if body:
            if 'greeting' in body:
                output.append(body['greeting'])
                output.append('')
            if 'opening' in body:
                output.append(body['opening'])
                output.append('')
            for paragraph in body.get('body_paragraphs', []):
                if 'section' in paragraph:
                    output.append(f'**{paragraph['section']}**')
                output.append(paragraph.get('content', ''))
                output.append('')
            if 'call_to_action' in body:
                output.append(body['call_to_action'])
                output.append('')
            if 'closing' in body:
                output.append(body['closing'])
                output.append('')
            if 'signature' in body:
                output.append(body['signature'])
        email_analysis = data.get('email_analysis', {})
        if email_analysis:
            output.append('\n' + '=' * 60)
            output.append('\n**Email Analysis:**')
            for key, value in email_analysis.items():
                label = key.replace('_', ' ').title()
                if isinstance(value, list):
                    output.append(f'{label}:')
                    for item in value:
                        output.append(f'  • {item}')
                else:
                    output.append(f'{label}: {value}')
        attachments = data.get('attachments_recommended', [])
        if attachments:
            output.append(f'\n**Recommended Attachments:**')
            for attachment in attachments:
                output.append(f'  • {attachment.get('name', 'File')} ({attachment.get('type', 'Document')})')
                output.append(f'    Status: {attachment.get('status', 'N/A')}')
        return '\n'.join(output)

    def _format_presentation_outline(self, data):
        """Format presentation outline with slide-by-slide breakdown."""
        output = []
        metadata = data.get('presentation_metadata', {})
        if metadata:
            output.append('**Presentation Details:**')
            output.append(f'Title: {metadata.get('title', 'Presentation')}')
            output.append(f'Subtitle: {metadata.get('subtitle', '')}')
            output.append(f'Audience: {metadata.get('audience', 'N/A')}')
            output.append(f'Duration: {metadata.get('duration', 'N/A')}')
            output.append(f'Total Slides: {metadata.get('total_slides', 0)}')
            output.append('')
        slides = data.get('slide_outline', [])
        if slides:
            output.append('**Slide-by-Slide Outline:**\n')
            for slide in slides:
                output.append(f'{'=' * 60}')
                output.append(f'**Slide {slide.get('slide_number', 0)}: {slide.get('title', 'Untitled')}**')
                content = slide.get('content', '')
                if content:
                    output.append(f'\nContent:')
                    output.append(content)
                visual = slide.get('visual', '')
                if visual:
                    output.append(f'\nVisual: {visual}')
                notes = slide.get('notes', '')
                if notes:
                    output.append(f'\nSpeaker Notes: {notes}')
                if slide.get('powerbi_chart'):
                    output.append(f'\n📊 Power BI Chart: {slide['powerbi_chart']}')
                output.append('')
        powerbi_integrations = data.get('powerbi_integrations', [])
        if powerbi_integrations:
            output.append(f'\n**Power BI Integrations:**')
            for integration in powerbi_integrations:
                output.append(f'  • {integration}')
        strengths = data.get('presentation_strengths', [])
        if strengths:
            output.append(f'\n**Presentation Strengths:**')
            for strength in strengths:
                output.append(f'  ✓ {strength}')
        tips = data.get('delivery_tips', [])
        if tips:
            output.append(f'\n**Delivery Tips:**')
            for tip in tips:
                output.append(f'  💡 {tip}')
        return '\n'.join(output)

    def _execute_agent_call(self, agent_call_config, user_guid, user_input, demo_data):
        """
        Execute an agent call with static and dynamic parameters.
        This is called when there's no display_result and the agent needs to be executed for real.

        Args:
            agent_call_config: The agent_call content block from JSON
            user_guid: User GUID for context
            user_input: User's message for dynamic parameter extraction
            demo_data: Full demo data for additional context

        Returns:
            Agent response or fallback message
        """
        try:
            agent_name = agent_call_config.get('agent', '')
            static_params = agent_call_config.get('static_parameters', {})
            dynamic_params_config = agent_call_config.get('dynamic_parameters', {})
            fallback = agent_call_config.get('fallback_response', 'Unable to complete the agent call.')
            description = agent_call_config.get('description', f'Calling {agent_name}')
            logging.info(f'Executing agent call: {agent_name} - {description}')
            dynamic_params = self._resolve_dynamic_parameters(dynamic_params_config, user_guid, user_input, demo_data)
            merged_params = {**static_params, **dynamic_params}
            logging.info(f'Agent call parameters: {json.dumps(merged_params, indent=2)}')
            agent = self._get_or_load_agent(agent_name)
            if not agent:
                logging.error(f"Agent '{agent_name}' not found locally or on GitHub")
                return fallback
            result = agent.perform(**merged_params)
            logging.info(f"Agent call to '{agent_name}' completed successfully")
            return result
        except Exception as e:
            logging.error(f'Error executing agent call: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
            return agent_call_config.get('fallback_response', f'Error executing agent: {str(e)}')

    def _resolve_dynamic_parameters(self, dynamic_params_config, user_guid, user_input, demo_data):
        """
        Resolve dynamic parameters from various sources.

        Dynamic parameter configuration format:
        {
            "param_name": {
                "source": "user_guid" | "user_input" | "context" | "infer",
                "description": "What this parameter is for",
                "extract_pattern": "Optional regex pattern for extraction",
                "default": "Optional default value"
            }
        }

        Or simplified format:
        {
            "param_name": "user_guid"  # Just the source as a string
        }
        """
        resolved_params = {}
        for param_name, config in dynamic_params_config.items():
            if isinstance(config, str):
                config = {'source': config}
            source = config.get('source', 'infer')
            default_value = config.get('default', None)
            extract_pattern = config.get('extract_pattern', None)
            resolved_value = None
            if source == 'user_guid':
                resolved_value = user_guid
            elif source == 'user_input':
                if extract_pattern:
                    match = re.search(extract_pattern, user_input, re.IGNORECASE)
                    if match:
                        resolved_value = match.group(1) if match.groups() else match.group(0)
                else:
                    resolved_value = user_input
            elif source == 'context':
                context_key = config.get('context_key', param_name)
                resolved_value = demo_data.get(context_key, default_value)
            elif source == 'infer':
                resolved_value = config.get('description', 'Inferred by assistant')
            if resolved_value is None and default_value is not None:
                resolved_value = default_value
            if resolved_value is not None:
                resolved_params[param_name] = resolved_value
        return resolved_params

    def _get_or_load_agent(self, agent_name):
        """
        Get an agent instance, loading from GitHub if not available locally.

        Args:
            agent_name: Name of the agent to load

        Returns:
            Agent instance or None if not found
        """
        if self.agent_manager:
            try:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    logging.info(f"Agent '{agent_name}' found locally via AgentManager")
                    return agent
            except Exception as e:
                logging.debug(f'Error checking local AgentManager: {str(e)}')
        if agent_name in self.remote_agent_cache:
            logging.info(f"Agent '{agent_name}' found in remote cache")
            return self.remote_agent_cache[agent_name]
        logging.info(f"Agent '{agent_name}' not found locally, attempting to load from GitHub...")
        agent = self._load_agent_from_github(agent_name)
        if agent:
            self.remote_agent_cache[agent_name] = agent
            logging.info(f"Agent '{agent_name}' successfully loaded from GitHub and cached")
            return agent
        logging.error(f"Agent '{agent_name}' not found locally or on GitHub")
        return None

    def _fetch_agent_manifest(self):
        """
        Attempt to fetch agent manifest from GitHub for faster agent discovery.
        This is optional - if manifest doesn't exist, falls back to path-based search.

        Returns:
            Manifest dict or None if not available
        """
        if self._agent_manifest_cache is not None:
            return self._agent_manifest_cache
        try:
            manifest_url = f'{self.GITHUB_RAW_BASE}/manifest.json'
            logging.debug(f'Attempting to fetch agent manifest from {manifest_url}')
            response = requests.get(manifest_url, timeout=5)
            response.raise_for_status()
            manifest = response.json()
            self._agent_manifest_cache = manifest
            logging.info(f'Agent manifest loaded successfully: {len(manifest.get('agents', []))} singular agents, {len(manifest.get('stacks', []))} stacks')
            return manifest
        except requests.exceptions.RequestException as e:
            logging.debug(f'No manifest found (will use path-based search): {str(e)}')
            self._agent_manifest_cache = {}
            return None
        except Exception as e:
            logging.debug(f'Error loading manifest: {str(e)}')
            self._agent_manifest_cache = {}
            return None

    def _find_agent_in_manifest(self, agent_name):
        """
        Find agent path using manifest if available.

        Args:
            agent_name: Name of the agent to find

        Returns:
            Agent file path or None if not found in manifest
        """
        manifest = self._fetch_agent_manifest()
        if not manifest:
            return None
        snake_case_name = self._convert_to_snake_case(agent_name)
        for agent in manifest.get('agents', []):
            if agent.get('id') == snake_case_name or agent.get('id') == agent_name:
                url = agent.get('url', '')
                if self.GITHUB_RAW_BASE in url:
                    path = url.replace(self.GITHUB_RAW_BASE + '/', '')
                    logging.info(f"Found agent '{agent_name}' in manifest: {path}")
                    return path
        for stack in manifest.get('stacks', []):
            for agent in stack.get('agents', []):
                if agent.get('id') == snake_case_name or agent.get('id') == agent_name:
                    url = agent.get('url', '')
                    if self.GITHUB_RAW_BASE in url:
                        path = url.replace(self.GITHUB_RAW_BASE + '/', '')
                        logging.info(f"Found stack agent '{agent_name}' in manifest: {path}")
                        return path
        return None

    def _load_agent_from_github(self, agent_name):
        """
        Load an agent from GitHub repository.

        Strategy:
        1. Check manifest (if available) for exact agent location
        2. Fall back to searching multiple possible locations:
           - agents/{agent_name}_agent.py
           - agent_stacks/*/{agent_name}_stack/agents/{agent_name}_agent.py

        Args:
            agent_name: Name of the agent to load

        Returns:
            Agent instance or None if not found
        """
        manifest_path = self._find_agent_in_manifest(agent_name)
        if manifest_path:
            agent = self._fetch_and_load_agent_from_path(agent_name, manifest_path)
            if agent:
                return agent
        snake_case_name = self._convert_to_snake_case(agent_name)
        possible_paths = [f'agents/{snake_case_name}.py', f'agents/{snake_case_name}_agent.py', f'agents/{agent_name}.py']
        stack_categories = ['b2b_sales', 'b2c_sales', 'energy', 'federal_government', 'financial_services', 'healthcare', 'manufacturing', 'professional_services', 'retail_cpg', 'slg_government', 'software_dp']
        for category in stack_categories:
            possible_paths.extend([f'agent_stacks/{category}_stacks/{snake_case_name}_stack/agents/{snake_case_name}_agent.py', f'agent_stacks/{category}_stacks/{snake_case_name}_stack/agents/{snake_case_name}.py', f'agent_stacks/{category}_stacks/{agent_name}_stack/agents/{agent_name}.py'])
        for path in possible_paths:
            agent = self._fetch_and_load_agent_from_path(agent_name, path)
            if agent:
                return agent
        return None

    def _fetch_and_load_agent_from_path(self, agent_name, file_path):
        """
        Fetch agent code from GitHub and dynamically load it.
        Uses requests library for robust HTTP handling.

        Args:
            agent_name: Name of the agent
            file_path: Path to the agent file in the repo

        Returns:
            Agent instance or None if fetch/load fails
        """
        try:
            url = f'{self.GITHUB_RAW_BASE}/{file_path}'
            logging.info(f'Attempting to fetch agent from: {url}')
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            agent_code = response.text
            logging.info(f'Successfully fetched agent code from {url} ({len(agent_code)} bytes)')
            agent_instance = self._load_agent_from_code(agent_name, agent_code, url)
            return agent_instance
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logging.debug(f'Agent not found at {url}')
            else:
                logging.warning(f'HTTP error fetching agent from {url}: {e.response.status_code}')
            return None
        except requests.exceptions.Timeout:
            logging.warning(f'Timeout fetching agent from {url}')
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f'Request error fetching agent from {url}: {str(e)}')
            return None
        except Exception as e:
            logging.error(f'Error fetching/loading agent from {url}: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
            return None

    def _load_agent_from_code(self, agent_name, code, source_url):
        """
        Dynamically load an agent from Python code string.

        Args:
            agent_name: Name of the agent
            code: Python code as string
            source_url: URL where code was fetched from (for reference)

        Returns:
            Agent instance or None if load fails
        """
        try:
            module_name = f'dynamic_agent_{agent_name}_{id(code)}'
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            exec(code, module.__dict__)
            agent_class = None
            for name, obj in module.__dict__.items():
                if isinstance(obj, type) and name.endswith('Agent') and (name != 'BasicAgent') and hasattr(obj, 'perform'):
                    agent_class = obj
                    break
            if not agent_class:
                logging.error(f'No agent class found in code from {source_url}')
                return None
            agent_instance = agent_class()
            logging.info(f'Successfully instantiated {agent_class.__name__} from {source_url}')
            return agent_instance
        except Exception as e:
            logging.error(f'Error loading agent from code: {str(e)}')
            import traceback
            logging.error(traceback.format_exc())
            return None

    def _convert_to_snake_case(self, name):
        """
        Convert CamelCase or PascalCase to snake_case.

        Args:
            name: String to convert

        Returns:
            snake_case version of the string
        """
        if name.endswith('Agent'):
            name = name[:-5]
        s1 = re.sub('(.)([A-Z][a-z]+)', '\\1_\\2', name)
        return re.sub('([a-z0-9])([A-Z])', '\\1_\\2', s1).lower()

    def format_error_response(self, error_message):
        """
        Format an error response in a consistent way.
        """
        response = {'status': 'error', 'error': error_message, 'available_actions': ['list_demos - List all available demo files', 'load_demo - Load a specific demo and see its structure', 'respond - Get canned response for user input']}
        return json.dumps(response, indent=2)

