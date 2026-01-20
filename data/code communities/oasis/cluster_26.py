# Cluster 26

def test_agent_graph(tmp_path):
    twitter_channel = Channel()
    graph = AgentGraph()
    agent_0 = SocialAgent(agent_id=0, user_info=UserInfo(name='0'), channel=twitter_channel)
    agent_1 = SocialAgent(agent_id=1, user_info=UserInfo(name='1'), channel=twitter_channel)
    agent_2 = SocialAgent(agent_id=2, user_info=UserInfo(name='2'), channel=twitter_channel)
    graph.add_agent(agent_0)
    graph.add_agent(agent_1)
    graph.add_agent(agent_2)
    assert graph.get_num_nodes() == 3
    assert len(graph.agent_mappings) == 3
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    assert graph.get_num_edges() == 2
    edges = graph.get_edges()
    assert len(edges) == 2
    assert edges[0] == (0, 1)
    assert edges[1] == (0, 2)
    img_path = osp.join(tmp_path, 'img.pdf')
    graph.visualize(img_path)
    assert osp.exists(img_path)
    agents = graph.get_agents()
    assert len(agents) == 3
    assert agents[0][0] == 0
    assert id(agents[0][1]) == id(agent_0)
    assert agents[1][0] == 1
    assert id(agents[1][1]) == id(agent_1)
    assert agents[2][0] == 2
    assert id(agents[2][1]) == id(agent_2)
    graph.remove_edge(0, 1)
    assert graph.get_num_edges() == 1
    graph.remove_agent(agent_0)
    assert len(graph.agent_mappings) == 2
    assert graph.get_num_nodes() == 2
    assert graph.get_num_edges() == 0
    graph.reset()
    assert len(graph.agent_mappings) == 0
    assert graph.get_num_nodes() == 0
    assert graph.get_num_edges() == 0

@pytest.mark.skipif(not neo4j_vars_set(), reason='One or more neo4j env variables are not set')
def test_agent_neo4j_graph():
    channel = Channel()
    graph = AgentGraph(backend='neo4j', neo4j_config=Neo4jConfig(uri=os.getenv('NEO4J_URI'), username=os.getenv('NEO4J_USERNAME'), password=os.getenv('NEO4J_PASSWORD')))
    agent_0 = SocialAgent(agent_id=0, user_info=UserInfo(name='0'), channel=channel)
    agent_1 = SocialAgent(agent_id=1, user_info=UserInfo(name='1'), channel=channel)
    agent_2 = SocialAgent(agent_id=2, user_info=UserInfo(name='2'), channel=channel)
    graph.add_agent(agent_0)
    graph.add_agent(agent_1)
    graph.add_agent(agent_2)
    assert graph.get_num_nodes() == 3
    assert len(graph.agent_mappings) == 3
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    assert graph.get_num_edges() == 2
    edges = graph.get_edges()
    assert len(edges) == 2
    assert edges[0] == (0, 1)
    assert edges[1] == (0, 2)
    agents = graph.get_agents()
    assert len(agents) == 3
    assert agents[0][0] == 0
    assert id(agents[0][1]) == id(agent_0)
    assert agents[1][0] == 1
    assert id(agents[1][1]) == id(agent_1)
    assert agents[2][0] == 2
    assert id(agents[2][1]) == id(agent_2)
    graph.remove_edge(0, 1)
    assert graph.get_num_edges() == 1
    graph.remove_agent(agent_0)
    assert len(graph.agent_mappings) == 2
    assert graph.get_num_nodes() == 2
    assert graph.get_num_edges() == 0
    graph.reset()
    assert len(graph.agent_mappings) == 0
    assert graph.get_num_nodes() == 0
    assert graph.get_num_edges() == 0

def neo4j_vars_set() -> bool:
    return os.getenv('NEO4J_URI') is not None and os.getenv('NEO4J_USERNAME') is not None and (os.getenv('NEO4J_PASSWORD') is not None)

class OasisEnv:

    def __init__(self, agent_graph: AgentGraph, platform: Union[DefaultPlatformType, Platform], database_path: str=None, semaphore: int=128) -> None:
        """Init the oasis environment.

        Args:
            agent_graph: The AgentGraph to use in the simulation.
            platform: The platform type to use. Including
                `DefaultPlatformType.TWITTER` or `DefaultPlatformType.REDDIT`.
                Or you can pass a custom `Platform` instance.
            database_path: The path to create a sqlite3 database. The file
                extension must be `.db` such as `twitter_simulation.db`.
        """
        self.agent_graph = agent_graph
        self.llm_semaphore = asyncio.Semaphore(semaphore)
        if isinstance(platform, DefaultPlatformType):
            if database_path is None:
                raise ValueError('database_path is required for DefaultPlatformType')
            self.platform = platform
            if platform == DefaultPlatformType.TWITTER:
                self.channel = Channel()
                self.platform = Platform(db_path=database_path, channel=self.channel, recsys_type='twhin-bert', refresh_rec_post_count=2, max_rec_post_len=2, following_post_count=3)
                self.platform_type = DefaultPlatformType.TWITTER
            elif platform == DefaultPlatformType.REDDIT:
                self.channel = Channel()
                self.platform = Platform(db_path=database_path, channel=self.channel, recsys_type='reddit', allow_self_rating=True, show_score=True, max_rec_post_len=100, refresh_rec_post_count=5)
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                raise ValueError(f'Invalid platform: {platform}. Only DefaultPlatformType.TWITTER or DefaultPlatformType.REDDIT are supported.')
        elif isinstance(platform, Platform):
            if database_path != platform.db_path:
                env_log.warning('database_path is not the same as the platform.db_path, using the platform.db_path')
            self.platform = platform
            self.channel = platform.channel
            if platform.recsys_type == RecsysType.REDDIT:
                self.platform_type = DefaultPlatformType.REDDIT
            else:
                self.platform_type = DefaultPlatformType.TWITTER
        else:
            raise ValueError(f'Invalid platform: {platform}. You should pass a DefaultPlatformType or a Platform instance.')

    async def reset(self) -> None:
        """Start the platform and sign up the agents."""
        self.platform_task = asyncio.create_task(self.platform.running())
        self.agent_graph = await generate_custom_agents(channel=self.channel, agent_graph=self.agent_graph)

    async def _perform_llm_action(self, agent):
        """Send the request to the llm model and execute the action.
        """
        async with self.llm_semaphore:
            return await agent.perform_action_by_llm()

    async def _perform_interview_action(self, agent, interview_prompt: str):
        """Send the request to the llm model and execute the interview.
        """
        async with self.llm_semaphore:
            return await agent.perform_interview(interview_prompt)

    async def step(self, actions: dict[SocialAgent, Union[ManualAction, LLMAction, List[Union[ManualAction, LLMAction]]]]) -> None:
        """Update the recommendation system and perform the actions.

        Args:
            actions(dict[SocialAgent, Union[ManualAction, LLMAction,
                List[Union[ManualAction, LLMAction]]]]): The actions to
                perform, including the manual(pre-defined) actions and llm
                actions.
        Returns:
            None
        """
        await self.platform.update_rec_table()
        env_log.info('update rec table.')
        tasks = []
        for agent, action in actions.items():
            if isinstance(action, list):
                for single_action in action:
                    if isinstance(single_action, ManualAction):
                        if single_action.action_type == ActionType.INTERVIEW:
                            interview_prompt = single_action.action_args.get('prompt', '')
                            tasks.append(self._perform_interview_action(agent, interview_prompt))
                        else:
                            tasks.append(agent.perform_action_by_data(single_action.action_type, **single_action.action_args))
                    elif isinstance(single_action, LLMAction):
                        tasks.append(self._perform_llm_action(agent))
            elif isinstance(action, ManualAction):
                if action.action_type == ActionType.INTERVIEW:
                    interview_prompt = action.action_args.get('prompt', '')
                    tasks.append(self._perform_interview_action(agent, interview_prompt))
                else:
                    tasks.append(agent.perform_action_by_data(action.action_type, **action.action_args))
            elif isinstance(action, LLMAction):
                tasks.append(self._perform_llm_action(agent))
        await asyncio.gather(*tasks)
        env_log.info('performed all actions.')
        if self.platform_type == DefaultPlatformType.TWITTER:
            self.platform.sandbox_clock.time_step += 1

    async def close(self) -> None:
        """Stop the platform and close the environment.
        """
        await self.channel.write_to_receive_queue((None, None, ActionType.EXIT))
        await self.platform_task
        env_log.info(f'Simulation finished! Please check the results in the database: {self.platform.db_path}. Note that the trace table stored all the actions of the agents.')

