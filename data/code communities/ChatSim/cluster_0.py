# Cluster 0

class ChatSim:

    def __init__(self, config):
        self.config = config
        self.scene = Scene(config['scene'])
        agents_config = config['agents']
        self.project_manager = ProjectManager(agents_config['project_manager'])
        self.asset_select_agent = AssetSelectAgent(agents_config['asset_select_agent'])
        self.deletion_agent = DeletionAgent(agents_config['deletion_agent'])
        self.foreground_rendering_agent = ForegroundRenderingAgent(agents_config['foreground_rendering_agent'])
        self.motion_agent = MotionAgent(agents_config['motion_agent'])
        self.view_adjust_agent = ViewAdjustAgent(agents_config['view_adjust_agent'])
        if agents_config['background_rendering_agent'].get('scene_representation', 'nerf') == 'nerf':
            self.background_rendering_agent = BackgroundRenderingAgent(agents_config['background_rendering_agent'])
        else:
            self.background_rendering_agent = BackgroundRendering3DGSAgent(agents_config['background_rendering_agent'])
        self.tech_agents = {'asset_select_agent': self.asset_select_agent, 'background_rendering_agent': self.background_rendering_agent, 'deletion_agent': self.deletion_agent, 'foreground_rendering_agent': self.foreground_rendering_agent, 'motion_agent': self.motion_agent, 'view_adjust_agent': self.view_adjust_agent}
        self.current_prompt = 'An empty prompt'

    def setup_init_frame(self):
        """Setup initial frame for ChatSim's reasoning and rendering.
        """
        if not os.path.exists(self.scene.init_img_path):
            print(f'{colored('[Note]', color='red', attrs=['bold'])} ', f'{colored('can not find init image, rendering it for the first time')}\n')
            self.background_rendering_agent.func_render_background(self.scene)
            imageio.imwrite(self.scene.init_img_path, self.scene.current_images[0])
        else:
            self.scene.current_images = [imageio.imread(self.scene.init_img_path)] * self.scene.frames

    def execute_llms(self, prompt):
        """Entry of ChatSim's reasoning.
        We perform multi-LLM reasoning for the user's prompt

        Input:
            prompt : str
                language prompt to ChatSim.
        """
        self.scene.setup_cars()
        self.current_prompt = prompt
        tasks = self.project_manager.decompose_prompt(self.scene, prompt)
        for task in tasks.values():
            print(f'{colored('[Performing Single Prompt]', on_color='on_blue', attrs=['bold'])} {colored(task, attrs=['bold'])}\n')
            self.project_manager.dispatch_task(self.scene, task, self.tech_agents)
        print(colored('scene.added_cars_dict', color='red', attrs=['bold']), end=' ')
        pprint.pprint(self.scene.added_cars_dict.keys())
        print(colored('scene.removed_cars', color='red', attrs=['bold']), end=' ')
        pprint.pprint(self.scene.removed_cars)

    def execute_funcs(self):
        """Entry of ChatSim's rendering functions
        We perform agent's functions following the self.scene's configuration.
        self.scene's configuration are updated in self.execute_llms()
        """
        self.background_rendering_agent.func_render_background(self.scene)
        self.deletion_agent.func_inpaint_scene(self.scene)
        self.asset_select_agent.func_retrieve_blender_file(self.scene)
        self.foreground_rendering_agent.func_blender_add_cars(self.scene)
        generate_video(self.scene, self.current_prompt)

