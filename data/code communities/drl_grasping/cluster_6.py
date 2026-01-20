# Cluster 6

class ExperimentManager(object):
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.

    Please take a look at `train.py` to have the details for each argument.
    """

    def __init__(self, args: argparse.Namespace, algo: str, env_id: str, log_folder: str, tensorboard_log: str='', n_timesteps: int=0, eval_freq: int=10000, n_eval_episodes: int=5, save_freq: int=-1, hyperparams: Optional[Dict[str, Any]]=None, env_kwargs: Optional[Dict[str, Any]]=None, trained_agent: str='', optimize_hyperparameters: bool=False, storage: Optional[str]=None, study_name: Optional[str]=None, n_trials: int=1, n_jobs: int=1, sampler: str='tpe', pruner: str='median', optimization_log_path: Optional[str]=None, n_startup_trials: int=0, n_evaluations: int=1, truncate_last_trajectory: bool=False, uuid_str: str='', seed: int=0, log_interval: int=0, save_replay_buffer: bool=False, preload_replay_buffer: str='', verbose: int=1, vec_env_type: str='dummy', n_eval_envs: int=1, no_optim_plots: bool=False):
        super(ExperimentManager, self).__init__()
        self.algo = algo
        self.env_id = env_id
        self.custom_hyperparams = hyperparams
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = n_timesteps
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = seed
        self.optimization_log_path = optimization_log_path
        self.vec_env_class = {'dummy': DummyVecEnv, 'subproc': SubprocVecEnv}[vec_env_type]
        self.vec_env_kwargs = {}
        self.specified_callbacks = []
        self.callbacks = []
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.n_eval_envs = n_eval_envs
        self.n_envs = 1
        self.n_actions = None
        self._hyperparams = {}
        self.trained_agent = trained_agent
        self.continue_training = trained_agent.endswith('.zip') and os.path.isfile(trained_agent)
        self.truncate_last_trajectory = truncate_last_trajectory
        self.preload_replay_buffer = preload_replay_buffer
        self._is_atari = self.is_atari(env_id)
        self._is_gazebo_env = self.is_gazebo_env(env_id)
        self.optimize_hyperparameters = optimize_hyperparameters
        self.storage = storage
        self.study_name = study_name
        self.no_optim_plots = no_optim_plots
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.sampler = sampler
        self.pruner = pruner
        self.n_startup_trials = n_startup_trials
        self.n_evaluations = n_evaluations
        self.deterministic_eval = not self.is_atari(self.env_id)
        self.log_folder = log_folder
        self.tensorboard_log = None if tensorboard_log == '' else os.path.join(tensorboard_log, env_id)
        self.verbose = verbose
        self.args = args
        self.log_interval = log_interval
        self.save_replay_buffer = save_replay_buffer
        self.log_path = f'{log_folder}/{self.algo}/'
        self.save_path = os.path.join(self.log_path, f'{self.env_id}_{get_latest_run_id(self.log_path, self.env_id) + 1}{uuid_str}')
        self.params_path = f'{self.save_path}/{self.env_id}'

    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks = self._preprocess_hyperparams(hyperparams)
        self._env = self.create_envs(self.n_envs, no_log=False)
        self.create_log_folder()
        self.create_callbacks()
        self._hyperparams = self._preprocess_action_noise(hyperparams, self._env)
        if self.continue_training:
            model = self._load_pretrained_agent(self._hyperparams, self._env)
        elif self.optimize_hyperparameters:
            return None
        else:
            model = ALGOS[self.algo](env=self._env, tensorboard_log=self.tensorboard_log, seed=self.seed, verbose=self.verbose, **self._hyperparams)
        if self.preload_replay_buffer:
            if self.preload_replay_buffer.endswith('.pkl'):
                replay_buffer_path = self.preload_replay_buffer
            else:
                replay_buffer_path = os.path.join(self.preload_replay_buffer, 'replay_buffer.pkl')
            if os.path.exists(replay_buffer_path):
                print('Pre-loading replay buffer')
                if self.algo == 'her':
                    model.load_replay_buffer(replay_buffer_path, self.truncate_last_trajectory)
                else:
                    model.load_replay_buffer(replay_buffer_path)
            else:
                raise Exception(f'Replay buffer {replay_buffer_path} does not exist')
        self._save_config(saved_hyperparams)
        return model

    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {'log_interval': self.log_interval}
        if len(self.callbacks) > 0:
            kwargs['callback'] = self.callbacks
        if self.continue_training:
            kwargs['reset_num_timesteps'] = False
            model.env.reset()
        try:
            model.learn(self.n_timesteps, **kwargs)
        except Exception as e:
            print(f'Caught an exception during training of the model: {e}')
            self.save_trained_model(model)
        finally:
            try:
                model.env.close()
            except EOFError:
                pass

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics

        :param model:
        """
        print(f'Saving to {self.save_path}')
        model.save(f'{self.save_path}/{self.env_id}')
        if hasattr(model, 'save_replay_buffer') and self.save_replay_buffer:
            print('Saving replay buffer')
            model.save_replay_buffer(os.path.join(self.save_path, 'replay_buffer.pkl'))
        if self.normalize:
            model.get_vec_normalize_env().save(os.path.join(self.params_path, 'vecnormalize.pkl'))

    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be use later
        to reproduce an experiment.

        :param saved_hyperparams:
        """
        with open(os.path.join(self.params_path, 'config.yml'), 'w') as f:
            yaml.dump(saved_hyperparams, f)
        with open(os.path.join(self.params_path, 'args.yml'), 'w') as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)
        print(f'Log path: {self.save_path}')

    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        hyperparams_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), *3 * [os.path.pardir], 'hyperparams'))
        with open(f'{hyperparams_dir}/{self.algo}.yml', 'r') as f:
            hyperparams_dict = yaml.safe_load(f)
            if self.env_id in list(hyperparams_dict.keys()):
                hyperparams = hyperparams_dict[self.env_id]
            elif self._is_atari:
                hyperparams = hyperparams_dict['atari']
            else:
                raise ValueError(f'Hyperparameters not found for {self.algo}-{self.env_id}')
        if self.custom_hyperparams is not None:
            hyperparams.update(self.custom_hyperparams)
        saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
        if self.verbose > 0:
            print('Default hyperparameters for environment (ones being tuned will be overridden):')
            pprint(saved_hyperparams)
        return (hyperparams, saved_hyperparams)

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['learning_rate', 'clip_range', 'clip_range_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f'Invalid value for {key}: {hyperparams[key]}')
        return hyperparams

    def _preprocess_normalization(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        if 'normalize' in hyperparams.keys():
            self.normalize = hyperparams['normalize']
            if isinstance(self.normalize, str):
                self.normalize_kwargs = eval(self.normalize)
                self.normalize = True
            if 'gamma' in hyperparams:
                self.normalize_kwargs['gamma'] = hyperparams['gamma']
            del hyperparams['normalize']
        return hyperparams

    def _preprocess_hyperparams(self, hyperparams: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Callable], List[BaseCallback]]:
        self.n_envs = hyperparams.get('n_envs', 1)
        if self.verbose > 0:
            print(f'Using {self.n_envs} environments')
        hyperparams = self._preprocess_schedules(hyperparams)
        if 'train_freq' in hyperparams and isinstance(hyperparams['train_freq'], list):
            hyperparams['train_freq'] = tuple(hyperparams['train_freq'])
        if self.n_timesteps > 0:
            if self.verbose:
                print(f'Overwriting n_timesteps with n={self.n_timesteps}')
        else:
            self.n_timesteps = int(hyperparams['n_timesteps'])
        hyperparams = self._preprocess_normalization(hyperparams)
        for kwargs_key in {'policy_kwargs', 'replay_buffer_class', 'replay_buffer_kwargs'}:
            if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
                hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])
        if 'n_envs' in hyperparams.keys():
            del hyperparams['n_envs']
        del hyperparams['n_timesteps']
        if 'frame_stack' in hyperparams.keys():
            self.frame_stack = hyperparams['frame_stack']
            del hyperparams['frame_stack']
        env_wrapper = get_wrapper_class(hyperparams)
        if 'env_wrapper' in hyperparams.keys():
            del hyperparams['env_wrapper']
        callbacks = get_callback_list(hyperparams)
        if 'callback' in hyperparams.keys():
            self.specified_callbacks = hyperparams['callback']
            del hyperparams['callback']
        return (hyperparams, env_wrapper, callbacks)

    def _preprocess_action_noise(self, hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
        if hyperparams.get('noise_type') is not None:
            noise_type = hyperparams['noise_type'].strip()
            noise_std = hyperparams['noise_std']
            self.n_actions = env.action_space.shape[0]
            if 'normal' in noise_type:
                hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(self.n_actions), sigma=noise_std * np.ones(self.n_actions))
            elif 'ornstein-uhlenbeck' in noise_type:
                hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(self.n_actions), sigma=noise_std * np.ones(self.n_actions))
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')
            print(f'Applying {noise_type} noise with std {noise_std}')
            del hyperparams['noise_type']
            del hyperparams['noise_std']
        return hyperparams

    def create_log_folder(self):
        os.makedirs(self.params_path, exist_ok=True)

    def create_callbacks(self):
        if self.save_freq > 0:
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(CheckpointCallbackWithReplayBuffer(save_freq=self.save_freq, save_path=self.save_path, name_prefix='rl_model', save_replay_buffer=self.save_replay_buffer, verbose=self.verbose))
        if self.eval_freq > 0 and (not self.optimize_hyperparameters):
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)
            if self.verbose > 0:
                print('Creating test environment')
            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = EvalCallback(eval_env=self._env, callback_on_new_best=save_vec_normalize, best_model_save_path=self.save_path, n_eval_episodes=self.n_eval_episodes, log_path=self.save_path, eval_freq=self.eval_freq, deterministic=self.deterministic_eval)
            self.callbacks.append(eval_callback)

    @staticmethod
    def is_atari(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return 'AtariEnv' in str(entry_point)

    @staticmethod
    def is_bullet(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return 'pybullet_envs' in str(entry_point)

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return 'gym.envs.robotics' in str(entry_point) or 'panda_gym.envs' in str(entry_point)

    @staticmethod
    def is_gazebo_env(env_id: str) -> bool:
        return 'Gazebo' in gym.envs.registry.env_specs[env_id].entry_point

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.

        :param env:
        :param eval_env:
        :return:
        """
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, 'vecnormalize.pkl')
        if os.path.exists(path_):
            print('Loading saved VecNormalize stats')
            env = VecNormalize.load(path_, env)
            if eval_env:
                env.training = False
                env.norm_reward = False
        elif self.normalize:
            local_normalize_kwargs = self.normalize_kwargs.copy()
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs['norm_reward'] = False
                else:
                    local_normalize_kwargs = {'norm_reward': False}
            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    print(f'Normalization activated: {local_normalize_kwargs}')
                else:
                    print('Normalizing input and reward')
            env.num_envs = self.n_envs
            env = VecNormalize(env, **local_normalize_kwargs)
        return env

    def create_envs(self, n_envs: int, eval_env: bool=False, no_log: bool=False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        log_dir = None if eval_env or no_log else self.save_path
        monitor_kwargs = {}
        if 'Neck' in self.env_id or self.is_robotics_env(self.env_id) or 'parking-v0' in self.env_id:
            monitor_kwargs = dict(info_keywords=('is_success',))
        env = make_vec_env(env_id=self.env_id, n_envs=n_envs, seed=self.seed, env_kwargs=self.env_kwargs, monitor_dir=log_dir, wrapper_class=self.env_wrapper, vec_env_cls=self.vec_env_class, vec_env_kwargs=self.vec_env_kwargs, monitor_kwargs=monitor_kwargs)
        env = self._maybe_normalize(env, eval_env)
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                print(f'Stacking {n_stack} frames')
        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (is_image_space(space) and (not is_image_space_channels_first(space)))
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and (not is_image_space_channels_first(env.observation_space))
            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    print('Wrapping the env in a VecTransposeImage.')
                env = VecTransposeImage(env)
        return env

    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        print(f"Loading pretrained agent '{self.trained_agent}' to continue its training")
        del hyperparams['policy']
        if 'policy_kwargs' in hyperparams.keys():
            del hyperparams['policy_kwargs']
        model = ALGOS[self.algo].load(self.trained_agent, env=env, seed=self.seed, tensorboard_log=self.tensorboard_log, verbose=self.verbose, **hyperparams)
        replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), 'replay_buffer.pkl')
        if not self.preload_replay_buffer and os.path.exists(replay_buffer_path):
            print('Loading replay buffer')
            model.load_replay_buffer(replay_buffer_path, truncate_last_traj=self.truncate_last_trajectory)
        return model

    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        if sampler_method == 'random':
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == 'tpe':
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        elif sampler_method == 'skopt':
            sampler = SkoptSampler(skopt_kwargs={'base_estimator': 'GP', 'acq_func': 'gp_hedge'})
        else:
            raise ValueError(f'Unknown sampler: {sampler_method}')
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == 'halving':
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == 'median':
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == 'none':
            pruner = NopPruner()
        else:
            raise ValueError(f'Unknown pruner: {pruner_method}')
        return pruner

    def objective(self, trial: optuna.Trial) -> float:
        kwargs = self._hyperparams.copy()
        trial.model_class = None
        trial.n_actions = self._env.action_space.shape[0]
        if kwargs.get('replay_buffer_class') == HerReplayBuffer:
            trial.her_kwargs = kwargs.get('replay_buffer_kwargs', {})
        kwargs.update(HYPERPARAMS_SAMPLER[self.algo](trial))
        print(f'\nRunning a new trial with hyperparameters: {kwargs}')
        trial_params_path = os.path.join(self.params_path, 'optimization')
        os.makedirs(trial_params_path, exist_ok=True)
        with open(os.path.join(trial_params_path, f'hyperparameters_trial_{trial.number}.yml'), 'w') as f:
            yaml.dump(kwargs, f)
        model = ALGOS[self.algo](env=self._env, tensorboard_log=self.tensorboard_log, seed=self.seed, verbose=self.verbose, **kwargs)
        if self.preload_replay_buffer:
            if self.preload_replay_buffer.endswith('.pkl'):
                replay_buffer_path = self.preload_replay_buffer
            else:
                replay_buffer_path = os.path.join(self.preload_replay_buffer, 'replay_buffer.pkl')
            if os.path.exists(replay_buffer_path):
                print('Pre-loading replay buffer')
                if self.algo == 'her':
                    model.load_replay_buffer(replay_buffer_path, self.truncate_last_trajectory)
                else:
                    model.load_replay_buffer(replay_buffer_path)
            else:
                raise Exception(f'Replay buffer {replay_buffer_path} does not exist')
        model.trial = trial
        eval_freq = int(self.n_timesteps / self.n_evaluations)
        eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        callbacks = get_callback_list({'callback': self.specified_callbacks})
        path = None
        if self.optimization_log_path is not None:
            path = os.path.join(self.optimization_log_path, f'trial_{str(trial.number)}')
        eval_callback = TrialEvalCallback(model.env, model.trial, best_model_save_path=path, log_path=path, n_eval_episodes=self.n_eval_episodes, eval_freq=eval_freq_, deterministic=self.deterministic_eval, verbose=self.verbose)
        callbacks.append(eval_callback)
        try:
            model.learn(self.n_timesteps, callback=callbacks)
            self._env.reset()
        except AssertionError as e:
            self._env.reset()
            print('Trial stopped:', e)
            raise optuna.exceptions.TrialPruned()
        except Exception as err:
            exception_type = type(err).__name__
            print('Trial stopped due to raised exception:', exception_type, err)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward
        print(f'\nFinished a trial with reward={reward}, is_pruned={is_pruned} for hyperparameters: {kwargs}')
        del model
        if is_pruned:
            raise optuna.exceptions.TrialPruned()
        return reward

    def hyperparameters_optimization(self) -> None:
        if self.verbose > 0:
            print('Optimizing hyperparameters')
        if self.storage is not None and self.study_name is None:
            warnings.warn(f'You passed a remote storage: {self.storage} but no `--study-name`.The study name will be generated by Optuna, make sure to re-use the same study name when you want to do distributed hyperparameter optimization.')
        if self.tensorboard_log is not None:
            warnings.warn('Tensorboard log is deactivated when running hyperparameter optimization')
            self.tensorboard_log = None
        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)
        if self.verbose > 0:
            print(f'Sampler: {self.sampler} - Pruner: {self.pruner}')
        study = optuna.create_study(sampler=sampler, pruner=pruner, storage=self.storage, study_name=self.study_name, load_if_exists=True, direction='maximize')
        try:
            study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs, gc_after_trial=True, show_progress_bar=True)
        except KeyboardInterrupt:
            pass
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('Value: ', trial.value)
        print('Params: ')
        for key, value in trial.params.items():
            print(f'    {key}: {value}')
        report_name = f'report_{self.env_id}_{self.n_trials}-trials-{self.n_timesteps}-{self.sampler}-{self.pruner}_{int(time.time())}'
        log_path = os.path.join(self.log_folder, self.algo, report_name)
        if self.verbose:
            print(f'Writing report to {log_path}')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(f'{log_path}.csv')
        with open(f'{log_path}.pkl', 'wb+') as f:
            pkl.dump(study, f)
        if self.no_optim_plots:
            return
        try:
            fig1 = plot_optimization_history(study)
            fig2 = plot_param_importances(study)
            fig1.show()
            fig2.show()
        except (ValueError, ImportError, RuntimeError):
            pass

    def collect_demonstration(self, model):
        action = np.array([model.env.action_space.sample()])
        obs = model.env.reset()
        for i in range(model.replay_buffer.buffer_size):
            next_obs, rewards, dones, infos = model.env.unwrapped.step(action)
            actual_actions = [info['actual_actions'] for info in infos]
            model.replay_buffer.add(obs, next_obs, actual_actions, rewards, dones)
            obs = next_obs
        print('Saving replay buffer')
        model.save_replay_buffer(os.path.join(self.save_path, 'replay_buffer.pkl'))
        model.env.close()
        exit

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value
    return func

