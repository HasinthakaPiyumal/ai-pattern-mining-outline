# Cluster 9

def test_deep_merge_dicts():
    a = EasyDict(dict(name='aaa', content=dict(team_num=4, map_width=1000)))
    b = EasyDict(dict(name='bbb', content=dict(map_width=2000)))
    c = deep_merge_dicts(a, b)
    assert c.name == 'bbb'
    assert c.content.map_width == 2000
    assert c.content.team_num == 4

def deep_merge_dicts(original: dict, new_dict: dict) -> dict:
    """
    Overview:
        Merge two dicts by calling ``deep_update``
    Arguments:
        - original (:obj:`dict`): Dict 1.
        - new_dict (:obj:`dict`): Dict 2.
    Returns:
        - merged_dict (:obj:`dict`): A new dict that is d1 and d2 deeply merged.
    """
    original = original or {}
    new_dict = new_dict or {}
    merged = copy.deepcopy(original)
    if new_dict:
        deep_update(merged, new_dict, True, [])
    return merged

@total_ordering
class BaseBall(ABC):
    """
    Overview:
        Base class of all balls
    """

    @staticmethod
    def default_config():
        """
        Overview:
            Default config
        """
        cfg = dict()
        return EasyDict(cfg)

    def __init__(self, ball_id, position, score, border, **kwargs):
        """
        Parameters:
             vel <Vector2> : the direction of the ball's speed 
             acc <Vector2> : the direction of the ball's acceleration
        """
        self.ball_id = ball_id
        self.position = position
        kwargs = EasyDict(kwargs)
        cfg = BaseBall.default_config()
        cfg = deep_merge_dicts(cfg, kwargs)
        self.score = score
        self.border = border
        self.radius = self.score_to_radius(self.score)
        self.is_remove = False
        self.quad_node = None

    def set_score(self, score: float) -> None:
        self.score = score
        self.radius = self.score_to_radius(self.score)

    def radius_to_score(self, radius):
        return (math.pow(radius, 2) - 0.15) / 0.042 * 100

    def score_to_radius(self, score):
        return math.sqrt(score / 100 * 0.042 + 0.15)

    def move(self, direction, duration):
        """
        Overview:
            Realize the movement of the ball, pass in the direction and time parameters, and return the new position
        Parameters:
            direction <Vector2>: A point in the unit circle
            duration <float>: time
        Returns:
            position <Vector2>: position after moving 
        """
        raise NotImplementedError

    def eat(self, ball):
        """
        Overview:
            Describe the rules of eating and being eaten
        Parameters:
            ball <BaseBall>: Eaten ball
        """
        raise NotImplementedError

    def remove(self):
        """
        Overview:
            Things to do when being removed from the map
        """
        self.is_remove = True

    def check_border(self):
        """
        Overview:
            Check to see if the position of the ball exceeds the bounds of the map. 
            If it exceeds, the speed and acceleration in the corresponding direction will be zeroed, and the position will be edged
        """
        if self.position.x < self.border.minx or self.position.x > self.border.maxx:
            self.position.x = max(self.position.x, self.border.minx)
            self.position.x = min(self.position.x, self.border.maxx)
        if self.position.y < self.border.miny or self.position.y > self.border.maxy:
            self.position.y = max(self.position.y, self.border.miny)
            self.position.y = min(self.position.y, self.border.maxy)

    def get_dis(self, ball):
        """
        Overview:
            Get the distance between the centers of the two balls
        Parameters:
            ball <BaseBall>: another ball
        """
        return (self.position - ball.position).length()

    def judge_cover(self, ball):
        """
        Overview:
            Determine whether the center of the two balls is covered
        Parameters:
            ball <BaseBall>: another ball
        Returns:
            is_covered <bool>: covered or not
        """
        if ball.ball_id == self.ball_id:
            return False
        dis = self.get_dis(ball)
        if self.radius > dis or ball.radius > dis:
            return True
        else:
            return False

    def judge_in_rectangle(self, rectangle):
        """
        Overview:
            Determine if the ball and rectangle intersect
        Parameters:
            rectangle <List>: left_top_x, left_top_y, right_bottom_x, right_bottom_y
        Returns:
            <bool> : intersect or not
        """
        dx = rectangle[0] - self.position.x if rectangle[0] > self.position.x else self.position.x - rectangle[2] if self.position.x > rectangle[2] else 0
        dy = rectangle[1] - self.position.y if rectangle[1] > self.position.y else self.position.y - rectangle[3] if self.position.y > rectangle[3] else 0
        return dx ** 2 + dy ** 2 <= self.radius ** 2

    def __repr__(self) -> str:
        return 'position={}, score={:.3f}, radius={:.3f}'.format(self.position, self.score, self.radius)

    def __eq__(self, other):
        return self.score == other.score

    def __le__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

class SporeBall(BaseBall):
    """
    Overview:
        Spores spit out by the player ball
        - characteristic:
        * Can't move actively
        * can not eat
        * Can be eaten by CloneBall and ThornsBall
        * There is an initial velocity at birth, and it decays to 0 within a period of time
    """

    @staticmethod
    def default_config():
        cfg = BaseBall.default_config()
        cfg.update(dict(score_init=1.5, vel_init=50, vel_zero_frame=10))
        return EasyDict(cfg)

    def __init__(self, ball_id, position, border, score, direction=Vector2(0, 0), owner=-1, **kwargs):
        kwargs = EasyDict(kwargs)
        cfg = SporeBall.default_config()
        cfg = deep_merge_dicts(cfg, kwargs)
        super(SporeBall, self).__init__(ball_id, position, score=score, border=border, **cfg)
        self.score_init = cfg.score_init
        self.vel_init = cfg.vel_init
        self.vel_zero_frame = cfg.vel_zero_frame
        self.direction = direction.normalize()
        self.vel = self.vel_init * self.direction
        self.vel_piece = self.vel / self.vel_zero_frame
        self.owner = owner
        self.move_frame = 0
        if self.score != self.score_init:
            self.set_score(self.score_init)
        self.moving = True
        self.check_border()

    def move(self, direction=None, duration=0.05):
        assert direction is None
        assert duration > 0
        if self.moving:
            self.position = self.position + self.vel * duration
            self.move_frame += 1
            if self.move_frame < self.vel_zero_frame:
                self.vel -= self.vel_piece
            else:
                self.vel = Vector2(0, 0)
                self.vel_piece = Vector2(0, 0)
                self.moving = False
        self.check_border()
        return True

    def eat(self, ball):
        logging.debug('SporeBall can not eat others')
        return

    def save(self):
        return [self.position.x, self.position.y, self.radius]

class CloneBall(BaseBall):
    """
    Overview:
        One of the balls that a single player can control
        - characteristic:
        * Can move
        * Can eat any other ball smaller than itself
        * Under the control of the player, the movement can be stopped immediately and contracted towards the center of mass of the player
        * Skill 1: Split each unit into two equally
        * Skill 2: Spit spores forward
        * There is a percentage of weight attenuation, and the radius will shrink as the weight attenuates
    """

    @staticmethod
    def default_config():
        cfg = BaseBall.default_config()
        cfg.update(dict(acc_weight=100, vel_max=20, score_init=1, part_num_max=16, on_thorns_part_num=10, on_thorns_part_score_max=3, split_score_min=2.5, eject_score_min=2.5, recombine_frame=320, split_vel_zero_frame=40, score_decay_min=2600, score_decay_rate_per_frame=5e-05, center_acc_weight=10))
        return EasyDict(cfg)

    def __init__(self, ball_id, position, score, border, team_id, player_id, vel_given=Vector2(0, 0), acc_given=Vector2(0, 0), from_split=False, from_thorns=False, split_direction=Vector2(0, 0), spore_settings=SporeBall.default_config(), sequence_generator=None, **kwargs):
        kwargs = EasyDict(kwargs)
        cfg = CloneBall.default_config()
        cfg = deep_merge_dicts(cfg, kwargs)
        super(CloneBall, self).__init__(ball_id, position, score, border, **cfg)
        self.acc_weight = cfg.acc_weight
        self.vel_max = cfg.vel_max
        self.score_init = cfg.score_init
        self.part_num_max = cfg.part_num_max
        self.on_thorns_part_num = cfg.on_thorns_part_num
        self.on_thorns_part_score_max = cfg.on_thorns_part_score_max
        self.split_score_min = cfg.split_score_min
        self.eject_score_min = cfg.eject_score_min
        self.recombine_frame = cfg.recombine_frame
        self.split_vel_zero_frame = cfg.split_vel_zero_frame
        self.score_decay_min = cfg.score_decay_min
        self.score_decay_rate_per_frame = cfg.score_decay_rate_per_frame
        self.center_acc_weight = cfg.center_acc_weight
        self.spore_settings = spore_settings
        self.sequence_generator = sequence_generator
        self.cfg = cfg
        self.team_id = team_id
        self.player_id = player_id
        self.vel_given = vel_given
        self.acc_given = acc_given
        if from_split:
            self.vel_split = self.cal_split_vel_init_from_split(self.radius) * split_direction
        elif from_thorns:
            self.vel_split = self.cal_split_vel_init_from_thorns(self.radius) * split_direction
        else:
            self.vel_split = Vector2(0, 0)
        self.vel_split_piece = self.vel_split / self.split_vel_zero_frame
        self.split_frame = 0
        self.frame_since_last_split = 0
        self.vel = self.vel_given + self.vel_split
        self.update_direction()
        self.check_border()

    def update_direction(self):
        if self.vel.length() != 0:
            self.direction = copy.deepcopy(self.vel.normalize())
        else:
            self.direction = Vector2(random.random(), random.random()).normalize()

    def cal_vel_max(self, radius, ratio):
        return (2.35 + 5.66 / radius) * ratio

    def cal_split_vel_init_from_split(self, radius):
        return (4.75 + 0.95 * radius) / (self.split_vel_zero_frame / 20) * 2

    def cal_split_vel_init_from_thorns(self, radius):
        return (13.0 - radius) / (self.split_vel_zero_frame / 20) * 2

    def move(self, given_acc=None, given_acc_center=None, duration=0.05):
        """
        Overview:
            Realize the movement of the ball, pass in the direction and time parameters
        """
        if given_acc is not None:
            if given_acc.length != 0:
                given_acc = given_acc if given_acc.length() < 1 else given_acc.normalize()
                self.acc_given = given_acc * self.acc_weight
        else:
            given_acc = self.acc_given / self.acc_weight
        if given_acc_center is not None:
            given_acc_center = given_acc_center / self.radius
            if given_acc_center.length() != 0 and given_acc_center.length() > 1:
                given_acc_center = given_acc_center.normalize()
            self.acc_given_center = given_acc_center * self.center_acc_weight
        else:
            given_acc_center = Vector2(0, 0)
            self.acc_given_center = Vector2(0, 0)
        self.acc_given_total = self.acc_given + self.acc_given_center
        vel_max_ratio_given = given_acc.length()
        vel_max_ratio_center = given_acc_center.length()
        vel_max_ratio = max(vel_max_ratio_given, vel_max_ratio_center)
        if self.split_frame < self.split_vel_zero_frame:
            self.vel_split -= self.vel_split_piece
            self.split_frame += 1
        else:
            self.vel_split = Vector2(0, 0)
        self.vel_given = self.vel_given + self.acc_given_total * duration
        self.vel_max_ball = self.cal_vel_max(self.radius, ratio=vel_max_ratio)
        self.vel_given = format_vector(self.vel_given, self.vel_max_ball)
        self.vel = self.vel_given + self.vel_split
        self.position = self.position + self.vel * duration
        self.update_direction()
        self.frame_since_last_split += 1
        self.check_border()

    def eat(self, ball, clone_num=None):
        """
        Parameters:
            clone_num <int>: The total number of balls for the current player
        """
        if isinstance(ball, SporeBall) or isinstance(ball, FoodBall) or isinstance(ball, CloneBall):
            self.set_score(add_score(self.score, ball.score))
        elif isinstance(ball, ThornsBall):
            assert clone_num is not None
            self.set_score(add_score(self.score, ball.score))
            if clone_num < self.part_num_max:
                split_num = min(self.part_num_max - clone_num, self.on_thorns_part_num)
                return self.on_thorns(split_num=split_num)
        else:
            logging.debug('CloneBall can not eat {}'.format(type(ball)))
        self.check_border()
        return True

    def on_thorns(self, split_num) -> list:
        """
        Overview:
            Split after encountering thorns, calculate the score, position, speed, acceleration of each ball after splitting
        Parameters:
            split_num <int>: Number of splits added
        Returns:
            Return a list that contains the newly added balls after the split, the distribution of the split balls is a circle and the center of the circle has a ball
        """
        around_score = min(self.score / (split_num + 1), self.on_thorns_part_score_max)
        around_radius = self.score_to_radius(around_score)
        middle_score = self.score - around_score * split_num
        self.set_score(middle_score)
        around_positions = []
        around_split_directions = []
        for i in range(split_num):
            angle = 2 * math.pi * (i + 1) / split_num
            unit_x = math.cos(angle)
            unit_y = math.sin(angle)
            split_direction = Vector2(unit_x, unit_y)
            around_position = self.position + Vector2((self.radius + around_radius) * unit_x, (self.radius + around_radius) * unit_y)
            around_positions.append(around_position)
            around_split_directions.append(split_direction)
        balls = []
        for p, s in zip(around_positions, around_split_directions):
            ball_id = uuid.uuid1() if self.sequence_generator is None else self.sequence_generator.get()
            around_ball = CloneBall(ball_id=ball_id, position=p, score=around_score, border=self.border, team_id=self.team_id, player_id=self.player_id, vel_given=copy.deepcopy(self.vel_given), acc_given=copy.deepcopy(self.acc_given), from_split=False, from_thorns=True, split_direction=s, spore_settings=self.spore_settings, sequence_generator=self.sequence_generator, **self.cfg)
            balls.append(around_ball)
        return balls

    def eject(self, direction=None) -> list:
        """
        Overview:
            When spit out spores, the spores spit out must be in the moving direction of the ball, and the position is tangent to the original ball after spitting out
        Returns:
            Return a list containing the spores spit out
        """
        if direction is None or direction.length() == 0:
            direction = self.direction
        else:
            direction = direction.normalize()
        if self.score >= self.eject_score_min:
            spore_score = self.spore_settings.score_init
            self.set_score(self.score - spore_score)
            spore_radius = self.score_to_radius(spore_score)
            position = self.position + direction * (self.radius + spore_radius)
            return SporeBall(ball_id=uuid.uuid1(), position=position, border=self.border, score=spore_score, direction=direction, owner=self.player_id, **self.spore_settings)
        else:
            return False

    def split(self, clone_num, direction=None) -> list:
        """
        Overview:
            Active splitting, the two balls produced by splitting have the same volume, and their positions are tangent to the forward direction
        Parameters:
            clone_num <int>: The total number of balls for the current player
        Returns:
            The return value is the new ball after the split
        """
        if direction is None or direction.length() == 0:
            direction = self.direction
        else:
            direction = direction.normalize()
        if self.score >= self.split_score_min and clone_num < self.part_num_max:
            split_score = self.score / 2
            self.set_score(split_score)
            clone_num += 1
            position = self.position + direction * (self.radius * 2)
            ball_id = uuid.uuid1() if self.sequence_generator is None else self.sequence_generator.get()
            return CloneBall(ball_id=ball_id, position=position, score=self.score, border=self.border, team_id=self.team_id, player_id=self.player_id, vel_given=copy.deepcopy(self.vel_given), acc_given=copy.deepcopy(self.acc_given), from_split=True, from_thorns=False, split_direction=direction, spore_settings=self.spore_settings, sequence_generator=self.sequence_generator, **self.cfg)
        else:
            return False

    def rigid_collision(self, ball):
        """
        Overview:
            When two balls collide, We need to determine whether the two balls belong to the same player
            A. If not, do nothing until one party is eaten at the end
            B. If the two balls are the same owner, judge whether the age of the two is full or not meet the fusion condition, if they are satisfied, do nothing.
            C. If the two balls are the same owner, judge whether the age of the two is full or not meet the fusion condition, Then the two balls will collide with rigid bodies
            This function completes the C part: the rigid body collision part, the logic is as follows:
             1. To determine the degree of fusion of the two balls, use [the radius of both] and subtract [the distance between the two] as the magnitude of the force
             2. Calculate the coefficient according to the weight, the larger the weight, the smaller the coefficient will be
             3. Correct the position of the two according to the coefficient and force
        Parameters:
            ball <CloneBall>: another ball
        Returns:
            state <bool>: the operation is successful or not
        """
        if ball.ball_id == self.ball_id:
            return True
        assert isinstance(ball, CloneBall), 'ball is not CloneBall but {}'.format(type(ball))
        assert self.player_id == ball.player_id
        assert self.frame_since_last_split < self.recombine_frame or ball.frame_since_last_split < ball.recombine_frame
        p = ball.position - self.position
        d = p.length()
        if self.radius + ball.radius > d:
            f = min(self.radius + ball.radius - d, (self.radius + ball.radius - d) / (d + 1e-08))
            self.position = self.position - f * p * (ball.score / (self.score + ball.score))
            ball.position = ball.position + f * p * (self.score / (self.score + ball.score))
        else:
            print('WARNINGS: self.radius ({}) + ball.radius ({}) <= d ({})'.format(self.radius, ball.radius, d))
        self.check_border()
        ball.check_border()
        return True

    def judge_rigid(self, ball):
        """
        Overview:
            Determine whether two balls will collide with a rigid body
        Parameters:
            ball <CloneBall>: another ball
        Returns:
            <bool>: collide or not
        """
        return self.frame_since_last_split < self.recombine_frame or ball.frame_since_last_split < ball.recombine_frame

    def score_decay(self):
        """
        Overview: 
            Control the score of the ball to decay over time
        """
        if self.score > self.score_decay_min:
            self.set_score(self.score * (1 - self.score_decay_rate_per_frame * math.sqrt(self.radius)))
        return True

    def flush_frame_since_last_split(self):
        self.frame_since_last_split = 0
        return True

    def __repr__(self) -> str:
        return '{}, vel_given={}, acc_given={}, frame_since_last_split={:.3f}, player_id={}, direction={}, team_id={}'.format(super().__repr__(), self.vel_given, self.acc_given, self.frame_since_last_split, self.player_id, self.direction, self.team_id)

    def save(self):
        return [self.position.x, self.position.y, self.radius, self.direction.x, self.direction.y, self.player_id, self.team_id]

class ThornsBall(BaseBall):
    """
    Overview:
        - characteristic:
        * Can't move actively
        * Can eat spores. When eating spores, it will inherit the momentum of the spores and move a certain distance.
        * Can only be eaten by balls heavier than him. After eating, it will split the host into multiple smaller units.
        * Nothing happens when a ball lighter than him passes by
    """

    @staticmethod
    def default_config():
        cfg = BaseBall.default_config()
        cfg.update(dict(score_min=3, score_max=5, eat_spore_vel_init=4, eat_spore_vel_zero_frame=10))
        return EasyDict(cfg)

    def __init__(self, ball_id, position, score, border, **kwargs):
        kwargs = EasyDict(kwargs)
        cfg = ThornsBall.default_config()
        cfg = deep_merge_dicts(cfg, kwargs)
        super(ThornsBall, self).__init__(ball_id, position, score=score, border=border, **cfg)
        self.score_min = cfg.score_min
        self.score_max = cfg.score_max
        self.eat_spore_vel_init = cfg.eat_spore_vel_init
        self.eat_spore_vel_zero_frame = cfg.eat_spore_vel_zero_frame
        self.move_frame = 0
        self.vel = Vector2(0, 0)
        self.vel_piece = Vector2(0, 0)
        self.moving = False
        self.check_border()

    def move(self, direction=None, duration=0.05, **kwargs):
        assert duration > 0
        if self.moving:
            self.position = self.position + self.vel * duration
            self.move_frame += 1
            if self.move_frame < self.eat_spore_vel_zero_frame:
                self.vel = self.vel - self.vel_piece
            else:
                self.vel = Vector2(0, 0)
                self.vel_piece = Vector2(0, 0)
                self.moving = False
        self.check_border()
        return True

    def eat(self, ball):
        if isinstance(ball, SporeBall):
            self.set_score(add_score(self.score, ball.score))
            if ball.vel.length() > 0:
                self.vel = self.eat_spore_vel_init * ball.vel.normalize()
                self.vel_piece = self.vel / self.eat_spore_vel_zero_frame
                self.move_time = 0
                self.moving = True
        else:
            logging.debug('ThornsBall can not eat {}'.format(type(ball)))
        return True

    def set_score(self, score: float) -> None:
        self.score = score
        if self.score > self.score_max:
            self.score = self.score_max
        elif self.score < self.score_min:
            self.score = self.score_min
        self.radius = self.score_to_radius(self.score)

    def save(self):
        return [self.position.x, self.position.y, self.radius]

class Model(nn.Module):

    def __init__(self, cfg={}, use_value_network=False):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.model_cfg = self.whole_cfg.model
        self.use_value_network = use_value_network
        self.encoder = Encoder(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.temperature = self.whole_cfg.agent.get('temperature', 1)

    def compute_action(self, obs):
        action_mask = obs.pop('action_mask', None)
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding, temperature=self.temperature)
        if action_mask is not None:
            logit.masked_fill_(mask=action_mask, value=-1000000000.0)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        return {'action': action, 'logit': logit}

class Model(nn.Module):

    def __init__(self, cfg={}, **kwargs):
        super(Model, self).__init__()
        self.whole_cfg = deep_merge_dicts(default_config, cfg)
        self.encoder = Encoder(self.whole_cfg)
        self.policy_head = PolicyHead(self.whole_cfg)
        self.value_head = ValueHead(self.whole_cfg)
        self.only_update_value = False
        self.ortho_init = self.whole_cfg.model.get('ortho_init', True)
        self.player_num = self.whole_cfg.env.player_num_per_team
        self.team_num = self.whole_cfg.env.team_num

    def forward(self, obs, temperature=0):
        obs = flatten_data(obs, start_dim=0, end_dim=1)
        embedding = self.encoder(obs)
        logit = self.policy_head(embedding)
        if temperature == 0:
            action = logit.argmax(dim=-1)
        else:
            logit = logit.div(temperature)
            dist = torch.distributions.Categorical(logits=logit)
            action = dist.sample()
        return {'action': action, 'logit': logit}

    def compute_value(self, obs):
        obs = flatten_data(obs, start_dim=0, end_dim=1)
        embedding = self.encoder(obs)
        batch_size = embedding.shape[0] // self.team_num // self.player_num
        team_embedding = embedding.reshape(batch_size * self.team_num, self.player_num, -1)
        team_embedding = self.transform_ctde(team_embedding, device=team_embedding.device)
        value = self.value_head(team_embedding)
        return {'value': value.reshape(-1)}

    def compute_logp_action(self, obs, **kwargs):
        obs = flatten_data(obs, start_dim=0, end_dim=1)
        embedding = self.encoder(obs)
        batch_size = embedding.shape[0] // self.team_num // self.player_num
        logit = self.policy_head(embedding)
        dist = torch.distributions.Categorical(logits=logit)
        action = dist.sample()
        action_log_probs = dist.log_prob(action)
        log_action_probs = action_log_probs
        team_embedding = embedding.reshape(batch_size * self.team_num, self.player_num, -1)
        team_embedding = self.transform_ctde(team_embedding, device=team_embedding.device)
        value = self.value_head(team_embedding)
        return {'action': action, 'action_logp': log_action_probs, 'logit': logit, 'value': value.reshape(-1)}

    def rl_train(self, inputs: dict, **kwargs) -> Dict[str, Any]:
        """
        Overview:
            Forward and backward function of learn mode.
        Arguments:
            - inputs (:obj:`dict`): Dict type data
        ArgumentsKeys:
            - obs shape     :math:`(T+1, B)`, where T is timestep, B is batch size
            - action_logp: behaviour logits, :math:`(T, B,action_size)`
            - action: behaviour actions, :math:`(T, B)`
            - reward: shape math:`(T, B)`
            - done:shape math:`(T, B)`
        Returns:
            - metric_dict (:obj:`Dict[str, Any]`):
              Including current total_loss, policy_gradient_loss, critic_loss and entropy_loss
        """
        obs = inputs['obs']
        obs = flatten_data(obs, start_dim=0, end_dim=1)
        embedding = self.encoder(obs)
        batch_size = embedding.shape[0] // self.player_num
        logits = self.policy_head(embedding)
        critic_input = embedding.reshape(batch_size, self.player_num, -1)
        critic_input = self.transform_ctde(critic_input, device=critic_input.device)
        if self.only_update_value:
            critic_input = detach_grad(critic_input)
        values = self.value_head(critic_input)
        outputs = {'value': values.squeeze(-1).reshape(-1), 'logit': logits, 'action': inputs['action'].reshape(-1), 'action_logp': inputs['action_logp'].reshape(-1), 'old_value': inputs['old_value'].reshape(-1), 'advantage': inputs['advantage'].reshape(-1), 'return': inputs['return'].reshape(-1)}
        return outputs

    def transform_ctde(self, array, device):
        ret = []
        for i in range(self.player_num):
            index = [i for i in range(self.player_num)]
            index.pop(i)
            other_array = torch.index_select(array, dim=1, index=torch.LongTensor(index).to(device))
            self_array = array[:, i, :].unsqueeze(dim=1)
            ret.append(torch.cat((self_array, other_array), dim=1).flatten(start_dim=1, end_dim=2).unsqueeze(1))
        ret = torch.cat(ret, dim=1)
        return ret

