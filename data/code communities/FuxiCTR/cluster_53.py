# Cluster 53

class SerialMaskNet(nn.Module):

    def __init__(self, input_dim, output_dim=None, output_activation=None, hidden_units=[], hidden_activations='ReLU', reduction_ratio=1, dropout_rates=0, layer_norm=True):
        super(SerialMaskNet, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(input_dim, self.hidden_units[idx], self.hidden_units[idx + 1], hidden_activations[idx], reduction_ratio, dropout_rates[idx], layer_norm))
        fc_layers = []
        if output_dim is not None:
            fc_layers.append(nn.Linear(self.hidden_units[-1], output_dim))
        if output_activation is not None:
            fc_layers.append(get_activation(output_activation))
        self.fc = None
        if len(fc_layers) > 0:
            self.fc = nn.Sequential(*fc_layers)

    def forward(self, V_emb, V_hidden):
        v_out = V_hidden
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](V_emb, v_out)
        if self.fc is not None:
            v_out = self.fc(v_out)
        return v_out

def get_activation(activation, hidden_units=None):
    if isinstance(activation, str):
        if activation.lower() in ['prelu', 'dice']:
            assert type(hidden_units) == int
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'softmax':
            return nn.Softmax(dim=-1)
        elif activation.lower() == 'prelu':
            return nn.PReLU(hidden_units, init=0.1)
        elif activation.lower() == 'dice':
            from fuxictr.pytorch.layers.activations import Dice
            return Dice(hidden_units)
        else:
            return getattr(nn, activation)()
    elif isinstance(activation, list):
        if hidden_units is not None:
            assert len(activation) == len(hidden_units)
            return [get_activation(act, units) for act, units in zip(activation, hidden_units)]
        else:
            return [get_activation(act) for act in activation]
    return activation

class MaskBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation='ReLU', reduction_ratio=1, dropout_rate=0, layer_norm=True):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, int(hidden_dim * reduction_ratio)), nn.ReLU(), nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, V_emb, V_hidden):
        V_mask = self.mask_layer(V_emb)
        v_out = self.hidden_layer(V_mask * V_hidden)
        return v_out

class FinalBlock(nn.Module):

    def __init__(self, input_dim, hidden_units=[], hidden_activations=None, dropout_rates=[], batch_norm=True, residual_type='sum'):
        super(FinalBlock, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FactorizedInteraction(hidden_units[idx], hidden_units[idx + 1], residual_type=residual_type))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))

    def forward(self, X):
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return X_i

class CCPM_ConvLayer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """

    def __init__(self, num_fields, channels=[3], kernel_heights=[3], activation='Tanh'):
        super(CCPM_ConvLayer, self).__init__()
        if not isinstance(kernel_heights, list):
            kernel_heights = [kernel_heights] * len(channels)
        elif len(kernel_heights) != len(channels):
            raise ValueError('channels={} and kernel_heights={} should have the same length.'.format(channels, kernel_heights))
        module_list = []
        self.channels = [1] + channels
        layers = len(kernel_heights)
        for i in range(1, len(self.channels)):
            in_channels = self.channels[i - 1]
            out_channels = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            module_list.append(nn.ZeroPad2d((0, 0, kernel_height - 1, kernel_height - 1)))
            module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_height, 1)))
            if i < layers:
                k = max(3, int((1 - pow(float(i) / layers, layers - i)) * num_fields))
            else:
                k = 3
            module_list.append(KMaxPooling(k, dim=2))
            module_list.append(get_activation(activation))
        self.conv_layer = nn.Sequential(*module_list)

    def forward(self, X):
        return self.conv_layer(X)

class APG_MLP(nn.Module):

    def __init__(self, input_dim, hidden_units=[], hidden_activations='ReLU', output_dim=None, output_activation=None, dropout_rates=0.0, batch_norm=False, bn_only_once=False, use_bias=True, hypernet_config={}, condition_dim=None, condition_mode='self-wise', rank_k=None, overparam_p=None, generate_bias=True):
        super(APG_MLP, self).__init__()
        self.hidden_layers = len(hidden_units)
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * self.hidden_layers
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * self.hidden_layers
        hidden_activations = get_activation(hidden_activations, hidden_units)
        if not isinstance(rank_k, list):
            rank_k = [rank_k] * self.hidden_layers
        if not isinstance(overparam_p, list):
            overparam_p = [overparam_p] * self.hidden_layers
        assert self.hidden_layers == len(dropout_rates) == len(hidden_activations) == len(rank_k) == len(overparam_p)
        hidden_units = [input_dim] + hidden_units
        self.dense_layers = nn.ModuleDict()
        if batch_norm and bn_only_once:
            self.dense_layers['bn_0'] = nn.BatchNorm1d(input_dim)
        self.condition_mode = condition_mode
        assert condition_mode in ['self-wise', 'group-wise', 'mix-wise'], 'Invalid condition_mode={}'.format(condition_mode)
        for idx in range(self.hidden_layers):
            if self.condition_mode == 'self-wise':
                condition_dim = hidden_units[idx]
            self.dense_layers['linear_{}'.format(idx + 1)] = APG_Linear(hidden_units[idx], hidden_units[idx + 1], condition_dim, bias=use_bias, rank_k=rank_k[idx], overparam_p=overparam_p[idx], generate_bias=generate_bias, hypernet_config=hypernet_config)
            if batch_norm and (not bn_only_once):
                self.dense_layers['bn_{}'.format(idx + 1)] = nn.BatchNorm1d(hidden_units[idx + 1])
            if hidden_activations[idx]:
                self.dense_layers['act_{}'.format(idx + 1)] = hidden_activations[idx]
            if dropout_rates[idx] > 0:
                self.dense_layers['drop_{}'.format(idx + 1)] = nn.Dropout(p=dropout_rates[idx])
        if output_dim is not None:
            self.dense_layers['out_proj'] = nn.Linear(hidden_units[-1], output_dim, bias=use_bias)
        if output_activation is not None:
            self.dense_layers['out_act'] = get_activation(output_activation)

    def forward(self, x, condition_z=None):
        if 'bn_0' in self.dense_layers:
            x = self.dense_layers['bn_0'](x)
        for idx in range(self.hidden_layers):
            if self.condition_mode == 'self-wise':
                x = self.dense_layers['linear_{}'.format(idx + 1)](x, x)
            else:
                x = self.dense_layers['linear_{}'.format(idx + 1)](x, condition_z)
            if 'bn_{}'.format(idx + 1) in self.dense_layers:
                x = self.dense_layers['bn_{}'.format(idx + 1)](x)
            if 'act_{}'.format(idx + 1) in self.dense_layers:
                x = self.dense_layers['act_{}'.format(idx + 1)](x)
            if 'drop_{}'.format(idx + 1) in self.dense_layers:
                x = self.dense_layers['drop_{}'.format(idx + 1)](x)
        if 'out_proj' in self.dense_layers:
            x = self.dense_layers['out_proj'](x)
        if 'out_act' in self.dense_layers:
            x = self.dense_layers['out_act'](x)
        return x

class ResidualBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_activation='ReLU', dropout_rate=0, use_residual=True, batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.activation_layer = get_activation(hidden_activation)
        self.layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), self.activation_layer, nn.Linear(hidden_dim, input_dim))
        self.use_residual = use_residual
        self.batch_norm = nn.BatchNorm1d(input_dim) if batch_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, X):
        X_out = self.layer(X)
        if self.use_residual:
            X_out = X_out + X
        if self.batch_norm is not None:
            X_out = self.batch_norm(X_out)
        output = self.activation_layer(X_out)
        if self.dropout is not None:
            output = self.dropout(output)
        return output

class PPNet_MLP(nn.Module):

    def __init__(self, input_dim, output_dim=1, gate_input_dim=64, gate_hidden_dim=None, hidden_units=[], hidden_activations='ReLU', dropout_rates=0.0, batch_norm=False, use_bias=True):
        super(PPNet_MLP, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        self.gate_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            layers = [nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias)]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx] is not None:
                layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                layers.append(nn.Dropout(p=dropout_rates[idx]))
            self.mlp_layers.append(nn.Sequential(*layers))
            self.gate_layers.append(GateNU(gate_input_dim, gate_hidden_dim, output_dim=hidden_units[idx + 1]))
        self.mlp_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))

    def forward(self, feature_emb, gate_emb):
        gate_input = torch.cat([feature_emb.detach(), gate_emb], dim=-1)
        h = feature_emb
        for i in range(len(self.gate_layers)):
            h = self.mlp_layers[i](h)
            g = self.gate_layers[i](gate_input)
            h = h * g
        out = self.mlp_layers[-1](h)
        return out

class GateNU(nn.Module):

    def __init__(self, input_dim, hidden_dim=None, output_dim=None, hidden_activation='ReLU', dropout_rate=0.0):
        super(GateNU, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        layers = [nn.Linear(input_dim, hidden_dim)]
        layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())
        self.gate = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.gate(inputs) * 2

class CGC_Layer(nn.Module):

    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations, net_dropout, batch_norm):
        super(CGC_Layer, self).__init__()
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = nn.ModuleList([MLP_Block(input_dim=input_dim, hidden_units=expert_hidden_units, hidden_activations=hidden_activations, output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm) for _ in range(self.num_shared_experts)])
        self.specific_experts = nn.ModuleList([nn.ModuleList([MLP_Block(input_dim=input_dim, hidden_units=expert_hidden_units, hidden_activations=hidden_activations, output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm) for _ in range(self.num_specific_experts)]) for _ in range(num_tasks)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim, output_dim=num_specific_experts + num_shared_experts if i < num_tasks else num_shared_experts, hidden_units=gate_hidden_units, hidden_activations=hidden_activations, output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm) for i in range(self.num_tasks + 1)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x, require_gate=False):
        """
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = []
        shared_expert_outputs = []
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        cgc_outputs = []
        gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                gate_input = torch.stack(specific_expert_outputs[i] + shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[i]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
            else:
                gate_input = torch.stack(shared_expert_outputs, dim=1)
                gate = self.gate_activation(self.gate[i](x[-1]))
                gates.append(gate.mean(0))
                cgc_output = torch.sum(gate.unsqueeze(-1) * gate_input, dim=1)
                cgc_outputs.append(cgc_output)
        if require_gate:
            return (cgc_outputs, gates)
        else:
            return cgc_outputs

class MMoE_Layer(nn.Module):

    def __init__(self, num_experts, num_tasks, input_dim, expert_hidden_units, gate_hidden_units, hidden_activations, net_dropout, batch_norm):
        super(MMoE_Layer, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.experts = nn.ModuleList([MLP_Block(input_dim=input_dim, hidden_units=expert_hidden_units, hidden_activations=hidden_activations, output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = nn.ModuleList([MLP_Block(input_dim=input_dim, hidden_units=gate_hidden_units, output_dim=num_experts, hidden_activations=hidden_activations, output_activation=None, dropout_rates=net_dropout, batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = get_activation('softmax')

    def forward(self, x):
        experts_output = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = self.gate_activation(gate_output)
            mmoe_output.append(torch.sum(torch.multiply(gate_output.unsqueeze(-1), experts_output), dim=1))
        return mmoe_output

class FGCNN_Layer(nn.Module):
    """
    Input X: tensor of shape (batch_size, 1, num_fields, embedding_dim)
    """

    def __init__(self, num_fields, embedding_dim, channels=[3], kernel_heights=[3], pooling_sizes=[2], recombined_channels=[2], activation='Tanh', batch_norm=True):
        super(FGCNN_Layer, self).__init__()
        self.embedding_dim = embedding_dim
        conv_list = []
        recombine_list = []
        self.channels = [1] + channels
        input_height = num_fields
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recombined_channel = recombined_channels[i - 1]
            conv_layer = [nn.Conv2d(in_channel, out_channel, kernel_size=(kernel_height, 1), padding=(int((kernel_height - 1) / 2), 0))] + ([nn.BatchNorm2d(out_channel)] if batch_norm else []) + [get_activation(activation), nn.MaxPool2d((pooling_size, 1), padding=(input_height % pooling_size, 0))]
            conv_list.append(nn.Sequential(*conv_layer))
            input_height = int(np.ceil(input_height / pooling_size))
            input_dim = input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            recombine_layer = nn.Sequential(nn.Linear(input_dim, output_dim), get_activation(activation))
            recombine_list.append(recombine_layer)
        self.conv_layers = nn.ModuleList(conv_list)
        self.recombine_layers = nn.ModuleList(recombine_list)

    def forward(self, X):
        conv_out = X
        new_feature_list = []
        for i in range(len(self.channels) - 1):
            conv_out = self.conv_layers[i](conv_out)
            flatten_out = torch.flatten(conv_out, start_dim=1)
            recombine_out = self.recombine_layers[i](flatten_out)
            new_feature_list.append(recombine_out.reshape(X.size(0), -1, self.embedding_dim))
        new_feature_emb = torch.cat(new_feature_list, dim=1)
        return new_feature_emb

class FeatureEmbeddingDict(Layer):

    def __init__(self, feature_map, embedding_dim, embedding_initializer='random_normal(stddev=1e-4)', embedding_regularizer=None, required_feature_columns=None, not_required_feature_columns=None, use_pretrain=True, use_sharing=True, name_prefix='emb_'):
        super(FeatureEmbeddingDict, self).__init__()
        self._feature_map = feature_map
        self.required_feature_columns = required_feature_columns
        self.not_required_feature_columns = not_required_feature_columns
        self.use_pretrain = use_pretrain
        self.embedding_initializer = embedding_initializer
        self.embedding_layers = OrderedDict()
        self.feature_encoders = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if self.is_required(feature):
                if not (use_pretrain and use_sharing) and embedding_dim == 1:
                    feat_emb_dim = 1
                    if feature_spec['type'] == 'sequence':
                        self.feature_encoders[feature] = layers.MaskedSumPooling()
                else:
                    feat_emb_dim = feature_spec.get('embedding_dim', embedding_dim)
                    if feature_spec.get('feature_encoder', None):
                        self.feature_encoders[feature] = self.get_feature_encoder(feature_spec['feature_encoder'])
                if use_sharing and feature_spec.get('share_embedding') in self.embedding_layers:
                    self.embedding_layers[feature] = self.embedding_layers[feature_spec['share_embedding']]
                    continue
                if feature_spec['type'] == 'numeric':
                    self.embedding_layers[feature] = tf.keras.layers.Dense(feat_emb_dim, use_bias=False)
                elif feature_spec['type'] == 'categorical':
                    padding_idx = feature_spec.get('padding_idx', None)
                    embedding_matrix = Embedding(feature_spec['vocab_size'], feat_emb_dim, embeddings_initializer=get_initializer(embedding_initializer), embeddings_regularizer=get_regularizer(embedding_regularizer), mask_zero=True if padding_idx == 0 else False, input_length=1, name=name_prefix + feature)
                    if use_pretrain and 'pretrained_emb' in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, feature_map, feature, freeze=feature_spec['freeze_emb'], padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix
                elif feature_spec['type'] == 'sequence':
                    padding_idx = feature_spec.get('padding_idx', None)
                    embedding_matrix = Embedding(feature_spec['vocab_size'], feat_emb_dim, embeddings_initializer=get_initializer(embedding_initializer), embeddings_regularizer=get_regularizer(embedding_regularizer), mask_zero=True if padding_idx == 0 else False, input_length=feature_spec['max_len'], name=name_prefix + feature)
                    if use_pretrain and 'pretrained_emb' in feature_spec:
                        embedding_matrix = self.load_pretrained_embedding(embedding_matrix, feature_map, feature, freeze=feature_spec['freeze_emb'], padding_idx=padding_idx)
                    self.embedding_layers[feature] = embedding_matrix

    def get_feature_encoder(self, encoder):
        try:
            if type(encoder) == list:
                encoder_list = []
                for enc in encoder:
                    encoder_list.append(eval(enc))
                encoder_layer = tf.keras.Sequential(*encoder_list)
            else:
                encoder_layer = eval(encoder)
            return encoder_layer
        except:
            raise ValueError('feature_encoder={} is not supported.'.format(encoder))

    def is_required(self, feature):
        """ Check whether feature is required for embedding """
        feature_spec = self._feature_map.features[feature]
        if feature_spec['type'] == 'meta':
            return False
        elif self.required_feature_columns and feature not in self.required_feature_columns:
            return False
        elif self.not_required_feature_columns and feature in self.not_required_feature_columns:
            return False
        else:
            return True

    def get_pretrained_embedding(self, pretrained_path, feature_name):
        with h5py.File(pretrained_path, 'r') as hf:
            embeddings = hf[feature_name][:]
        return embeddings

    def load_pretrained_embedding(self, embedding_matrix, feature_map, feature_name, freeze=False, padding_idx=None):
        pretrained_path = os.path.join(feature_map.data_dir, feature_map.features[feature_name]['pretrained_emb'])
        embeddings = self.get_pretrained_embedding(pretrained_path, feature_name)
        if padding_idx is not None:
            embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
        assert embeddings.shape[-1] == embedding_matrix.embedding_dim, "{}'s embedding_dim is not correctly set to match its pretrained_emb shape".format(feature_name)
        embedding_matrix.set_weights([embeddings])
        if freeze:
            embedding_matrix.trainable = False
        return embedding_matrix

    def dict2tensor(self, embedding_dict, feature_list=[], feature_source=[], feature_type=[], flatten_emb=False):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_list = []
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec['source'] not in feature_source:
                continue
            if feature_type and feature_spec['type'] not in feature_type:
                continue
            if feature_list and feature not in feature_list:
                continue
            if feature in embedding_dict:
                feature_emb_list.append(embedding_dict[feature])
        if flatten_emb:
            feature_emb = tf.squeeze(tf.concat(feature_emb_list, axis=-1), axis=1)
        else:
            feature_emb = tf.concat(feature_emb_list, axis=1)
        return feature_emb

    def call(self, inputs, feature_source=[], feature_type=[]):
        if type(feature_source) != list:
            feature_source = [feature_source]
        if type(feature_type) != list:
            feature_type = [feature_type]
        feature_emb_dict = OrderedDict()
        for feature, feature_spec in self._feature_map.features.items():
            if feature_source and feature_spec['source'] not in feature_source:
                continue
            if feature_type and feature_spec['type'] not in feature_type:
                continue
            if feature in self.embedding_layers:
                if feature_spec['type'] == 'numeric':
                    inp = tf.reshape(inputs[feature], (-1, 1))
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec['type'] == 'categorical':
                    inp = inputs[feature]
                    embeddings = self.embedding_layers[feature](inp)
                elif feature_spec['type'] == 'sequence':
                    inp = inputs[feature]
                    embeddings = self.embedding_layers[feature](inp)
                else:
                    raise NotImplementedError
                if feature in self.feature_encoders:
                    embeddings = self.feature_encoders[feature](embeddings)
                feature_emb_dict[feature] = embeddings
        return feature_emb_dict

def get_initializer(initializer, seed=20222023):
    if isinstance(initializer, str):
        try:
            if '(' in initializer:
                return eval(initializer.rstrip(')') + ', seed={})'.format(seed))
            else:
                return eval(initializer)(seed=seed)
        except:
            raise ValueError('initializer={} not supported.'.format(initializer))
    return initializer

def get_regularizer(reg):
    if type(reg) in [int, float]:
        return l2(reg)
    elif isinstance(reg, str):
        if '(' in reg:
            try:
                return eval(reg)
            except:
                raise ValueError('reg={} is not supported.'.format(reg))
    return reg

class Linear(Layer):

    def __init__(self, output_dim, use_bias=True, initializer='glorot_normal', regularizer=None):
        super(Linear, self).__init__()
        self.linear = Dense(output_dim, use_bias=use_bias, kernel_initializer=get_initializer(initializer), kernel_regularizer=get_regularizer(regularizer), bias_regularizer=get_regularizer(regularizer))

    def call(self, inputs):
        return self.linear(inputs)

class MLP_Block(Layer):

    def __init__(self, input_dim, hidden_units=[], hidden_activations='ReLU', output_dim=None, output_activation=None, dropout_rates=0.0, batch_norm=False, layer_norm=False, norm_before_activation=True, use_bias=True, initializer='glorot_normal', regularizer=None):
        super(MLP_Block, self).__init__()
        self.mlp = tf.keras.Sequential()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = [get_activation(x) for x in hidden_activations]
        hidden_units = [input_dim] + hidden_units
        for idx in range(len(hidden_units) - 1):
            self.mlp.add(Dense(hidden_units[idx + 1], use_bias=use_bias, kernel_initializer=get_initializer(initializer), kernel_regularizer=get_regularizer(regularizer), bias_regularizer=get_regularizer(regularizer)))
            if norm_before_activation:
                if batch_norm:
                    self.mlp.add(BatchNormalization(hidden_units[idx + 1]))
                elif layer_norm:
                    self.mlp.add(LayerNormalization(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                self.mlp.add(hidden_activations[idx])
            if not norm_before_activation:
                if batch_norm:
                    self.mlp.add(BatchNormalization(hidden_units[idx + 1]))
                elif layer_norm:
                    self.mlp.add(LayerNormalization(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.mlp.add(Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            self.mlp.add(Dense(output_dim, use_bias=use_bias, kernel_initializer=get_initializer(initializer), kernel_regularizer=get_regularizer(regularizer), bias_regularizer=get_regularizer(regularizer)))
        if output_activation is not None:
            self.mlp.add(get_activation(output_activation))

    def call(self, inputs, training=None):
        return self.mlp(inputs, training=training)

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            return tf.keras.layers.Activation('relu')
        elif activation.lower() == 'sigmoid':
            return tf.keras.layers.Activation('sigmoid')
        elif activation.lower() == 'tanh':
            return tf.keras.layers.Activation('tanh')
        elif activation.lower() == 'softmax':
            return tf.keras.layers.Softmax()
        else:
            return getattr(tf.keras.layers, activation)()
    else:
        return activation

class MLP_Block(nn.Module):

    def __init__(self, input_dim, hidden_units=[], hidden_activations='ReLU', output_dim=None, output_activation=None, dropout_rates=0.0, batch_norm=False, bn_only_once=False, use_bias=True):
        super(MLP_Block, self).__init__()
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        hidden_activations = get_activation(hidden_activations, hidden_units)
        hidden_units = [input_dim] + hidden_units
        if batch_norm and bn_only_once:
            dense_layers.append(nn.BatchNorm1d(input_dim))
        for idx in range(len(hidden_units) - 1):
            dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
            if batch_norm and (not bn_only_once):
                dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if hidden_activations[idx]:
                dense_layers.append(hidden_activations[idx])
            if dropout_rates[idx] > 0:
                dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
        if output_activation is not None:
            dense_layers.append(get_activation(output_activation))
        self.mlp = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.mlp(inputs)

