# Cluster 16

class LSTMForwardWrapper(object):
    """
    Overview:
        abstract class used to wrap the LSTM forward method
    Interface:
        _before_forward, _after_forward
    """

    def _before_forward(self, inputs, prev_state):
        """
        Overview:
            preprocess the inputs and previous states
        Arguments:
            - inputs (:obj:`tensor`): input vector of cell, tensor of size [seq_len, batch_size, input_size]
            - prev_state (:obj:`tensor` or :obj:`list`):
                None or tensor of size [num_directions*num_layers, batch_size, hidden_size], if None then prv_state
                will be initialized to all zeros.
        Returns:
            - prev_state (:obj:`tensor`): batch previous state in lstm
        """
        assert hasattr(self, 'num_layers')
        assert hasattr(self, 'hidden_size')
        seq_len, batch_size = inputs.shape[:2]
        if prev_state is None:
            num_directions = 1
            zeros = torch.zeros(num_directions * self.num_layers, batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            prev_state = (zeros, zeros)
        elif is_sequence(prev_state):
            if len(prev_state) == 2 and isinstance(prev_state[0], torch.Tensor):
                pass
            else:
                if len(prev_state) != batch_size:
                    raise RuntimeError('prev_state number is not equal to batch_size: {}/{}'.format(len(prev_state), batch_size))
                num_directions = 1
                zeros = torch.zeros(num_directions * self.num_layers, 1, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
                state = []
                for prev in prev_state:
                    if prev is None:
                        state.append([zeros, zeros])
                    else:
                        state.append(prev)
                state = list(zip(*state))
                prev_state = [torch.cat(t, dim=1) for t in state]
        else:
            raise TypeError('not support prev_state type: {}'.format(type(prev_state)))
        return prev_state

    def _after_forward(self, next_state, list_next_state=False):
        """
        Overview:
            post process the next_state, return list or tensor type next_states
        Arguments:
            - next_state (:obj:`list` :obj:`Tuple` of :obj:`tensor`): list of Tuple contains the next (h, c)
            - list_next_state (:obj:`bool`): whether return next_state with list format, default set to False
        Returns:
            - next_state(:obj:`list` of :obj:`tensor` or :obj:`tensor`): the formated next_state
        """
        if list_next_state:
            h, c = [torch.stack(t, dim=0) for t in zip(*next_state)]
            batch_size = h.shape[1]
            next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
            next_state = list(zip(*next_state))
        else:
            next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        return next_state

def is_sequence(data):
    return isinstance(data, list) or isinstance(data, tuple)

