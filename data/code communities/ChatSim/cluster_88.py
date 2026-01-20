# Cluster 88

class SegmentationAwarePairwiseScore(SegmentationAwareScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, individual_values = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std(), **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std(), **distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)}
        return (total_results, group_results)

    def reset(self):
        super().reset()
        self.individual_values = []

def get_segmentation_idx2name():
    return {i - 1: name for i, name in segm_options['classes'].set_index('Idx', drop=True)['Name'].to_dict().items()}

class SegmentationAwarePairwiseScore(SegmentationAwareScore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.individual_values = []
        self.segm_idx2name = get_segmentation_idx2name()

    def forward(self, pred_batch, target_batch, mask):
        cur_class_stats = super().forward(pred_batch, target_batch, mask)
        score_values = self.calc_score(pred_batch, target_batch, mask)
        self.individual_values.append(score_values)
        return cur_class_stats + (score_values,)

    @abstractmethod
    def calc_score(self, pred_batch, target_batch, mask):
        raise NotImplementedError()

    def get_value(self, groups=None, states=None):
        """
        :param groups:
        :return:
            total_results: dict of kind {'mean': score mean, 'std': score std}
            group_results: None, if groups is None;
                else dict {group_idx: {'mean': score mean among group, 'std': score std among group}}
        """
        if states is not None:
            target_class_freq_by_image_total, target_class_freq_by_image_mask, pred_class_freq_by_image_mask, individual_values = states
        else:
            target_class_freq_by_image_total = self.target_class_freq_by_image_total
            target_class_freq_by_image_mask = self.target_class_freq_by_image_mask
            pred_class_freq_by_image_mask = self.pred_class_freq_by_image_mask
            individual_values = self.individual_values
        target_class_freq_by_image_total = np.concatenate(target_class_freq_by_image_total, axis=0)
        target_class_freq_by_image_mask = np.concatenate(target_class_freq_by_image_mask, axis=0)
        pred_class_freq_by_image_mask = np.concatenate(pred_class_freq_by_image_mask, axis=0)
        individual_values = np.concatenate(individual_values, axis=0)
        total_results = {'mean': individual_values.mean(), 'std': individual_values.std(), **distribute_values_to_classes(target_class_freq_by_image_mask, individual_values, self.segm_idx2name)}
        if groups is None:
            return (total_results, None)
        group_results = dict()
        grouping = get_groupings(groups)
        for label, index in grouping.items():
            group_class_freq = target_class_freq_by_image_mask[index]
            group_scores = individual_values[index]
            group_results[label] = {'mean': group_scores.mean(), 'std': group_scores.std(), **distribute_values_to_classes(group_class_freq, group_scores, self.segm_idx2name)}
        return (total_results, group_results)

    def reset(self):
        super().reset()
        self.individual_values = []

