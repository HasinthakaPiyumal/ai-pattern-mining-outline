# Cluster 115

class SegmentationMask:

    def __init__(self, confidence_threshold=0.5, rigidness_mode=RigidnessMode.rigid, max_object_area=0.3, min_mask_area=0.02, downsample_levels=6, num_variants_per_mask=4, max_mask_intersection=0.5, max_foreground_coverage=0.5, max_foreground_intersection=0.5, max_hidden_area=0.2, max_scale_change=0.25, horizontal_flip=True, max_vertical_shift=0.1, position_shuffle=True):
        """
        :param confidence_threshold: float; threshold for confidence of the panoptic segmentator to allow for
        the instance.
        :param rigidness_mode: RigidnessMode object
            when soft, checks intersection only with the object from which the mask_object was produced
            when rigid, checks intersection with any foreground class object
        :param max_object_area: float; allowed upper bound for to be considered as mask_object.
        :param min_mask_area: float; lower bound for mask to be considered valid
        :param downsample_levels: int; defines width of the resized segmentation to obtain shifted masks;
        :param num_variants_per_mask: int; maximal number of the masks for the same object;
        :param max_mask_intersection: float; maximum allowed area fraction of intersection for 2 masks
        produced by horizontal shift of the same mask_object; higher value -> more diversity
        :param max_foreground_coverage: float; maximum allowed area fraction of intersection for foreground object to be
        covered by mask; lower value -> less the objects are covered
        :param max_foreground_intersection: float; maximum allowed area of intersection for the mask with foreground
        object; lower value -> mask is more on the background than on the objects
        :param max_hidden_area: upper bound on part of the object hidden by shifting object outside the screen area;
        :param max_scale_change: allowed scale change for the mask_object;
        :param horizontal_flip: if horizontal flips are allowed;
        :param max_vertical_shift: amount of vertical movement allowed;
        :param position_shuffle: shuffle
        """
        assert DETECTRON_INSTALLED, 'Cannot use SegmentationMask without detectron2'
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file('COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml'))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml')
        self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.predictor = DefaultPredictor(self.cfg)
        self.rigidness_mode = RigidnessMode(rigidness_mode)
        self.max_object_area = max_object_area
        self.min_mask_area = min_mask_area
        self.downsample_levels = downsample_levels
        self.num_variants_per_mask = num_variants_per_mask
        self.max_mask_intersection = max_mask_intersection
        self.max_foreground_coverage = max_foreground_coverage
        self.max_foreground_intersection = max_foreground_intersection
        self.max_hidden_area = max_hidden_area
        self.position_shuffle = position_shuffle
        self.max_scale_change = max_scale_change
        self.horizontal_flip = horizontal_flip
        self.max_vertical_shift = max_vertical_shift

    def get_segmentation(self, img):
        im = img_as_ubyte(img)
        panoptic_seg, segment_info = self.predictor(im)['panoptic_seg']
        return (panoptic_seg, segment_info)

    @staticmethod
    def _is_power_of_two(n):
        return n != 0 and n & n - 1 == 0

    def identify_candidates(self, panoptic_seg, segments_info):
        potential_mask_ids = []
        for segment in segments_info:
            if not segment['isthing']:
                continue
            mask = (panoptic_seg == segment['id']).int().detach().cpu().numpy()
            area = mask.sum().item() / np.prod(panoptic_seg.shape)
            if area >= self.max_object_area:
                continue
            potential_mask_ids.append(segment['id'])
        return potential_mask_ids

    def downsample_mask(self, mask):
        height, width = mask.shape
        if not (self._is_power_of_two(height) and self._is_power_of_two(width)):
            raise ValueError('Image sides are not power of 2.')
        num_iterations = width.bit_length() - 1 - self.downsample_levels
        if num_iterations < 0:
            raise ValueError(f'Width is lower than 2^{self.downsample_levels}.')
        if height.bit_length() - 1 < num_iterations:
            raise ValueError('Height is too low to perform downsampling')
        downsampled = mask
        for _ in range(num_iterations):
            downsampled = zero_corrected_countless(downsampled)
        return downsampled

    def _augmentation_params(self):
        scaling_factor = np.random.uniform(1 - self.max_scale_change, 1 + self.max_scale_change)
        if self.horizontal_flip:
            horizontal_flip = bool(np.random.choice(2))
        else:
            horizontal_flip = False
        vertical_shift = np.random.uniform(-self.max_vertical_shift, self.max_vertical_shift)
        return {'scaling_factor': scaling_factor, 'horizontal_flip': horizontal_flip, 'vertical_shift': vertical_shift}

    def _get_intersection(self, mask_array, mask_object):
        intersection = mask_array[mask_object.up:mask_object.down, mask_object.left:mask_object.right] & mask_object.mask
        return intersection

    def _check_masks_intersection(self, aug_mask, total_mask_area, prev_masks):
        for existing_mask in prev_masks:
            intersection_area = self._get_intersection(existing_mask, aug_mask).sum()
            intersection_existing = intersection_area / existing_mask.sum()
            intersection_current = 1 - (aug_mask.area() - intersection_area) / total_mask_area
            if intersection_existing > self.max_mask_intersection or intersection_current > self.max_mask_intersection:
                return False
        return True

    def _check_foreground_intersection(self, aug_mask, foreground):
        for existing_mask in foreground:
            intersection_area = self._get_intersection(existing_mask, aug_mask).sum()
            intersection_existing = intersection_area / existing_mask.sum()
            if intersection_existing > self.max_foreground_coverage:
                return False
            intersection_mask = intersection_area / aug_mask.area()
            if intersection_mask > self.max_foreground_intersection:
                return False
        return True

    def _move_mask(self, mask, foreground):
        orig_mask = ObjectMask(mask)
        chosen_masks = []
        chosen_parameters = []
        scaling_factor_lower_bound = 0.0
        for var_idx in range(self.num_variants_per_mask):
            augmentation_params = self._augmentation_params()
            augmentation_params['scaling_factor'] = min([augmentation_params['scaling_factor'], 2 * min(orig_mask.up, orig_mask.height - orig_mask.down) / orig_mask.height + 1.0, 2 * min(orig_mask.left, orig_mask.width - orig_mask.right) / orig_mask.width + 1.0])
            augmentation_params['scaling_factor'] = max([augmentation_params['scaling_factor'], scaling_factor_lower_bound])
            aug_mask = deepcopy(orig_mask)
            aug_mask.rescale(augmentation_params['scaling_factor'], inplace=True)
            if augmentation_params['horizontal_flip']:
                aug_mask.horizontal_flip(inplace=True)
            total_aug_area = aug_mask.area()
            if total_aug_area == 0:
                scaling_factor_lower_bound = 1.0
                continue
            vertical_area = aug_mask.mask.sum(axis=1) / total_aug_area
            max_hidden_up = np.searchsorted(vertical_area.cumsum(), self.max_hidden_area)
            max_hidden_down = np.searchsorted(vertical_area[::-1].cumsum(), self.max_hidden_area)
            augmentation_params['vertical_shift'] = np.clip(augmentation_params['vertical_shift'], -(aug_mask.up + max_hidden_up) / aug_mask.height, (aug_mask.height - aug_mask.down + max_hidden_down) / aug_mask.height)
            vertical_shift = int(round(aug_mask.height * augmentation_params['vertical_shift']))
            aug_mask.shift(vertical=vertical_shift, inplace=True)
            aug_mask.crop_to_canvas(vertical=True, horizontal=False, inplace=True)
            max_hidden_area = self.max_hidden_area - (1 - aug_mask.area() / total_aug_area)
            horizontal_area = aug_mask.mask.sum(axis=0) / total_aug_area
            max_hidden_left = np.searchsorted(horizontal_area.cumsum(), max_hidden_area)
            max_hidden_right = np.searchsorted(horizontal_area[::-1].cumsum(), max_hidden_area)
            allowed_shifts = np.arange(-max_hidden_left, aug_mask.width - (aug_mask.right - aug_mask.left) + max_hidden_right + 1)
            allowed_shifts = -(aug_mask.left - allowed_shifts)
            if self.position_shuffle:
                np.random.shuffle(allowed_shifts)
            mask_is_found = False
            for horizontal_shift in allowed_shifts:
                aug_mask_left = deepcopy(aug_mask)
                aug_mask_left.shift(horizontal=horizontal_shift, inplace=True)
                aug_mask_left.crop_to_canvas(inplace=True)
                prev_masks = [mask] + chosen_masks
                is_mask_suitable = self._check_masks_intersection(aug_mask_left, total_aug_area, prev_masks) & self._check_foreground_intersection(aug_mask_left, foreground)
                if is_mask_suitable:
                    aug_draw = aug_mask_left.restore_full_mask()
                    chosen_masks.append(aug_draw)
                    augmentation_params['horizontal_shift'] = horizontal_shift / aug_mask_left.width
                    chosen_parameters.append(augmentation_params)
                    mask_is_found = True
                    break
            if not mask_is_found:
                break
        return chosen_parameters

    def _prepare_mask(self, mask):
        height, width = mask.shape
        target_width = width if self._is_power_of_two(width) else 1 << width.bit_length()
        target_height = height if self._is_power_of_two(height) else 1 << height.bit_length()
        return resize(mask.astype('float32'), (target_height, target_width), order=0, mode='edge').round().astype('int32')

    def get_masks(self, im, return_panoptic=False):
        panoptic_seg, segments_info = self.get_segmentation(im)
        potential_mask_ids = self.identify_candidates(panoptic_seg, segments_info)
        panoptic_seg_scaled = self._prepare_mask(panoptic_seg.detach().cpu().numpy())
        downsampled = self.downsample_mask(panoptic_seg_scaled)
        scene_objects = []
        for segment in segments_info:
            if not segment['isthing']:
                continue
            mask = downsampled == segment['id']
            if not np.any(mask):
                continue
            scene_objects.append(mask)
        mask_set = []
        for mask_id in potential_mask_ids:
            mask = downsampled == mask_id
            if not np.any(mask):
                continue
            if self.rigidness_mode is RigidnessMode.soft:
                foreground = [mask]
            elif self.rigidness_mode is RigidnessMode.rigid:
                foreground = scene_objects
            else:
                raise ValueError(f'Unexpected rigidness_mode: {rigidness_mode}')
            masks_params = self._move_mask(mask, foreground)
            full_mask = ObjectMask((panoptic_seg == mask_id).detach().cpu().numpy())
            for params in masks_params:
                aug_mask = deepcopy(full_mask)
                aug_mask.rescale(params['scaling_factor'], inplace=True)
                if params['horizontal_flip']:
                    aug_mask.horizontal_flip(inplace=True)
                vertical_shift = int(round(aug_mask.height * params['vertical_shift']))
                horizontal_shift = int(round(aug_mask.width * params['horizontal_shift']))
                aug_mask.shift(vertical=vertical_shift, horizontal=horizontal_shift, inplace=True)
                aug_mask = aug_mask.restore_full_mask().astype('uint8')
                if aug_mask.mean() <= self.min_mask_area:
                    continue
                mask_set.append(aug_mask)
        if return_panoptic:
            return (mask_set, panoptic_seg.detach().cpu().numpy())
        else:
            return mask_set

def zero_corrected_countless(data):
    """
  Vectorized implementation of downsampling a 2D 
  image by 2 on each side using the COUNTLESS algorithm.
  
  data is a 2D numpy array with even dimensions.
  """
    data, upgraded = upgrade_type(data)
    data += 1
    sections = []
    factor = (2, 2)
    for offset in np.ndindex(factor):
        part = data[tuple((np.s_[o::f] for o, f in zip(offset, factor)))]
        sections.append(part)
    a, b, c, d = sections
    ab = a * (a == b)
    ac = a * (a == c)
    bc = b * (b == c)
    a = ab | ac | bc
    result = a + (a == 0) * d - 1
    if upgraded:
        return downgrade_type(result)
    data -= 1
    return result

