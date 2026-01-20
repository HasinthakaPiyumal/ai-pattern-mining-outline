# Cluster 3

def filter_patch_sharpness(patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp):
    patches_src_c, patches_trg_c, patches_src_l, patches_src_r = ([], [], [], [])
    fitnessVal_3 = []
    fitnessVal_7 = []
    fitnessVal_11 = []
    fitnessVal_15 = []
    num_of_img_patches = len(patches_trg_c_temp)
    for i in range(num_of_img_patches):
        fitnessVal_3.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 3))
        fitnessVal_7.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 7))
        fitnessVal_11.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 11))
        fitnessVal_15.append(shapness_measure(cv2.cvtColor(patches_trg_c_temp[i], cv2.COLOR_BGR2GRAY), 15))
    fitnessVal_3 = np.asarray(fitnessVal_3)
    fitnessVal_7 = np.asarray(fitnessVal_7)
    fitnessVal_11 = np.asarray(fitnessVal_11)
    fitnessVal_15 = np.asarray(fitnessVal_15)
    fitnessVal_3 = (fitnessVal_3 - np.min(fitnessVal_3)) / np.max(fitnessVal_3 - np.min(fitnessVal_3))
    fitnessVal_7 = (fitnessVal_7 - np.min(fitnessVal_7)) / np.max(fitnessVal_7 - np.min(fitnessVal_7))
    fitnessVal_11 = (fitnessVal_11 - np.min(fitnessVal_11)) / np.max(fitnessVal_11 - np.min(fitnessVal_11))
    fitnessVal_15 = (fitnessVal_15 - np.min(fitnessVal_15)) / np.max(fitnessVal_15 - np.min(fitnessVal_15))
    fitnessVal_all = fitnessVal_3 * fitnessVal_7 * fitnessVal_11 * fitnessVal_15
    to_remove_patches_number = int(to_remove_ratio * num_of_img_patches)
    for itr in range(to_remove_patches_number):
        minArrInd = np.argmin(fitnessVal_all)
        fitnessVal_all[minArrInd] = 2
    for itr in range(num_of_img_patches):
        if fitnessVal_all[itr] != 2:
            patches_src_c.append(patches_src_c_temp[itr])
            patches_trg_c.append(patches_trg_c_temp[itr])
            patches_src_l.append(patches_src_l_temp[itr])
            patches_src_r.append(patches_src_r_temp[itr])
    return (patches_src_c, patches_trg_c, patches_src_l, patches_src_r)

def shapness_measure(img_temp, kernel_size):
    conv_x = cv2.Sobel(img_temp, cv2.CV_64F, 1, 0, ksize=kernel_size)
    conv_y = cv2.Sobel(img_temp, cv2.CV_64F, 0, 1, ksize=kernel_size)
    temp_arr_x = deepcopy(conv_x * conv_x)
    temp_arr_y = deepcopy(conv_y * conv_y)
    temp_sum_x_y = temp_arr_x + temp_arr_y
    temp_sum_x_y = np.sqrt(temp_sum_x_y)
    return np.sum(temp_sum_x_y)

def train_files(file_):
    lrL_file, lrR_file, lrC_file, hrC_file = file_
    filename = os.path.splitext(os.path.split(lrC_file)[-1])[0]
    lrL_img = cv2.imread(lrL_file, -1)
    lrR_img = cv2.imread(lrR_file, -1)
    lrC_img = cv2.imread(lrC_file, -1)
    hrC_img = cv2.imread(hrC_file, -1)
    lrC_patches, hrC_patches, lrL_patches, lrR_patches = slice_stride(lrC_img, hrC_img, lrL_img, lrR_img)
    lrC_patches, hrC_patches, lrL_patches, lrR_patches = filter_patch_sharpness(lrC_patches, hrC_patches, lrL_patches, lrR_patches)
    num_patch = 0
    for lrC_patch, hrC_patch, lrL_patch, lrR_patch in zip(lrC_patches, hrC_patches, lrL_patches, lrR_patches):
        num_patch += 1
        lrL_savename = os.path.join(lrL_tar, filename + '-' + str(num_patch) + '.png')
        lrR_savename = os.path.join(lrR_tar, filename + '-' + str(num_patch) + '.png')
        lrC_savename = os.path.join(lrC_tar, filename + '-' + str(num_patch) + '.png')
        hrC_savename = os.path.join(hrC_tar, filename + '-' + str(num_patch) + '.png')
        cv2.imwrite(lrL_savename, lrL_patch)
        cv2.imwrite(lrR_savename, lrR_patch)
        cv2.imwrite(lrC_savename, lrC_patch)
        cv2.imwrite(hrC_savename, hrC_patch)

def slice_stride(_img_src_c, _img_trg_c, _img_src_l, _img_src_r):
    coordinates_list = []
    coordinates_list.append([0, 0, 0, 0])
    patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp = ([], [], [], [])
    for r in range(0, _img_src_c.shape[0], stride[0]):
        for c in range(0, _img_src_c.shape[1], stride[1]):
            if r + patch_size[0] <= _img_src_c.shape[0] and c + patch_size[1] <= _img_src_c.shape[1]:
                patches_src_c_temp.append(_img_src_c[r:r + patch_size[0], c:c + patch_size[1]])
                patches_trg_c_temp.append(_img_trg_c[r:r + patch_size[0], c:c + patch_size[1]])
                patches_src_l_temp.append(_img_src_l[r:r + patch_size[0], c:c + patch_size[1]])
                patches_src_r_temp.append(_img_src_r[r:r + patch_size[0], c:c + patch_size[1]])
            elif r + patch_size[0] <= _img_src_c.shape[0] and (not [r, r + patch_size[0], _img_src_c.shape[1] - patch_size[1], _img_src_c.shape[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[r:r + patch_size[0], _img_src_c.shape[1] - patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(_img_trg_c[r:r + patch_size[0], _img_trg_c.shape[1] - patch_size[1]:_img_trg_c.shape[1]])
                patches_src_l_temp.append(_img_src_l[r:r + patch_size[0], _img_src_l.shape[1] - patch_size[1]:_img_src_l.shape[1]])
                patches_src_r_temp.append(_img_src_r[r:r + patch_size[0], _img_src_r.shape[1] - patch_size[1]:_img_src_r.shape[1]])
                coordinates_list.append([r, r + patch_size[0], _img_src_c.shape[1] - patch_size[1], _img_src_c.shape[1]])
            elif c + patch_size[1] <= _img_src_c.shape[1] and (not [_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], c, c + patch_size[1]] in coordinates_list):
                patches_src_c_temp.append(_img_src_c[_img_src_c.shape[0] - patch_size[0]:_img_src_c.shape[0], c:c + patch_size[1]])
                patches_trg_c_temp.append(_img_trg_c[_img_trg_c.shape[0] - patch_size[0]:_img_trg_c.shape[0], c:c + patch_size[1]])
                patches_src_l_temp.append(_img_src_l[_img_src_l.shape[0] - patch_size[0]:_img_src_l.shape[0], c:c + patch_size[1]])
                patches_src_r_temp.append(_img_src_r[_img_src_r.shape[0] - patch_size[0]:_img_src_r.shape[0], c:c + patch_size[1]])
                coordinates_list.append([_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], c, c + patch_size[1]])
            elif not [_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], _img_src_c.shape[1] - patch_size[1], _img_src_c.shape[1]] in coordinates_list:
                patches_src_c_temp.append(_img_src_c[_img_src_c.shape[0] - patch_size[0]:_img_src_c.shape[0], _img_src_c.shape[1] - patch_size[1]:_img_src_c.shape[1]])
                patches_trg_c_temp.append(_img_trg_c[_img_trg_c.shape[0] - patch_size[0]:_img_trg_c.shape[0], _img_trg_c.shape[1] - patch_size[1]:_img_trg_c.shape[1]])
                patches_src_l_temp.append(_img_src_l[_img_src_l.shape[0] - patch_size[0]:_img_src_l.shape[0], _img_src_l.shape[1] - patch_size[1]:_img_src_l.shape[1]])
                patches_src_r_temp.append(_img_src_r[_img_src_r.shape[0] - patch_size[0]:_img_src_r.shape[0], _img_src_r.shape[1] - patch_size[1]:_img_src_r.shape[1]])
                coordinates_list.append([_img_src_c.shape[0] - patch_size[0], _img_src_c.shape[0], _img_src_c.shape[1] - patch_size[1], _img_src_c.shape[1]])
    return (patches_src_c_temp, patches_trg_c_temp, patches_src_l_temp, patches_src_r_temp)

