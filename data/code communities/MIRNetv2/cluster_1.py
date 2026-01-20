# Cluster 1

def get_weights_and_parameters(task, parameters):
    if task == 'real_denoising':
        weights = os.path.join('Real_Denoising', 'pretrained_models', 'real_denoising.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/real_denoising.pth -P Real_Denoising/pretrained_models/')
    elif task == 'super_resolution':
        weights = os.path.join('Super_Resolution', 'pretrained_models', 'sr_x4.pth')
        parameters['scale'] = 4
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/sr_x4.pth -P Super_Resolution/pretrained_models/')
    elif task == 'contrast_enhancement':
        weights = os.path.join('Enhancement', 'pretrained_models', 'enhancement_fivek.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_fivek.pth -P Enhancement/pretrained_models/')
    elif task == 'lowlight_enhancement':
        weights = os.path.join('Enhancement', 'pretrained_models', 'enhancement_lol.pth')
        if not os.path.exists(weights):
            os.system('wget https://github.com/swz30/MIRNetv2/releases/download/v1.0.0/enhancement_lol.pth -P Enhancement/pretrained_models/')
    return (weights, parameters)

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

