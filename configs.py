class Config(object):
    """Global Config class"""
    def __init__(self,):

        # Dataset configs
        self.spatial_dimentions = 2
        self.img_size = (256, 256)
        self.batch_size = 4
        self.data_loaders_num_workers = 4
        self.data_dir = None
        self.cut_ratio_1 = 0.8
        self.cut_ratio_2 = 1
        self.method = "RicianNoise"  # RicianNoise, MotionBlur, Acceleration

        # Training configs
        self.learning_rate = 0.001
        self.models_dir = None
        self.num_epochs = 50
        self.num_epochs_per_saving = 10
        self.normalize_input = False

        # Model configs
        self.net = "CUNet"
        self.unet_global_residual_conn = False
        self.kernel_size = 3
        self.bias = True
        self.norm = 'RadialNorm' # RadialNorm, CNorm
        self.activation = 'CReLU' # CReLU, ModReLU, ZReLU
        self.activation_params = {
            'inplace': True,
        }
        self.bn_t = 1
        self.dropout_ratio = 0.0


config = Config()


