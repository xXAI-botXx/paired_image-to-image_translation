from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=lambda x: float('inf') if x == 'inf' else int(x), default=float('inf'), help='how many test images to run')
        
        # options for loading an external complex model
        parser.add_argument('--use_external_complex_model', action='store_true', help='use external complex model (for pix2pix_cfo model)')
        # add physgen options
        parser.add_argument('--complex_model_name', type=str, default='', help='name of the complex model to load')
        parser.add_argument('--complex_model_only_reflexions', action='store_true', help='Whether to use only the reflexions as input if using reflexions.')
        parser.add_argument('--complex_model_input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')

        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
