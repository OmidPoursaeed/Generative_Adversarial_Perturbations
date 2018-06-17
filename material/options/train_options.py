from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--lr', help='learning rate',
            type=float, default=0.00008
        )
        self.parser.add_argument(
            '--beta1', help='momentum term of adam',
            type=float, default=0.9
        )
        self.parser.add_argument(
            '--nEpochs', help='# of epochs',
            type=int, default=20
        )
        self.parser.add_argument(
            '--lr_policy', type=str, default='lambda',
            help='learning rate policy: lambda|step|plateau'
        )
        self.parser.add_argument(
            '--lr_decay_iters', type=int, default=50,
            help='multiply by a gamma every lr_decay_iters iterations'
        )
        self.parser.add_argument(
            '--display_freq', type=int, default=100,
            help='frequency of showing training results on screen'
        )

        self.isTrain = True
