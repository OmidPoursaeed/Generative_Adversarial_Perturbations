import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, args):
        self.args = args
        self.gpu_ids = args.gpu_ids
        self.isTrain = args.isTrain
        self.FloatTensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.save_dir = os.path.join(args.checkpoints_dir, args.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no back propagation
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
