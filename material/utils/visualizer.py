import numpy as np
import os
import time
import os.path as osp
import torch
from . import util
from pdb import set_trace as st


class Visualizer():
    def __init__(self, args):
        self.display_id = args.display_id
        # self.use_html =
        self.window_size = args.display_window_size
        self.name = args.name

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=args.display_port, env=self.name)
        self.log_path = osp.join(args.checkpoints_dir, args.name, 'log.txt')
        with open(self.log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_visuals(self, res, epoch):
        for name in list(res.keys()):
            self.vis.image(
                res[name],
                opts=dict(title=name),
                win=name
            )

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, total_iter, opt, errors, list_name):
        if list_name == 'train':
            if not hasattr(self, 'plot_train_data'):
                self.plot_train_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
                self.plot_train_data['X'].append(total_iter)
                self.plot_train_data['Y'].append([errors[k] for k in self.plot_train_data['legend']])
                self.vis.line(
                    X=np.broadcast_to(np.array([total_iter]), (1, 2)),
                    Y=np.broadcast_to(np.array(self.plot_train_data['Y'][total_iter]), (1, 2)),
                    opts={
                        'legend': self.plot_train_data['legend'],
                        'xlabel': 'iteration',
                    },
                    win='train_plot'
                )
                return

            self.plot_train_data['X'].append(total_iter)
            self.plot_train_data['Y'].append([errors[k] for k in self.plot_train_data['legend']])
            self.vis.line(
                X=np.broadcast_to(np.array([total_iter]), (1, 2)),
                Y=np.broadcast_to(np.array(self.plot_train_data['Y'][total_iter]), (1, 2)),
                win='train_plot',
                update='append'
            )
        elif list_name == 'test':
            if not hasattr(self, 'plot_test_data'):
                self.plot_test_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
                self.plot_test_data['X'].append(epoch)
                self.plot_test_data['Y'].append([errors[k] for k in self.plot_test_data['legend']])
                self.vis.line(
                    X=np.broadcast_to(np.array([epoch]), (1, 2)),
                    Y=np.broadcast_to(np.array(self.plot_test_data['Y'][epoch]), (1, 2)),
                    opts={
                        'legend': self.plot_test_data['legend'],
                        'xlabel': 'epoch',
                    },
                    win='test_plot'
                )
                return

            self.plot_test_data['X'].append(epoch)
            self.plot_test_data['Y'].append([errors[k] for k in self.plot_test_data['legend']])
            self.vis.line(
                X=np.broadcast_to(np.array([epoch]), (1, 2)),
                Y=np.broadcast_to(np.array(self.plot_test_data['Y'][epoch]), (1, 2)),
                win='test_plot',
                update='append'
            )
