import tqdm
import torch
import shutil
import os
import torch.backends.cudnn as cudnn
from material.options.train_options import TrainOptions
from material.data.data_loader import create_data_loader
from material.models.models import create_seg_model
from material.utils.visualizer import Visualizer


args = TrainOptions().parse()
cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
train_data_loader, test_data_loader = create_data_loader(args)
n_class = len(test_data_loader.dataset.class_names)
trainset_size = len(train_data_loader)
testset_size = len(test_data_loader)
model = create_seg_model(args, n_class)
visualizer = Visualizer(args)
total_iter = 0
start_epoch = 0
expr_dir = os.path.join(args.checkpoints_dir, args.name)

if args.metric == 0:
    # if the metrics is success_rate,
    # higher the greater
    best_metric = 0.0
    higher = True
elif args.metric == 2:
    # if the metrics is mean IOU,
    # lower the greater
    best_metric = 1.0
    higher = False

# processing the resume
if args.resume is True:
    print('resume from experiment '.format(args.resume_name))
    resume_dir = os.path.join(args.checkpoints_dir, args.resume_name)
    checkpoint = torch.load(os.path.join(resume_dir, 'checkpoint.pth.tar'))
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

for epoch in tqdm.trange(start_epoch, args.nEpochs):
    model.train(True)
    for batch_idx, data in tqdm.tqdm(
            enumerate(train_data_loader), total=len(train_data_loader),
            desc='Train epoch=%d' % epoch, leave=False
    ):
        model.set_input(data)
        model.optimize_parameters()

        errors = model.get_current_errors()
        if args.display_id > 0:
            visualizer.plot_current_errors(epoch, total_iter, args, errors, 'train')
        if total_iter % args.display_freq == 0:
            visualizer.display_current_visuals(
                model.get_current_visuals(), epoch
            )

        total_iter += args.batch_size

        if args.test and batch_idx >= 5:
            break

    model.update_learning_rate()

    model.train(False)
    for batch_idx, data in tqdm.tqdm(
            enumerate(test_data_loader), total=len(test_data_loader),
            desc='Validation epoch=%d' % epoch, leave=False
    ):
        model.set_input(data)
        model.optimize_parameters()

        if batch_idx % args.display_freq == 0:
            visualizer.display_current_visuals(
                model.get_current_visuals(), epoch
            )

        if args.test and batch_idx >= 5:
            break

    errors = model.get_current_errors(batch_idx)
    visualizer.plot_current_errors(epoch, total_iter, args, errors, 'test')

    if args.save is True:
        print(errors['acc'])
        # save the model after validation
        if higher is True:
            is_best = best_metric < errors['acc']
        else:
            is_best = best_metric > errors['acc']
        if is_best:
            best_metric = errors['acc']
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': model.optimizerG.state_dict(),
            'generator_state_dict': model.generator.state_dict(),
            'best_metric': best_metric,
        }, os.path.join(expr_dir, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(os.path.join(expr_dir, 'checkpoint.pth.tar'),
                        os.path.join(expr_dir, 'model_best.pth.tar'))
