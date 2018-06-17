import os
from . import cityscape
from . import pascalvoc
from torch.utils.data import DataLoader


def create_data_loader(args):
    if args.dataset == 'cityscapes':
        root = os.path.expanduser(args.dataroot)
        dataset_dir = os.path.join(root, 'cityscapes-{}'.format(args.resolution))
        train_set = cityscape.CityScapesClassSeg(
            dataset_dir, split=['train'], transform=1,
        )
        test_set = cityscape.CityScapesClassSeg(
            dataset_dir, split=['val'], transform=1,
        )
        training_data_loader = DataLoader(
            dataset=train_set, num_workers=args.nThreads,
            batch_size=args.batch_size, shuffle=True
        )
        testing_data_loader = DataLoader(
            dataset=test_set, num_workers=args.nThreads,
            batch_size=args.batch_size, shuffle=False
        )
        return training_data_loader, testing_data_loader
    elif args.dataset == 'pascalvoc':
        root = os.path.expanduser('~/data/datasets')
        kwargs = {'num_workers': 4, 'pin_memory': True}
        train_loader = DataLoader(
            pascalvoc.SBDClassSeg(
                root, split='train', transform=1
            ),
            batch_size=1, shuffle=True, **kwargs)
        val_loader = DataLoader(
            pascalvoc.VOC2011ClassSeg(
                root, split='seg11valid', transform=1
            ),
            batch_size=1, shuffle=False, **kwargs)
        return train_loader, val_loader
