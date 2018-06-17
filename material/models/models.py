from .main_model import MainSegModel
def create_seg_model(args, n_class):
    model = None
    print(args.generator)
    model = MainSegModel()
    model.initialize(args, n_class)
    print("model [%s] was created" % (model.name()))
    return model
