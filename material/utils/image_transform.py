import PIL.Image
import numpy as np
import torch
from . import label


full_to_train = {
    -1: 19, 0: 19, 1: 0, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19,
    7: 0, 8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19,
    15: 19, 16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9,
    23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19,
    30: 19, 31: 16, 32: 17, 33: 18
}
train_to_full = {
    0: 7, 1: 8, 2: 11, 3: 12, 4: 13, 5: 17, 6: 19, 7: 20, 8: 21, 9: 22,
    10: 23, 11: 24, 12: 25, 13: 26, 14: 27, 15: 28, 16: 31, 17: 32, 18: 33,
    19: 0
}

trainId2color = {label.trainId: label.color for label in label.labels}

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


cmap = color_map()


def myfunc(a):
    # print(a)
    if a in full_to_train:
        return full_to_train[a]
    else:
        19


def process_img_file(img_file):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    return img


def process_lbl_file(lbl_file):
    lbl = PIL.Image.open(lbl_file)
    lbl = np.array(lbl, dtype=np.int32)

    w, h = lbl.shape
    lbl = lbl.reshape(-1)
    vfunc = np.vectorize(myfunc)
    lbl = vfunc(lbl).reshape(w, h)
    return lbl

def mapping_func(lbl, n_class):
    # print(lbl)
    if lbl >= 0 and lbl < n_class:
        return mapping_global[lbl]


def do_mapping(lbl, mapping, n_class):
    lbl = lbl.numpy()
    global mapping_global
    mapping_global = mapping
    _, w, h = lbl.shape
    lbl = lbl.reshape(-1)
    vfunc = np.vectorize(mapping_func)
    lbl = vfunc(lbl, n_class).reshape(1, w, h)
    return torch.from_numpy(lbl).long()


def to_tensor(img, lbl):
    img_tensor = None
    lbl_tensor = None

    if img is not None:
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).float()
    if lbl is not None:
        lbl_tensor = torch.from_numpy(lbl).long()
    return img_tensor, lbl_tensor


def to_numpy(img, lbl):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]

    if lbl is not None:
        lbl = lbl.numpy()
        return img, lbl
    else:
        return img


def transform(img, lbl, mode, dataset, optional=-1.0):
    if mode == 1:
        img = img.div_(128.0).add_(optional)
    elif mode == 2:
        if dataset == 'cityscapes':
            img[0] = img[0].add(-72.78044)
            img[1] = img[1].add(-83.21195)
            img[2] = img[2].add(-73.45286)

        elif dataset == 'pascalvoc':
            img[0].add(-104.00698793)
            img[1].add(-116.66876762)
            img[2].add(-122.67891434)

    if lbl is not None:
        return img, lbl
    else:
        return img


def untransform(img, lbl, mode, dataset):
    if mode == 1:
        img = img.add_(1.0).mul_(128.0)
        img = torch.clamp(img, 0, 255)
    elif mode == 2:
        if dataset == 'cityscapes':
            img[0] = img[0].add(-72.78044)
            img[1] = img[1].add(-83.21195)
            img[2] = img[2].add(-73.45286)
        elif dataset == 'pascalvoc':
            img[0].add(104.00698793)
            img[1].add(116.66876762)
            img[2].add(122.67891434)

    if lbl is not None:
        return img, lbl
    else:
        return img


def inf_norm_adjust(img, eps):
    for i in range(img.size()[0]):
        # for loop for RGB
        for col in range(3):
            inf_norm = img[i].data[col].abs().max()
            coef = min(1.0, eps / inf_norm)
            img[i].data[col] *= coef
    return img


def l2_loss(tensor):
    crt = torch.nn.MSELoss().cuda()
    base = torch.FloatTensor(tensor.size()).zero_()
    base = torch.autograd.Variable(base, requires_grad=False).cuda()
    loss = crt(tensor, base)
    loss = loss * 3
    loss = torch.sqrt(loss)
    return loss


def l2_norm_adjust(delta, eps):
    norm = l2_loss(delta).data[0]
    delta.data *= min(1.0, eps / norm)
    return delta


def transform_pred_res(pred, dataset):
    if dataset == 'cityscapes':
        h, w = pred.shape
        data = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                trainId = pred[i][j]
                if trainId == 19:
                    trainId = 255
                data[i][j] = trainId2color[trainId]
        data = data.transpose(2, 0, 1)
        data = torch.FloatTensor(data)
        return data
    elif dataset == 'pascalvoc':
        h, w = pred.shape
        data = np.zeros((h, w, 3), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                trainId = pred[i][j]
                data[i][j] = cmap[trainId]
        data = data.transpose(2, 0, 1)
        data = torch.FloatTensor(data)
        return data


def save_img(np_img, fp):
    img = PIL.Image.fromarray(np_img)
    img.save(fp, format='PNG')


def save_transform(img):
    img = img.numpy()
    img = img.astype(np.uint8)
    img = img[::-1, :, :]

    return img
