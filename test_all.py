# import matlab.engine
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
from skimage import color,io
from torchvision import transforms
from PIL import Image
import os

parser = argparse.ArgumentParser(description="PyTorch MDDM Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./mddm.pth", type=str, help="model path")

opt = parser.parse_args()
cuda = opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

def demoire(root,image_name,model,out_path):
    im = Image.open('%s/%s'%(root,image_name))
    im_array = np.array(im)
    TS = transforms.Compose([transforms.ToTensor()])
    im_input = TS(im).view(-1,3,im_array.shape[0],im_array.shape[1])

    if cuda:
        model = model.cuda()
        im_input = im_input.cuda()
    else:
        model = model.cpu()

    start_time = time.time()
    clear = model(im_input)
    elapsed_time = time.time() - start_time

    clear = clear.cpu()

    im_h = clear.data[0].numpy().astype(np.float32)
    print(im_h.shape)
    im_h = im_h*255.
    im_h = np.clip(im_h, 0., 255.)
    im_h = im_h.transpose(1,2,0).astype(np.float32)
    io.imsave(out_path+image_name,im_h/255.)
    return elapsed_time

if __name__ == "__main__":
    root = './test'
    images = os.listdir(root)
    out_path = './out/'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    process_time = []
    model = torch.load(opt.model)["model"]
    model.eval()
    for image in images:
        elapsed_time = demoire(root,image,model,out_path)
        process_time.append(elapsed_time)
    avg_time = np.mean(process_time)
    print(avg_time)


