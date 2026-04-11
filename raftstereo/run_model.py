import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt


DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = img[:,:,:3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def run_model(left_im_path, right_im_path):
    args = SimpleNamespace(
        restore_ckpt="models/raftstereo-middlebury.pth",
        corr_implementation="alt",
        left_img=left_im_path
        right_img=right_im_path,
        output_directory="demo_output",
        mixed_precision=True,
    )
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt,map_location=torch.device(DEVICE)))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        image1 = load_image(left_img)
        image2 = load_image(right_img)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        _, flow_up = model(image1, image2, iters=args.valid_iters, test_mode=True)
        flow_up = padder.unpad(flow_up).squeeze()

        file_stem = imfile1.split('/')[-2]
        # np.save(output_directory / "disparity.npy", flow_up.cpu().numpy().squeeze())
        # plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')
        return flow_up.cpu().numpy().squeeze()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
#     parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
#     parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH/*/im0.png")
#     parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH/*/im1.png")
#     parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
#     parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
#     parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

#     # Architecture choices
#     parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
#     parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
#     parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
#     parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
#     parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
#     parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
#     parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
#     parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
#     parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
#     args = parser.parse_args()

#     demo(args)
