import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from dataloader import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import time
import argparse
from utils import h_split,l_split
from models.model import HCSD_Net

def _test(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.dataset_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()

        # Hardware warm-up
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, h_img, _ = data
            input_img = input_img.to(device)
            tm = time.time()
            _ = model(input_img, h_img)
            _ = time.time() - tm

            if iter_idx == 20:
                break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img,h_img, name = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img, h_img)[1]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f time: %f' % (iter_idx + 1, psnr, elapsed))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print("Average time: %f" % adder.average())

############## parser #############
parser = argparse.ArgumentParser(description='HCSD_Net test')
parser.add_argument("--gpu_id",type=str, default='0', help='GPU id')
parser.add_argument('--dataset_dir', type=str, default='datasets/Snow100K', help='dir of test data')
parser.add_argument('--test_model', type=str, default='results/Snow100K/weights/Best.pkl')
parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
parser.add_argument('--model_name', default='HCSD_Net', type=str)
parser.add_argument("--model_epo",type=str, default='best', help='test model epoch')

args = parser.parse_args()
args.result_dir = os.path.join('results/', args.model_name, 'result_image/',args.model_epo)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# l_test = l_split(os.path.join(args.dataset_dir, 'test'))
model = HCSD_Net()
model = torch.nn.DataParallel(model)
if torch.cuda.is_available():
    model.cuda()
_test(model,args)
