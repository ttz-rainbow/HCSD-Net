#!/usr/bin/env python
#-*-coding:utf-8-*-
#@Author:ZhangTing
#@Email:ting_zhang369@163.com
#@CodeFunction:
import os
import torch
import argparse
from torch.backends import cudnn
from models.HCSD_Nett import HCSD_Net
from valid import val
from dataloader import train_dataloader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import h_split,Adder,Timer,check_lr

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import irfft2
    from torch.fft import rfft2
    def rfft(x, d):
        t = rfft2(x, dim = (-d))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return irfft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))

if __name__ == '__main__':
    ################## parser ####################
    parser = argparse.ArgumentParser(description='HCSD_Net train')
    parser.add_argument("--gpu_id",type=str, default='0', help='GPU id')
    parser.add_argument('--dataset_dir', type=str, default='D:\ZT\Datasets\desnow_datasets\Snow100K\\train_val_set', help='dir of train/val data')
    parser.add_argument('--nepoch', type=int, default=500, help='training epochs')

    parser.add_argument('--resume', action='store_true',default=False)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--train_patch', type=int, default=8, help='patch size')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
    parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_steps', type=list, default=[(x + 1) * 500 for x in range(3000 // 500)])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
    parser.add_argument('--model_name', default='HCSD_Net_on_Snow100k', type=str)

    args = parser.parse_args()
    print(args)
    args.model_save_dir = os.path.join('results/', args.model_name, 'weights/')
    args.result_dir = os.path.join('results/', args.model_name, 'result_image/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
        os.makedirs(args.result_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    ################ models #######################
    model = HCSD_Net()
    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.cuda()

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    epoch = 1

    # h_train = h_split(os.path.join(args.dataset_dir,'train'))
    # h_val = h_split(os.path.join(args.dataset_dir,'valid'))

    dataloader = train_dataloader(args.dataset_dir, args.train_patch, args.num_worker)
    max_iter = len(dataloader)
    ############### resume ########################
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        print('epoch',epoch)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d' % epoch)
        epoch += 1

    ############## writer ##########################
    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr=-1

    for epoch_idx in range(epoch, args.nepoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img, c_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            c_img = c_img.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img, c_img)
            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            l2 = criterion(pred_img[0], label_img2)
            l3 = criterion(pred_img[1], label_img)
            loss_content = l2 + l3

            label_fft2 = rfft(label_img2,2)
            pred_fft2 = rfft(pred_img[0], 2)
            label_fft3 = rfft(label_img,2)
            pred_fft3 = rfft(pred_img[1], 2)

            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f2 + f3

            loss = loss_content + 0.1 * loss_fft
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())
            #print(iter_idx+1)
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
        overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        #50epoch保存一次模型
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch_idx}, save_name)
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % args.valid_freq == 0:
            # val_model = HCSD_Net(task='test')
            # val_model = torch.nn.DataParallel(val_model)
            # if torch.cuda.is_available():
            #     val_model.cuda()
            val_gopro = val(model, args, epoch_idx)
            print('%03d epoch \n Average CSD PSNR %.2f dB' % (epoch_idx, val_gopro))
            writer.add_scalar('PSNR_derain', val_gopro, epoch_idx)
            if val_gopro >= best_psnr:
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))
    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    torch.save({'model': model.state_dict()}, save_name)


