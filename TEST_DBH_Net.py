import torch
import torch.nn as nn
import math
import scipy.io as sio
import numpy as np
import os
import glob
from time import time
import cv2

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
from argparse import ArgumentParser
import types
from model import DBHNet

parser = ArgumentParser(description='DBH_Net')

parser.add_argument('--epoch_num', type=int, default=440, help='epoch number of model')
parser.add_argument('--layer_num', type=int, default=25, help='phase number of DBNet')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=5, help='from {5, 10, 20, 30, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--test_name', type=str, default='BrainImages', choices=['BrainImages', 'ixi', 'CC359'],
                    help='name of test set, choose BrainImages, ixi, CC359')
parser.add_argument('--net', type=str, default='DBH_Net', help='Name of Net')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training or test data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')
parser.add_argument('--result_dir', type=str, default='result', help='result directory')

args = parser.parse_args()

epoch_num = args.epoch_num
layer_num = args.layer_num
group_num = args.group_num
cs_ratio = args.cs_ratio
gpu_list = args.gpu_list
test_name = args.test_name

###########################################################################################

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except:
    pass


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###########################################################################################
Phi_data_Name = './%s/mask_%d.mat' % (args.matrix_dir, cs_ratio)
Phi_data = sio.loadmat(Phi_data_Name)
mask_matrix = Phi_data['mask']

mask_matrix = torch.from_numpy(mask_matrix).type(torch.FloatTensor)
mask = mask_matrix.to(device)
###########################################################################################

model = DBHNet(layer_num)
model = nn.DataParallel(model)
model = model.to(device)

###########################################################################################
model_dir = "./%s/MRI_CS_%s_%s_layer_%d" % (args.model_dir, args.net, args.test_name, layer_num)
model.load_state_dict(torch.load('./%s/net_params_%d.pkl' % (model_dir, epoch_num)))

def psnr(img1, img2):
    img1.astype(np.float32)
    img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()

test_dir = os.path.join(args.data_dir, test_name)
filepaths = glob.glob(test_dir + '/*.png')

result_dir = os.path.join(args.result_dir, test_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

ImgNum = len(filepaths)
PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)

Init_PSNR_All = np.zeros([1, ImgNum], dtype=np.float32)
Init_SSIM_All = np.zeros([1, ImgNum], dtype=np.float32)


print('\n')
print("MRI CS Reconstruction Start")

with torch.no_grad():
    for img_no in range(ImgNum):

        imgName = filepaths[img_no]

        Iorg = cv2.imread(imgName, 0)

        Icol = Iorg.reshape(1, 1, 256, 256) / 255.0

        Img_output = Icol

        start = time()

        batch_x = torch.from_numpy(Img_output)
        batch_x = batch_x.type(torch.FloatTensor)
        batch_x = batch_x.to(device)

        x_in_k_space = torch.fft.fft2(batch_x)
        masked_x_in_k_space = x_in_k_space * mask

        PhiTb = torch.fft.ifft2(masked_x_in_k_space)
        PhiTb = torch.view_as_real(PhiTb).squeeze(1).permute(0, 3, 1, 2)
        masked_x_in_k_space = torch.view_as_real(masked_x_in_k_space).squeeze(1).permute(0, 3, 1, 2)
        x_output = model(PhiTb, masked_x_in_k_space, mask)
        PhiTb = complex_abs(PhiTb)
        x_output = complex_abs(x_output)

        end = time()
        initial_result = PhiTb.cpu().data.numpy().reshape(256, 256)

        Prediction_value = x_output.cpu().data.numpy().reshape(256, 256)

        X_init = np.clip(initial_result, 0, 1).astype(np.float64)
        X_rec = np.clip(Prediction_value, 0, 1).astype(np.float64)

        init_PSNR = psnr(X_init * 255, Iorg.astype(np.float64))
        init_SSIM = ssim(X_init * 255, Iorg.astype(np.float64), data_range=255)

        rec_PSNR = psnr(X_rec*255., Iorg.astype(np.float64))
        rec_SSIM = ssim(X_rec*255., Iorg.astype(np.float64), data_range=255)

        print("[%02d/%02d] Run time for %s is %.4f, Initial  PSNR is %.2f, Initial  SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), init_PSNR, init_SSIM))
        print("[%02d/%02d] Run time for %s is %.4f, Proposed PSNR is %.2f, Proposed SSIM is %.4f" % (img_no, ImgNum, imgName, (end - start), rec_PSNR, rec_SSIM))

        # im_rec_rgb = np.clip(X_rec*255, 0, 255).astype(np.uint8)
        # resultName = imgName.replace(args.data_dir, args.result_dir)
        # cv2.imwrite("%s_DBD_Net_ratio_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio, rec_PSNR, rec_SSIM), im_rec_rgb)

        del x_output

        PSNR_All[0, img_no] = rec_PSNR
        SSIM_All[0, img_no] = rec_SSIM

        Init_PSNR_All[0, img_no] = init_PSNR
        Init_SSIM_All[0, img_no] = init_SSIM
print('\n')
init_data =   "CS ratio is %d, Avg Initial  PSNR/SSIM for %s is %.2f/%.4f" % (cs_ratio, args.test_name, np.mean(Init_PSNR_All), np.mean(Init_SSIM_All))
output_data = "CS ratio is %d, Avg Proposed PSNR/SSIM for %s is %.2f/%.4f, Epoch number of model is %d \n" % (cs_ratio, args.test_name, np.mean(PSNR_All), np.mean(SSIM_All), epoch_num)
print(init_data)
print(output_data)

output_file_name = "./%s/PSNR_SSIM_Results_MRI_CS_DBH_Net_layer_%d_group_%d_ratio_%d.txt" % (args.log_dir, layer_num, group_num, cs_ratio)

output_file = open(output_file_name, 'a')
output_file.write(output_data)
output_file.close()

print("MRI CS Reconstruction End")