import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--best_AtoB_path', type=str)
parser.add_argument('--best_BtoA_path', type=str)
parser.add_argument('--target_path', type=str)
opt = parser.parse_args()

best_AtoB_model = torch.load(opt.best_AtoB_path, map_location='cpu')
best_BtoA_model = torch.load(opt.best_BtoA_path, map_location='cpu')

best_ckpt = {
    'G_A': best_AtoB_model['G_A'],
    'G_B': best_BtoA_model['G_B'],
    'D_A': best_AtoB_model['D_A'],
    'D_B': best_BtoA_model['D_B'],
    'fid': (best_AtoB_model['fid'][0], best_BtoA_model['fid'][1])
}
torch.save(best_ckpt, opt.target_path)
print('save done!')