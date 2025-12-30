import os
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from physical_consistency_losses import score_candidates

transform = transforms.ToTensor()

def get_img(filename, size):
    image = Image.open(filename)
    image = image.resize(size, Image.Resampling.LANCZOS)
    tensor_image = transform(image)
    return tensor_image

org_dir = './org'
img_dir = './Qwen'
out_fused_dir = os.path.join(img_dir, 'results', 'fused')
out_best_dir = os.path.join(img_dir, 'results', 'best')
if not os.path.exists(out_fused_dir):
    os.makedirs(out_fused_dir)
if not os.path.exists(out_best_dir):
    os.makedirs(out_best_dir)

types = ['normal', 'constrast', 'color', 'denoise']
img_list = os.listdir(org_dir)
for img in img_list:
    org_img = Image.open(os.path.join(org_dir, img))
    size = org_img.size
    org_tensor = transform(org_img)
    candidate_files = [os.path.join(img_dir, t, img) for t in types]
    candidates = [get_img(f, size) for f in candidate_files]
    results, best_idx = score_candidates(org_tensor, candidates)

    totals = torch.tensor([r['total'] for r in results])
    weights = F.softmax((-totals) / 0.03, dim=0)
    fused = sum(w * cand for w, cand in zip(weights, candidates))
    fused_to_save = fused.cpu()
    out_file = os.path.join(out_fused_dir, img)
    save_image(fused_to_save, out_file)

    best = candidates[best_idx]
    print("scores:", [r['total'] for r in results])
    best_to_save = best.cpu()
    out_file = os.path.join(out_best_dir, img)
    save_image(best_to_save, out_file)
