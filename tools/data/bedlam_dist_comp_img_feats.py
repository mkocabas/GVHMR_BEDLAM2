import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import glob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import argparse
from typing import List
import math

# Reusing your existing imports
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from hmr4d.utils.preproc import Extractor
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

DEBUG = False


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch

class SimpleDataset(Dataset):
    def __init__(self, data, seq_file, images_folder, prefix, split):
        self.images_folder = images_folder
        self.seq_file = seq_file
        self.seq_folder = seq_file.split('/')[-2]
        self.seq_name = seq_file.split('/')[-1].split('-')[0]
        self.save_path = seq_file.replace(f'{split}_labels_30fps', f'imgfeats/{prefix}')
        
        # self.subject_id = seq_file.split('/')[-1].split('-')[1].replace('.pt', '')
        self.data = data
        for idx, imgname in enumerate(self.data['imgnames']):
            self.data['imgnames'][idx] = imgname.replace('png', 'jpg')

    def __len__(self):
        try:
            return len(self.data['imgnames'])
        except:
            print(f'{self.seq_file} is corrupted')
            print(self.data.keys())
            return 0
    
    def __getitem__(self, idx):
        imgname = self.data['imgnames'][idx]
        imgname = os.path.join(self.images_folder, self.seq_folder, 'jpg', imgname)
        bbox = self.data['bboxes'][idx:idx+1]
        cam_int = self.data['cam_int'][idx:idx+1]
        image_cv = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
        boxes = torch.from_numpy(bbox).float()
        boxes = torch.cat([boxes, torch.ones_like(boxes)[...,:1]], dim=-1)
        item = {
            'imgname': imgname,
            'boxes': boxes,
            'cam_int': torch.from_numpy(cam_int).float(),
            'image_cv': image_cv,
            'seq_file': self.seq_file,
            'save_path': self.save_path,
            'last_item': True if idx == len(self.data['imgnames']) - 1 else False,
            'first_item': True if idx == 0 else False,
            'seqlen': len(self.data['imgnames']),
            'kpts': None,
            # 'subject_id': self.subject_id,
            # 'seq_folder': self.seq_folder,
            # 'seq_name': self.seq_name,
        }
        item = pad_image(item, IMG_SIZE=896)
        item['image'] = normalization(item['image_cv'])
        item['image_cv'] = torch.tensor(item['image_cv'])
        return item
    

def setup(rank, world_size, args):
    """Initialize distributed process group across multiple nodes."""
    # Add debug logs to check environment and arguments
    print(f"[Debug] Setup called with rank={rank}, world_size={world_size}")
    print(f"[Debug] dist_url={args.dist_url}")
    print(f"[Debug] MASTER_ADDR env var: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"[Debug] MASTER_PORT env var: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    
    # Always set the environment variables regardless of dist_url
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    print(f"[Debug] Set environment variables: MASTER_ADDR={args.master_addr}, MASTER_PORT={args.master_port}")
    
    # Initialize process group with appropriate backend
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Process group initialized. Rank: {rank}, World size: {world_size}")

def cleanup():
    dist.destroy_process_group()

def get_seq_files_for_rank(seq_files: List[str], rank: int, world_size: int):
    """Distribute sequence files across ranks."""
    num_files = len(seq_files)
    files_per_rank = math.ceil(num_files / world_size)
    start_idx = rank * files_per_rank
    end_idx = min(start_idx + files_per_rank, num_files)
    return seq_files[start_idx:end_idx]


@torch.no_grad()
def process_rank(local_rank, world_size, args, seq_files, prefix):
    # Calculate global rank based on node rank and local rank
    global_rank = args.node_rank * args.gpus_per_node + local_rank
    print(f"[Debug] CUDA available: {torch.cuda.is_available()}")
    print(f"[Debug] CUDA device count: {torch.cuda.device_count()}")
    print(f"[Debug] Local rank: {local_rank}, Global rank: {global_rank}")
    print(f"[Debug] World size: {world_size}")
    
    if local_rank >= torch.cuda.device_count():
        print(f"[ERROR] Local rank {local_rank} exceeds available CUDA devices ({torch.cuda.device_count()})")
        return  # Early return to avoid crashing
    
    setup(global_rank, world_size, args)
    
    # Set device for this process - use local rank for device selection
    torch.cuda.set_device(local_rank)
    
    print(f'Loading HMR2 model...')
    model = Extractor()
    model.extractor = DDP(model.extractor, device_ids=[local_rank])
    model.extractor.compile()
    print(f'Loaded HMR2 model...')
    
    # Get sequence files for this rank - use global_rank
    rank_seq_files = get_seq_files_for_rank(seq_files, global_rank, world_size)

    for seq_file in tqdm(rank_seq_files, desc=f'Rank {global_rank}'):
        data = torch.load(seq_file)
        
        if data is None:
            print(f'{seq_file} is corrupted')
            with open('.tmp/corrupted_seq_files.txt', 'a') as f:
                f.write(f'{seq_file}\n')
            continue
        
        save_path = seq_file.replace(f'{args.split}_labels_30fps', f'imgfeats/{prefix}')
        
        bbx_xyxy = data['bboxes']
        bbx_xys = get_bbx_xys_from_xyxy(torch.from_numpy(bbx_xyxy), base_enlarge=1.0)
        videos_base_dir = 'videos' if args.split == 'training' else 'videos_test'
        video_path = '/'.join(seq_file.replace(f'{args.split}_labels_30fps', videos_base_dir).split('/')[:-1])
        video_path = os.path.join(video_path, 'mp4', seq_file.split('/')[-1].split('-')[0] + '.mp4')

        vit_features = model.extract_video_features(video_path, bbx_xys)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(vit_features, save_path)
        
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--cfg', type=str, default='configs/config_fhmr_170.yaml')
    parser.add_argument('--dataset', type=str, default='bedlam1')
    parser.add_argument('--prefix', type=str, default='hmr2')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--split', type=str, default='training', choices=['training', 'test'])
    # Add multi-node arguments
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes to use for distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of the current node')
    parser.add_argument('--gpus_per_node', type=int, default=8, help='Number of GPUs per node')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master node address')
    parser.add_argument('--master_port', type=str, default='23456', help='Master node port')
    parser.add_argument('--dist_url', type=str, default='env://', help='URL used to set up distributed training')
    args = parser.parse_args()

    prefix = args.prefix
    if args.dataset == 'bedlam1':
        args.out_dir = f'/home/muhammed/projects/GVHMR/inputs/BEDLAM1/imgfeats/{prefix}'
        args.inp_dir = '/home/muhammed/projects/GVHMR/inputs/BEDLAM1/training_labels_30fps/'
        args.images_folder = '/home/data/datasets/BEDLAM/images/'
    elif args.dataset == 'bedlam2':
        args.out_dir = f'/home/muhammed/projects/GVHMR/inputs/BEDLAM2/imgfeats/{prefix}'
        args.inp_dir = f'/home/muhammed/projects/GVHMR/inputs/BEDLAM2/{args.split}_labels_30fps/'
        if args.split == 'training':
            args.images_folder = '/home/data/datasets/BEDLAM2_0/images/'
        else:
            args.images_folder = '/home/data/datasets/BEDLAM2_0/images_test/'
        
    print(args)

    # Get all sequence files
    seq_files = sorted(glob.glob(f'{args.inp_dir}/*/*.pt'))
    remaining_seq_files = []
    for seq_file in seq_files:
        if not os.path.exists(seq_file.replace(f'{args.split}_labels_30fps', f'imgfeats/{prefix}')):
            remaining_seq_files.append(seq_file)
    seq_files = remaining_seq_files
    print(f'Found {len(seq_files)} sequences')
    
    # Calculate world_size properly for multi-node setup
    world_size = args.nodes * args.gpus_per_node
    
    if args.num_gpus > 1:
        # Launch processes for this node only
        mp.spawn(
            process_rank,
            args=(world_size, args, seq_files, prefix),
            nprocs=args.gpus_per_node,  # Only spawn processes for this node
            join=True
        )
    else:
        # For single GPU case, use node_rank to get global rank
        global_rank = args.node_rank * args.gpus_per_node
        process_rank(0, world_size, args, seq_files, prefix)

if __name__ == "__main__":
    main()