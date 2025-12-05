import os
import torch
import numpy as np
import random
from smplcodec.codec import SMPLCodec


def main():
    gvhmr_data = torch.load("inputs/BEDLAM/hmr4d_support/smplpose_v2.pth")
    all_seqs = list(gvhmr_data.keys())
    
    random.shuffle(all_seqs)
    tid = 0
    for seq in all_seqs:
        print(seq)
        seq_name = seq.split("/")[-3]
        seq_id = seq.split('/')[-1].split('-')[0].split('/')[-1].split('.')[0]
        subject = seq.split("-")[-1]
        bedlam_data = np.load(f"/mnt/data/datasets/BEDLAM/data_30fps/bedlam_labels_30fps/{seq_name}.npz")
        bedlam_subj = bedlam_data["sub"]
        
        th2np = lambda x: x.cpu().numpy()
        gvhmr_pack = gvhmr_data[seq]
        gvhmr_pose = th2np(gvhmr_pack["pose"])
        gvhmr_trans = th2np(gvhmr_pack["trans"])
        gvhmr_betas = th2np(gvhmr_pack["beta"])
        gvhmr_trans_cam = th2np(gvhmr_pack["trans_incam"])
        gvhmr_go_cam = th2np(gvhmr_pack["global_orient_incam"])
        gvhmr_cam_ext = th2np(gvhmr_pack["cam_ext"])
        gvhmr_cam_int = th2np(gvhmr_pack["cam_int"])
        
        idxs = []
        for im_idx, im_name in enumerate(bedlam_data['imgname']):
            if im_name.startswith(seq_id):
                if bedlam_subj[im_idx] == subject:
                    idxs.append(im_idx)

        bedlam_pose = bedlam_data["pose_world"][idxs][:, :66]
        bedlam_trans = bedlam_data["trans_world"][idxs]
        bedlam_betas = bedlam_data["shape"][idxs]
        bedlam_cam_ext = bedlam_data["cam_ext"][idxs]
        bedlam_cam_int = bedlam_data["cam_int"][idxs]
        bedlam_pose_cam = bedlam_data["pose_cam"][idxs][:, :66]
        bedlam_trans_cam = bedlam_data["trans_cam"][idxs]
        imgnames = bedlam_data['imgname'][idxs]
        from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
        # rotate bedlam_pose[:, :3] (axis angle) and bedlam_trans around X axis 180 degrees
        R = axis_angle_to_matrix(torch.tensor(bedlam_pose[:, :3]))
        R = R @ torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])[None].float()
        bedlam_pose[:, :3] = matrix_to_axis_angle(R).numpy()
        # bedlam_trans = (bedlam_trans[:, None, :] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])[None])[:, 0]

        
        print('GVHMR shape', gvhmr_pose.shape)
        print('BEDLAM shape', bedlam_pose.shape)
        if gvhmr_pose.shape[0] != bedlam_pose.shape[0]:
            print(f"Mismatch in {seq}")
            continue
        
        os.makedirs('.tmp', exist_ok=True)
        smpl_file_bedlam = f'.tmp/{tid:04d}_bedlam.smpl'
        SMPLCodec(
            shape_parameters=bedlam_betas[0, :10],
            body_pose=bedlam_pose.reshape(-1,22,3), 
            body_translation=bedlam_trans,
            frame_count=bedlam_pose.shape[0], 
            frame_rate=30.0,
        ).write(smpl_file_bedlam)
        
        smpl_file_gvhmr = f'.tmp/{tid:04d}_gvhmr.smpl'
        SMPLCodec(
            shape_parameters=gvhmr_betas,
            body_pose=gvhmr_pose.reshape(-1,22,3), 
            body_translation=gvhmr_trans,
            frame_count=gvhmr_pose.shape[0], 
            frame_rate=30.0,
        ).write(smpl_file_gvhmr)
        
        tid += 1
        
        import ipdb; ipdb.set_trace()
        

if __name__ == "__main__":
    main()