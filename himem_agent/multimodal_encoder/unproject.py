import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

def unproject(intrinsics, poses, depths):
    """
    Inputs:
        intrinsics: B X V X 3 X 3
        poses: B X V X 4 X 4 (torch.tensor)
        depths: B X V X H X W (torch.tensor)
    
    Outputs:
        world_coords: B X V X H X W X 3
    """
    # (B, V, 336, 336)
    B, V, H, W = depths.shape 
    # (B, V, 1)
    fx, fy, px, py = intrinsics[..., 0, 0][..., None], intrinsics[..., 1, 1][..., None], intrinsics[..., 0, 2][..., None], intrinsics[..., 1, 2][..., None]

    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    y = y[None, None].repeat(B, V, 1, 1).flatten(2)  # (B, V, H*W)
    z = depths.flatten(2)  # (B, V, H*W)
    x = (x - px) * z / fx
    y = (y - py) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)

    world_coords = (poses @ cam_coords.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    world_coords = world_coords[..., :3] / world_coords[..., 3][..., None]

    world_coords = world_coords.reshape(B, V, H, W, 3)

    return world_coords

def backproject_depth(depths, poses, intrinsics=None):
    B, V, H, W = depths.shape
    xyz = unproject(intrinsics, poses, depths)
    return xyz  # (B X V X H X W X 3)

def interpolate_depth(xyz, multi_scale_features, method="nearest"):
    multi_scale_xyz = []
    B, V, H, W, _ = xyz.shape
    for feat in multi_scale_features:
        h, w = feat.shape[2:]
        xyz_ = torch.nn.functional.interpolate(
            xyz.reshape(B*V, H, W, 3).permute(0, 3, 1, 2), size=(h, w),
            mode=method).permute(0, 2, 3, 1).reshape(B, V, h, w, 3)
        multi_scale_xyz.append(xyz_)
    return multi_scale_xyz



def project_depth_v2(depth_img:torch.Tensor, pos:torch.Tensor, quat_wb:torch.Tensor, fov:torch.Tensor):
    d_H, d_W = depth_img.shape[0], depth_img.shape[1]
    # 计算相机内参
    fov_rad = torch.deg2rad(torch.tensor(fov, device=depth_img.device))
    fx = (d_W * 0.5) / torch.tan(fov_rad * 0.5)
    cx = d_W / 2.0
    cy = d_H / 2.0
    
    # 转换旋转矩阵
    r_wb = R.from_quat(quat_wb.cpu().numpy()).as_matrix()
    r_wb = torch.from_numpy(r_wb).float().to(depth_img.device)
    
    # 世界坐标位置 (3, 1)
    world_pos = pos.reshape(3, 1)
    
    # 创建网格坐标
    v_coords, u_coords = torch.meshgrid(
        torch.arange(d_H, device=depth_img.device),
        torch.arange(d_W, device=depth_img.device),
        indexing='ij'
    )
    
    # 展平所有坐标
    d = depth_img.reshape(-1)
    v = v_coords.reshape(-1)
    u = u_coords.reshape(-1)
    
    # 过滤有效点
    d_valid = d
    v_valid = v
    u_valid = u
    
    # 向量化计算3D坐标
    x = d_valid
    y = (u_valid.float() - cx) * d_valid / fx
    z = (v_valid.float() - cy) * d_valid / fx
    
    # 构建点云 (3, N)
    pointcloud = torch.stack([x, y, z], dim=0)
    
    # 转换到世界坐标系
    pointcloud = (r_wb @ pointcloud + world_pos).T
    
    # 像素坐标
    # pixel_cords = torch.stack([v_valid, u_valid], dim=1).long()
    
    return pointcloud.float()

def project_depth_v2_batch(depth_img: torch.Tensor, pos: torch.Tensor, quat_wb: torch.Tensor, fov: torch.Tensor,max_depth: float=None):
    """
    批处理版本的深度投影
    Args:
        depth_img: (B, H, W) 深度图
        pos: (B, 3) 位置
        quat_wb: (B, 4) 四元数
        fov: (B,) 视场角
    Returns:
        pointclouds: (B, H, W, 3) 点云
    """
    B, d_H, d_W = depth_img.shape
    device = depth_img.device
    # 计算相机内参 (B,)
    fov_rad = torch.deg2rad(fov)
    fx = (d_W * 0.5) / torch.tan(fov_rad * 0.5)  # (B,)
    cx = d_W / 2.0
    cy = d_H / 2.0
    
    # 批量转换四元数到旋转矩阵 (B, 3, 3)
    # 注意：scipy的quat顺序是 (x, y, z, w)，根据你的数据格式可能需要调整
    r_wb_list = []
    for i in range(B):
        r = R.from_quat(quat_wb[i].cpu().numpy()).as_matrix()
        r_wb_list.append(torch.from_numpy(r).float())
    r_wb = torch.stack(r_wb_list, dim=0).to(device)  # (B, 3, 3)
    
    # 创建网格坐标
    v_coords, u_coords = torch.meshgrid(
        torch.arange(d_H, device=device),
        torch.arange(d_W, device=device),
        indexing='ij'
    )
    
    # 展平深度图 (B, H*W)
    d = depth_img.reshape(B, -1)
    v = v_coords.reshape(-1).unsqueeze(0).expand(B, -1)  # (B, H*W)
    u = u_coords.reshape(-1).unsqueeze(0).expand(B, -1)  # (B, H*W)

    
    # 向量化计算3D相机坐标 (B, H*W)
    x = d
    y = (u.float() - cx) * d / fx
    z = (v.float() - cy) * d / fx
    
    
    # 构建点云 (B, 3, H*W)
    pointcloud = torch.stack([x, y, z], dim=1)
    
    # 结果: (B, 3, H*W)
    pointcloud_world = torch.bmm(r_wb, pointcloud.float()) + pos.unsqueeze(2)
    # 转置并reshape (B, H*W, 3) -> (B, H, W, 3)
    pointcloud_world = pointcloud_world.permute(0, 2, 1)
    
    pixel_coords = torch.stack([v, u], dim=2).long()  # (B, H*W, 2)
    
    if max_depth is not None:
        # 创建有效掩码（向量化的条件判断）
        valid_mask = ~torch.isnan(d) & (d > 0.001) & (d <= max_depth)
        return pointcloud_world.float(),valid_mask

    return pointcloud_world.float()


def create_points(depth_img, params, poses):
    (B, V, H, W) = depth_img.shape
    depth_imgs = depth_img.flatten(0,1)  # (B*V, H, W)
    poses = poses.flatten(0,1)  # (B*V, H, W)
    quat_wbs = params[:,:,:4].flatten(0,1)  # (B*V, H, W)
    fovs = params[:,:,4:].flatten(0,1)  # (B*V, H, W)
    
    # pointclouds = torch.stack([project_depth_v2(d_img, p, q, f) for d_img,p,q,f in zip(depth_imgs, poses, quat_wbs, fovs)])
    pointclouds = project_depth_v2_batch(depth_imgs, poses, quat_wbs, fovs,)
    # pointclouds,valid_masks = project_depth_v2_batch(depth_imgs, poses, quat_wbs, fovs,max_depth=500)
    
    pointclouds = pointclouds.reshape(B,V,H,W,3)

    return pointclouds

# def backprojector_dataloader(
#     multi_scale_features, depths, poses,
#     intrinsics=None, method='nearest', 
#     padding=None):
def backprojector_dataloader(
    multi_scale_features,
    depths,
    params,
    poses,
    method='nearest', 
    padding=None):
    """
    Inputs:
        multi_scale_features: list
            [B*V, 1024, 24, 24], [B*V, 1024, 48, 48], [B*V, 1024, 96, 96]
        depths: tensor [B, 5, 336, 336]
        poses: tensor [B, 5, 4, 4]
        intrinsics: tensor [B, 5, 4, 4]

    Outputs:
        list: []
            B, V, H, W, 3
    """
    
    # (B, V, H, W, 3)
    # new_xyz = backproject_depth(depths, poses, intrinsics)
    st_time = time.time()
    new_xyz = create_points(depths, params, poses)
    save_xyzs = new_xyz.flatten(0,1)
    save_xyzs = save_xyzs.cpu().numpy()
    
    if padding is not None:
        new_xyz = F.pad(new_xyz.permute(0, 1, 4, 2, 3), (0, padding[1], 0, padding[0]), mode='constant', value=0).permute(0, 1, 3, 4, 2)

    multi_scale_xyz = interpolate_depth(new_xyz, multi_scale_features,method=method)
    if len(multi_scale_xyz) == 1:
        multi_scale_xyz = multi_scale_xyz[0]
    
    return multi_scale_xyz, new_xyz

def voxelize(xyz, voxel_size=0.28):
    """
    Inputs: 
        xyz: list of tensors [B, V, H, W, 3]
        voxel_size: voxel size

    Outputs:
        N=V*H*W
        p2v: tensors [B, N]
    """
    B, V, H, W, _ = xyz.shape
    xyz = xyz.reshape(B, V*H*W, 3)
    p2v = voxelization(xyz, voxel_size)
    return p2v

def voxelize_v2(xyz, voxel_size=0.28):
    """
    Inputs: 
        xyz: list of tensors [B, V, H, W, 3]
        voxel_size: voxel size

    Outputs:
        N=V*H*W
        p2v: tensors [B, N]
    """
    if len(xyz.shape) == 2:
        xyz = xyz.unsqueeze(0)
    p2v = voxelization(xyz, voxel_size)
    return p2v

def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert len(arr.shape) == 3
    arr -= arr.min(1, keepdims=True)[0].to(torch.long)
    arr_max = arr.max(1, keepdims=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=torch.long).to(arr.device)

    # Fortran style indexing
    for j in range(arr.shape[2] - 1):
        keys += arr[..., j]
        keys *= arr_max[..., j + 1]
    keys += arr[..., -1]
    return keys

def voxelization(xyz, voxel_size):
    """
    Inputs:
        xyz: tensor [B, N, 3]
        voxel_size: float
    Outputs: 
        point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
    """
    B, N, _ = xyz.shape
    xyz = xyz / voxel_size
    xyz = torch.round(xyz).long()
    xyz = xyz - xyz.min(1, keepdim=True)[0]

    keys = ravel_hash_vec(xyz)

    point_to_voxel = torch.stack(
        [torch.unique(keys[b], return_inverse=True)[1] for b in range(B)], 0)
    return point_to_voxel

def voxel_map_to_source(voxel_map, poin2voxel):
    """
    Input:
        voxel_map (B, N1, C)
        point2voxel (B, N)
    Output:
        src_new (B, N, C)
    """
    bs, n, c = voxel_map.shape
    src_new = torch.stack([voxel_map[i, poin2voxel[i]] for i in range(bs)])
    return src_new
