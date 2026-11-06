
import torch
import torch.nn as nn
import time
from .multimodal_encoder.spatial_aware_module import SpatialAwareModule
from .multimodal_encoder.unproject import backprojector_dataloader, voxelize,voxelize_v2, create_points, interpolate_depth
from .multimodal_encoder.oct_utils import build_octree_boxes_with_points,build_octree_boxes_with_points_v2,filter_boxes_by_distance,filter_boxes_by_distance_v2,build_point_to_leaf_mapping,build_octree_v36_compiled
from torch_scatter import scatter_mean

def cartesian_to_spherical(feat_xyz, cur_pose):
    N, V, H, W, _ = feat_xyz.shape
    # 确保 cur_pose 形状正确
    if cur_pose.dim() == 1:
        cur_pose = cur_pose.unsqueeze(0)  # (1, 3)
    cur_pose = cur_pose.view(N, 1, 1, 1, 3)  # (N, 1, 1, 1, 3)
    
    # 计算相对坐标
    rel_xyz = feat_xyz - cur_pose  # (N, V, H, W, 3)
    x, y, z = rel_xyz[..., 0], rel_xyz[..., 1], rel_xyz[..., 2]
    # 计算极坐标
    # r: 径向距离
    r = torch.sqrt(x**2 + y**2 + z**2)  # (N, V, H, W)
    
    # phi: 方位角 (azimuth), 范围 [0, 2π] 或 [-π, π]
    phi = torch.atan2(y, x)  # (N, V, H, W)
    
    # theta: 极角/仰角 (elevation), 范围 [0, π]
    # theta = 0 表示正上方(+z), theta = π 表示正下方(-z)
    theta = torch.acos(torch.clamp(z / (r + 1e-8), -1.0, 1.0))  # (N, V, H, W)
    
    # 拼接原始坐标和极坐标
    combined_feat = torch.stack([
        x, y, z,      # 笛卡尔相对坐标
        r,            # 径向距离
        phi,          # 方位角
        theta         # 极角
    ], dim=-1)  # (N, V, H, W, 6)
    
    return combined_feat


class RGBDSceneTower(nn.Module):
    def __init__(self,latent_dim=1024, video_tower='SpatialAwareModule',voxel_size=0.2, args=None, delay_load=False):
        super().__init__()
        self.is_loaded = False
        # self.pooling = 'voxelize'
        self.pooling = 'hira_vox'
        # self.pooling = 'hira_oct'
        # self.pooling = 'target_oct'
        self.voxel_size = voxel_size
        self.video_tower_name = video_tower
        self.latent_dim = latent_dim
        self.oct_levels = 6
        
        self.pe_dim = 6
        self.oct_distance = 50
        self.max_distance=  500
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule(dim=self.pe_dim,latent_dim=self.latent_dim)
        else:
            raise NotImplementedError
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        
    def get_valid_mask(self,feat_points,traj_poses,valid_dis=50):
        points_expanded = feat_points.unsqueeze(1)  # [N, 1, 3]
        traj_poses_expanded = traj_poses.unsqueeze(0)  # [1, M, 3]
        region_valid_distances = torch.norm(points_expanded - traj_poses_expanded, dim=2)  # [N, M]
        region_mask = (region_valid_distances <= valid_dis).any(dim=1)  # [N]
        return region_mask
    
    def encode_features_with_xyz(self,features_xyz):
        # features_xyz [L,C+pe_dim]
        features, feature_xyz = torch.split(features_xyz, [self.latent_dim, self.pe_dim], dim=1)
        pos_embed = self.video_tower.encode_pe(feature_xyz)
        features_pe = features + pos_embed
        return features_pe
    
    def unproject_current(self, features, depths, params, poses):
        """Return current-frame features/world positions/validity, without additive PE."""
        B, V, C, H, W = features.shape
        if V != 1:
            raise ValueError("Memory accepts one new frame per environment step, not replayed history")
        with torch.no_grad(), torch.autocast(device_type=features.device.type, enabled=False):
            world = create_points(depths.float(), params.float(), poses.float())
            xyz = interpolate_depth(world, [features.flatten(0, 1)], method="nearest")[0]
            patch_depth = torch.nn.functional.interpolate(
                depths.flatten(0, 1).unsqueeze(1).float(), size=(H, W), mode="nearest"
            ).reshape(B, -1)
            xyz = xyz.reshape(B, -1, 3)
            distance = torch.linalg.vector_norm(xyz - poses[:, -1:, :].float(), dim=-1)
            valid = torch.isfinite(xyz).all(-1) & torch.isfinite(patch_depth)
            valid = valid & (patch_depth > 0) & (distance <= self.max_distance)
        visual = features.permute(0, 1, 3, 4, 2).reshape(B, -1, C)
        return visual, xyz, valid

    def get_pooled_features_with_xyz(self,features,depths,params, poses,valid_region_dis=50):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images
        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        # assert intrinsics.dim() == 4
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths,params, poses)
        cur_poses = poses[:,-1]
        rel_feat_xyz = cartesian_to_spherical(feat_xyz,cur_poses)
        # video_features = self.video_tower([features.flatten(0, 1)], [rel_feat_xyz.flatten(0, 1).to(features.dtype)], (B, V))[0]
        
        features = features.reshape(B, V, C, H, W).permute(0, 1, 3, 4, 2)
        video_features = torch.cat([features,rel_feat_xyz.to(features.dtype)],dim=-1)
        video_features = video_features.flatten(1, 3)
        
        # if lengths is not None:
        #     lengths = lengths*H*W
        b_near_pooled_video_features = None
        b_far_pooled_video_features = None
    
        if self.pooling  == 'voxelize':
            p2v = voxelize(feat_xyz, self.voxel_size)  # （B, N)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(len(video_features))]) # bn, F
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
        elif self.pooling == 'hira_vox':
            pooled_video_features = []
            b_near_pooled_video_features = []
            b_far_pooled_video_features = []
            
            batch_offset = []
            batch_feat_points = feat_xyz.reshape(B, V*H*W, 3)

            for b in range(len(video_features)):
                cur_pose = poses[b][-1]
                traj_poses = poses[b]
                
                # xyz_distances = torch.norm(batch_feat_points[b] - cur_pose, dim=1)
                # valid_indices = torch.where(xyz_distances <= self.max_distance)[0]
                valid_mask = self.get_valid_mask(batch_feat_points[b],traj_poses,valid_dis=self.max_distance)
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    valid_feat_points = batch_feat_points[b][valid_indices]
                    valid_video_features = video_features[b][valid_indices]

                    region_mask = self.get_valid_mask(valid_feat_points,traj_poses,valid_dis=valid_region_dis)

                    near_indices = torch.where(region_mask)[0]
                    far_indices = torch.where(~region_mask)[0]
                    
                    # 提取近距离的points和features
                    if len(near_indices) > 0:
                        near_feat_points = valid_feat_points[near_indices]
                        near_video_features = valid_video_features[near_indices]
                    else:
                        near_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        near_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)
                    
                    if len(far_indices) > 0:
                        far_feat_points = valid_feat_points[far_indices]
                        far_video_features = valid_video_features[far_indices]
                    else:
                        far_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        far_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)  

                    st_time = time.time()
                    if near_feat_points.shape[0] > 0:
                        new_near_feat_points =near_feat_points.unsqueeze(0)
                        new_near_video_features =near_video_features.unsqueeze(0)
                        st_time = time.time()
                        near_p2v = voxelize_v2(new_near_feat_points, self.voxel_size)  # （B, N)
                        
                        near_pooled_video_features = torch.cat([scatter_mean(new_near_video_features[b], near_p2v[b], dim=0) for b in range(len(new_near_video_features))]) # bn, F
                        near_batch_offset = ((near_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        near_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device)
 
                    if far_feat_points.shape[0] > 0:
                        new_far_feat_points =far_feat_points.unsqueeze(0)
                        new_far_video_features =far_video_features.unsqueeze(0)
                        st_time = time.time()
                        far_p2v = voxelize_v2(new_far_feat_points, self.voxel_size*5)  # （B, N)
                        far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features[b], far_p2v[b], dim=0) for b in range(len(new_far_video_features))]) # bn, F
                        far_batch_offset = ((far_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        far_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device)
                        

                        
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    cur_pooled_video_features = torch.cat([near_pooled_video_features,far_pooled_video_features])
                    
                    
                    cur_batch_offset = torch.tensor(
                                        [cur_pooled_video_features.shape[0]], 
                                        dtype=torch.int32, 
                                        device=cur_pooled_video_features.device
                                )
                    pooled_video_features.append(cur_pooled_video_features)
                    batch_offset.append(cur_batch_offset)
                else:
                    valid_feat_points = batch_feat_points[b]
                    valid_video_features = video_features[b]
                    # random 64 tokens
                    indices = torch.randperm(valid_video_features.size(0))[:64]
                    
                    empty_features = valid_video_features[indices]
                    
                    pooled_video_features.append(empty_features)
                    
                    batch_offset.append(torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device))
                    
                    near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    
            b_near_pooled_video_features = torch.cat(b_near_pooled_video_features,dim=0)
            b_far_pooled_video_features = torch.cat(b_far_pooled_video_features,dim=0)
            
            pooled_video_features = torch.cat(pooled_video_features,dim=0)
            batch_offset = torch.cat(batch_offset,dim=0)
        elif self.pooling == 'hira_oct':
            
            pooled_video_features = []
            b_near_pooled_video_features = []
            b_far_pooled_video_features = []
            
            batch_offset = []
            batch_feat_points = feat_xyz.reshape(B, V*H*W, 3)
            
            for b in range(len(video_features)):
                traj_poses = poses[b]
                
                if len(traj_poses) > 8:
                    traj_poses = traj_poses[-8:]
                
                valid_mask = self.get_valid_mask(batch_feat_points[b],traj_poses,valid_dis=self.max_distance)
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    valid_feat_points = batch_feat_points[b][valid_indices]
                    valid_video_features = video_features[b][valid_indices]
                    
                    region_mask = self.get_valid_mask(valid_feat_points,traj_poses,valid_dis=valid_region_dis)

                    near_indices = torch.where(region_mask)[0]
                    far_indices = torch.where(~region_mask)[0]
                    
                    # 提取近距离的points和features
                    if len(near_indices) > 0:
                        near_feat_points = valid_feat_points[near_indices]
                        near_video_features = valid_video_features[near_indices]
                    else:
                        near_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        near_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)
                    
                    if len(far_indices) > 0:
                        far_feat_points = valid_feat_points[far_indices]
                        far_video_features = valid_video_features[far_indices]
                    else:
                        far_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        far_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)  

                    st_time = time.time()
                    if near_feat_points.shape[0] > 0:
                        new_near_feat_points =near_feat_points
                        new_near_video_features =near_video_features
                        st_time = time.time()
                        # near_p2v = voxelize_v2(new_near_feat_points, self.voxel_size)  # （B, N)
                        near_leaf_boxes, (near_leaf_point_indices_flat, near_leaf_point_counts) = build_octree_v36_compiled(new_near_feat_points.cpu(), max_level=6,min_voxel_size=self.voxel_size,max_points_per_node=4)
                        near_point_to_leaf = build_point_to_leaf_mapping((near_leaf_point_indices_flat, near_leaf_point_counts),new_near_feat_points.shape[0])
                        near_point_to_leaf = near_point_to_leaf.to(new_near_video_features.device)
                        near_pooled_video_features = torch.cat([scatter_mean(new_near_video_features, near_point_to_leaf, dim=0)]) # bn, F
                        

                        # near_batch_offset = ((near_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        # near_batch_offset = torch.zeros(1, dtype=torch.int32, 
                        #                             device=valid_video_features.device)
 
                    if far_feat_points.shape[0] > 0:
                        new_far_feat_points = far_feat_points
                        new_far_video_features =far_video_features
                        st_time = time.time()
                        # far_p2v = voxelize_v2(new_far_feat_points, self.voxel_size*5)  # （B, N)
                        # far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features[b], far_p2v[b], dim=0) for b in range(len(new_far_video_features))]) # bn, F

                        far_leaf_boxes, (far_leaf_point_indices_flat, far_leaf_point_counts) = build_octree_v36_compiled(new_far_feat_points.cpu(), max_level=4,min_voxel_size=self.voxel_size*5,max_points_per_node=8)
                        far_point_to_leaf = build_point_to_leaf_mapping((far_leaf_point_indices_flat, far_leaf_point_counts),new_far_feat_points.shape[0])
                        far_point_to_leaf = far_point_to_leaf.to(new_far_video_features.device)
                        far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features, far_point_to_leaf, dim=0)]) # bn, F
                        
                        
                        # far_batch_offset = ((far_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        # far_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    # device=valid_video_features.device)
                        
                        
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    cur_pooled_video_features = torch.cat([near_pooled_video_features,far_pooled_video_features])
                    
                    
                    cur_batch_offset = torch.tensor(
                                        [cur_pooled_video_features.shape[0]], 
                                        dtype=torch.int32, 
                                        device=cur_pooled_video_features.device
                                )
                    pooled_video_features.append(cur_pooled_video_features)
                    batch_offset.append(cur_batch_offset)
                else:
                    valid_feat_points = batch_feat_points[b]
                    valid_video_features = video_features[b]

                    empty_features = torch.empty(0, valid_video_features.shape[-1], 
                                  dtype=valid_video_features.dtype,
                                  device=valid_video_features.device)
                    pooled_video_features.append(empty_features)
                    
                    batch_offset.append(torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device))

                    near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
            b_near_pooled_video_features = torch.cat(b_near_pooled_video_features,dim=0)
            b_far_pooled_video_features = torch.cat(b_far_pooled_video_features,dim=0)
            
            pooled_video_features = torch.cat(pooled_video_features,dim=0)
            batch_offset = torch.cat(batch_offset,dim=0)
        else:
            raise NotImplementedError
        
        return pooled_video_features, batch_offset,b_near_pooled_video_features,b_far_pooled_video_features
    def forward(self, features, depths,params, poses, lengths=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        # assert intrinsics.dim() == 4
        # (B, V, 24, 24, 3)
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths,params, poses)
        cur_poses = poses[:,-1]
        rel_feat_xyz = cartesian_to_spherical(feat_xyz,cur_poses)
        video_features = self.video_tower([features.flatten(0, 1)], [rel_feat_xyz.flatten(0, 1).to(features.dtype)], (B, V))[0]
        # video_xyz = feat_xyz.reshape(B, V*H*W, 3)
        
        if lengths is not None:
            lengths = lengths*H*W

        b_near_pooled_video_features = None
        b_far_pooled_video_features = None
    
        if self.pooling  == 'voxelize':
            p2v = voxelize(feat_xyz, self.voxel_size)  # （B, N)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(len(video_features))]) # bn, F
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
        elif self.pooling == 'hira_vox':
            pooled_video_features = []
            b_near_pooled_video_features = []
            b_far_pooled_video_features = []
            
            batch_offset = []
            batch_feat_points = feat_xyz.reshape(B, V*H*W, 3)

            for b in range(len(video_features)):
                cur_pose = poses[b][-1]
                traj_poses = poses[b]
                
                # xyz_distances = torch.norm(batch_feat_points[b] - cur_pose, dim=1)
                # valid_indices = torch.where(xyz_distances <= self.max_distance)[0]

                valid_mask = self.get_valid_mask(batch_feat_points[b],traj_poses,valid_dis=self.max_distance)
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    valid_feat_points = batch_feat_points[b][valid_indices]
                    valid_video_features = video_features[b][valid_indices]
                    
                    # points_expanded = valid_feat_points.unsqueeze(1)  # [N, 1, 3]
                    # traj_poses_expanded = traj_poses.unsqueeze(0)  # [1, M, 3]
                    # region_valid_distances = torch.norm(points_expanded - traj_poses_expanded, dim=2)  # [N, M]
                    # region_mask = (region_valid_distances <= 50).any(dim=1)  # [N]
                    region_mask = self.get_valid_mask(valid_feat_points,traj_poses,valid_dis=50)

                    near_indices = torch.where(region_mask)[0]
                    far_indices = torch.where(~region_mask)[0]
                    
                    # 提取近距离的points和features
                    if len(near_indices) > 0:
                        near_feat_points = valid_feat_points[near_indices]
                        near_video_features = valid_video_features[near_indices]
                    else:
                        near_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        near_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)
                    
                    if len(far_indices) > 0:
                        far_feat_points = valid_feat_points[far_indices]
                        far_video_features = valid_video_features[far_indices]
                    else:
                        far_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        far_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)  

                    st_time = time.time()
                    if near_feat_points.shape[0] > 0:
                        new_near_feat_points =near_feat_points.unsqueeze(0)
                        new_near_video_features =near_video_features.unsqueeze(0)
                        st_time = time.time()
                        near_p2v = voxelize_v2(new_near_feat_points, self.voxel_size)  # （B, N)
                        
                        near_pooled_video_features = torch.cat([scatter_mean(new_near_video_features[b], near_p2v[b], dim=0) for b in range(len(new_near_video_features))]) # bn, F
                        near_batch_offset = ((near_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        near_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device)
 
                    if far_feat_points.shape[0] > 0:
                        new_far_feat_points =far_feat_points.unsqueeze(0)
                        new_far_video_features =far_video_features.unsqueeze(0)
                        st_time = time.time()
                        far_p2v = voxelize_v2(new_far_feat_points, self.voxel_size*5)  # （B, N)
                        far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features[b], far_p2v[b], dim=0) for b in range(len(new_far_video_features))]) # bn, F
                        far_batch_offset = ((far_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        far_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device)
                        

                        
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    cur_pooled_video_features = torch.cat([near_pooled_video_features,far_pooled_video_features])
                    
                    
                    cur_batch_offset = torch.tensor(
                                        [cur_pooled_video_features.shape[0]], 
                                        dtype=torch.int32, 
                                        device=cur_pooled_video_features.device
                                )
                    pooled_video_features.append(cur_pooled_video_features)
                    batch_offset.append(cur_batch_offset)
                else:
                    valid_feat_points = batch_feat_points[b]
                    valid_video_features = video_features[b]

                    empty_features = torch.empty(0, valid_video_features.shape[-1], 
                                  dtype=valid_video_features.dtype,
                                  device=valid_video_features.device)
                    pooled_video_features.append(empty_features)
                    
                    batch_offset.append(torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device))

                    near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    
            b_near_pooled_video_features = torch.cat(b_near_pooled_video_features,dim=0)
            b_far_pooled_video_features = torch.cat(b_far_pooled_video_features,dim=0)
            
            pooled_video_features = torch.cat(pooled_video_features,dim=0)
            batch_offset = torch.cat(batch_offset,dim=0)
        elif self.pooling == 'hira_oct':
            
            pooled_video_features = []
            b_near_pooled_video_features = []
            b_far_pooled_video_features = []
            
            batch_offset = []
            batch_feat_points = feat_xyz.reshape(B, V*H*W, 3)
            
            for b in range(len(video_features)):
                traj_poses = poses[b]
                
                if len(traj_poses) > 8:
                    traj_poses = traj_poses[-8:]
                
                valid_mask = self.get_valid_mask(batch_feat_points[b],traj_poses,valid_dis=self.max_distance)
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) > 0:
                    valid_feat_points = batch_feat_points[b][valid_indices]
                    valid_video_features = video_features[b][valid_indices]
                    
                    region_mask = self.get_valid_mask(valid_feat_points,traj_poses,valid_dis=50)

                    near_indices = torch.where(region_mask)[0]
                    far_indices = torch.where(~region_mask)[0]
                    
                    # 提取近距离的points和features
                    if len(near_indices) > 0:
                        near_feat_points = valid_feat_points[near_indices]
                        near_video_features = valid_video_features[near_indices]
                    else:
                        near_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        near_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)
                    
                    if len(far_indices) > 0:
                        far_feat_points = valid_feat_points[far_indices]
                        far_video_features = valid_video_features[far_indices]
                    else:
                        far_feat_points = torch.empty(0, valid_feat_points.shape[1], device=valid_feat_points.device)
                        far_video_features = torch.empty(0, valid_video_features.shape[1], device=valid_video_features.device)  

                    st_time = time.time()
                    if near_feat_points.shape[0] > 0:
                        new_near_feat_points =near_feat_points
                        new_near_video_features =near_video_features
                        st_time = time.time()
                        # near_p2v = voxelize_v2(new_near_feat_points, self.voxel_size)  # （B, N)
                        near_leaf_boxes, (near_leaf_point_indices_flat, near_leaf_point_counts) = build_octree_v36_compiled(new_near_feat_points.cpu(), max_level=6,min_voxel_size=self.voxel_size,max_points_per_node=4)
                        near_point_to_leaf = build_point_to_leaf_mapping((near_leaf_point_indices_flat, near_leaf_point_counts),new_near_feat_points.shape[0])
                        near_point_to_leaf = near_point_to_leaf.to(new_near_video_features.device)
                        near_pooled_video_features = torch.cat([scatter_mean(new_near_video_features, near_point_to_leaf, dim=0)]) # bn, F
                        

                        # near_batch_offset = ((near_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        # near_batch_offset = torch.zeros(1, dtype=torch.int32, 
                        #                             device=valid_video_features.device)
 
                    if far_feat_points.shape[0] > 0:
                        new_far_feat_points = far_feat_points
                        new_far_video_features =far_video_features
                        st_time = time.time()
                        # far_p2v = voxelize_v2(new_far_feat_points, self.voxel_size*5)  # （B, N)
                        # far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features[b], far_p2v[b], dim=0) for b in range(len(new_far_video_features))]) # bn, F

                        far_leaf_boxes, (far_leaf_point_indices_flat, far_leaf_point_counts) = build_octree_v36_compiled(new_far_feat_points.cpu(), max_level=4,min_voxel_size=self.voxel_size*5,max_points_per_node=8)
                        far_point_to_leaf = build_point_to_leaf_mapping((far_leaf_point_indices_flat, far_leaf_point_counts),new_far_feat_points.shape[0])
                        far_point_to_leaf = far_point_to_leaf.to(new_far_video_features.device)
                        far_pooled_video_features = torch.cat([scatter_mean(new_far_video_features, far_point_to_leaf, dim=0)]) # bn, F
                        
                        
                        # far_batch_offset = ((far_p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
                    else:
                        far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                        # far_batch_offset = torch.zeros(1, dtype=torch.int32, 
                                                    # device=valid_video_features.device)
                        
                        
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    cur_pooled_video_features = torch.cat([near_pooled_video_features,far_pooled_video_features])
                    
                    
                    cur_batch_offset = torch.tensor(
                                        [cur_pooled_video_features.shape[0]], 
                                        dtype=torch.int32, 
                                        device=cur_pooled_video_features.device
                                )
                    pooled_video_features.append(cur_pooled_video_features)
                    batch_offset.append(cur_batch_offset)
                else:
                    valid_feat_points = batch_feat_points[b]
                    valid_video_features = video_features[b]

                    empty_features = torch.empty(0, valid_video_features.shape[-1], 
                                  dtype=valid_video_features.dtype,
                                  device=valid_video_features.device)
                    pooled_video_features.append(empty_features)
                    
                    batch_offset.append(torch.zeros(1, dtype=torch.int32, 
                                                    device=valid_video_features.device))

                    near_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    far_pooled_video_features = torch.empty(0, valid_video_features.shape[-1], 
                                    dtype=valid_video_features.dtype,
                                    device=valid_video_features.device)
                    
                    b_near_pooled_video_features.append(near_pooled_video_features)
                    b_far_pooled_video_features.append(far_pooled_video_features)
                    
                    
            b_near_pooled_video_features = torch.cat(b_near_pooled_video_features,dim=0)
            b_far_pooled_video_features = torch.cat(b_far_pooled_video_features,dim=0)
            
            pooled_video_features = torch.cat(pooled_video_features,dim=0)
            batch_offset = torch.cat(batch_offset,dim=0)
        else:
            raise NotImplementedError
        
        return pooled_video_features, batch_offset,b_near_pooled_video_features,b_far_pooled_video_features
              
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
