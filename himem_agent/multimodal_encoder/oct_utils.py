import torch
import numpy as np
import time
from typing import List, Dict, Tuple

def ravel_hash(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    arr: [N, 3]
    """
    assert len(arr.shape) == 2
    arr = arr - arr.min(0, keepdim=True)[0].to(torch.long)
    arr_max = arr.max(0, keepdim=True)[0].to(torch.long) + 1

    keys = torch.zeros(arr.shape[0], dtype=torch.long, device=arr.device)

    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[:, j + 1]
    keys += arr[:, -1]
    return keys

def build_octree_boxes_with_points(points_tensor, max_level=4, max_points_per_node=100, device='cpu'):
    """
    标准八叉树实现，同时标记叶子节点
    """
    min_bound = points_tensor.min(dim=0)[0]
    max_bound = points_tensor.max(dim=0)[0]
    
    root_center = (min_bound + max_bound) / 2
    root_size = (max_bound - min_bound).max().item()
    
    # 存储结果
    all_boxes = [[] for _ in range(max_level + 1)]
    all_points = [[] for _ in range(max_level + 1)]
    all_is_leaf = [[] for _ in range(max_level + 1)]  # 标记是否为叶子节点
    
    # 递归构建
    def build_node(center, size, point_indices, level):
        # 添加当前节点
        box = torch.cat([center, torch.tensor([size], device=device)])
        all_boxes[level].append(box.cpu())
        all_points[level].append(points_tensor[point_indices])
        
        # 终止条件：达到最大深度 或 点数少于阈值
        if level >= max_level or len(point_indices) <= max_points_per_node:
            all_is_leaf[level].append(True)  # 这是叶子节点
            return
        
        all_is_leaf[level].append(False)  # 这不是叶子节点
        
        # 八个子节点的偏移
        child_size = size / 2
        quarter_size = child_size / 2
        offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]
        ], dtype=torch.float32, device=device) * quarter_size
        
        # 为8个子节点分配点
        child_points = points_tensor[point_indices]
        relative_pos = child_points - center
        
        has_children = False
        for i, offset in enumerate(offsets):
            child_center = center + offset
            
            in_child = (
                (relative_pos[:, 0] * offset[0] >= 0) &
                (relative_pos[:, 1] * offset[1] >= 0) &
                (relative_pos[:, 2] * offset[2] >= 0)
            )
            
            child_indices = point_indices[in_child]
            
            if len(child_indices) > 0:
                has_children = True
                build_node(child_center, child_size, child_indices, level + 1)
        
        # 如果实际上没有子节点，更新为叶子节点
        if not has_children:
            all_is_leaf[level][-1] = True
    
    # 从根节点开始构建
    all_point_indices = torch.arange(len(points_tensor), device=device)
    build_node(root_center, root_size, all_point_indices, 0)
    
    # 转换为tensor并获取叶子节点
    leaf_boxes = []
    leaf_points = []
    
    for level in range(max_level + 1):
        if all_boxes[level]:
            all_boxes[level] = torch.stack(all_boxes[level])
            all_is_leaf[level] = torch.tensor(all_is_leaf[level])
            
            # 提取该层的叶子节点
            leaf_mask = all_is_leaf[level]
            if leaf_mask.any():
                level_leaf_boxes = all_boxes[level][leaf_mask]
                level_leaf_points = [all_points[level][i] for i in range(len(all_points[level])) if all_is_leaf[level][i]]
                
                leaf_boxes.append(level_leaf_boxes)
                leaf_points.extend(level_leaf_points)
            
        else:
            all_boxes[level] = torch.empty((0, 4), device='cpu')
            all_is_leaf[level] = torch.empty((0,), dtype=torch.bool)
    
    # 合并所有层的叶子节点
    if leaf_boxes:
        leaf_boxes = torch.cat(leaf_boxes, dim=0)
    else:
        leaf_boxes = torch.empty((0, 4), device='cpu')
    
    return all_boxes, all_points, all_is_leaf, leaf_boxes, leaf_points


def build_octree_boxes_with_points_v2(points_tensor, max_level=4, max_points_per_node=100, device='cpu'):
    """
    标准八叉树实现，同时标记叶子节点，存储点的索引
    """
    min_bound = points_tensor.min(dim=0)[0]
    max_bound = points_tensor.max(dim=0)[0]
    
    root_center = (min_bound + max_bound) / 2
    root_size = (max_bound - min_bound).max().item()
    
    # 存储结果
    all_boxes = [[] for _ in range(max_level + 1)]
    all_point_indices = [[] for _ in range(max_level + 1)]  # 改为存储索引
    all_is_leaf = [[] for _ in range(max_level + 1)]
    
    # 递归构建
    def build_node(center, size, point_indices, level):
        """递归构建八叉树节点"""
        # 添加当前节点
        box = torch.cat([center, torch.tensor([size], device=device)])
        all_boxes[level].append(box.cpu())
        all_point_indices[level].append(point_indices.cpu())  # 存储索引而非点
        
        # 终止条件：达到最大深度 或 点数少于阈值
        if level >= max_level or len(point_indices) <= max_points_per_node:
            all_is_leaf[level].append(True)
            return
        
        all_is_leaf[level].append(False)
        
        # 八个子节点的偏移
        child_size = size / 2
        quarter_size = child_size / 2
        offsets = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]
        ], dtype=torch.float32, device=device) * quarter_size
        
        # 为8个子节点分配点
        child_points = points_tensor[point_indices]
        relative_pos = child_points - center
        
        has_children = False
        for i, offset in enumerate(offsets):
            child_center = center + offset
            
            in_child = (
                (relative_pos[:, 0] * offset[0] >= 0) &
                (relative_pos[:, 1] * offset[1] >= 0) &
                (relative_pos[:, 2] * offset[2] >= 0)
            )
            
            child_indices = point_indices[in_child]
            
            if len(child_indices) > 0:
                has_children = True
                build_node(child_center, child_size, child_indices, level + 1)
        
        if not has_children:
            all_is_leaf[level][-1] = True
    
    # 从根节点开始构建
    all_point_indices_root = torch.arange(len(points_tensor), device=device)
    build_node(root_center, root_size, all_point_indices_root, 0)
    
    # 转换为tensor并获取叶子节点
    leaf_boxes = []
    leaf_point_indices = []  # 存储索引
    
    for level in range(max_level + 1):
        if all_boxes[level]:
            all_boxes[level] = torch.stack(all_boxes[level])
            all_is_leaf[level] = torch.tensor(all_is_leaf[level])
            
            # 提取该层的叶子节点
            leaf_mask = all_is_leaf[level]
            if leaf_mask.any():
                level_leaf_boxes = all_boxes[level][leaf_mask]
                level_leaf_indices = [all_point_indices[level][i] for i in range(len(all_point_indices[level])) if all_is_leaf[level][i]]
                
                leaf_boxes.append(level_leaf_boxes)
                leaf_point_indices.extend(level_leaf_indices)
            
        else:
            all_boxes[level] = torch.empty((0, 4), device='cpu')
            all_is_leaf[level] = torch.empty((0,), dtype=torch.bool)
    
    # 合并所有层的叶子节点
    if leaf_boxes:
        leaf_boxes = torch.cat(leaf_boxes, dim=0)
    else:
        leaf_boxes = torch.empty((0, 4), device='cpu')
    
    
    return all_boxes, all_point_indices, all_is_leaf, leaf_boxes, leaf_point_indices

def filter_boxes_by_distance(boxes, pos, dis):
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=boxes.dtype, device=boxes.device)
    else:
        pos = pos.to(boxes.device)
    # 提取box中心点 [N, 3]
    box_centers = boxes[:, :3]
    # 计算中心点到pos的欧氏距离 [N,]
    distances = torch.norm(box_centers - pos, dim=1)
    # 创建mask
    mask = distances <= dis
    
    return mask

def filter_boxes_by_distance_v2(boxes_points, pos, dis):
    """使用box中最近点的距离来判断"""
    if len(boxes_points) == 0:
        return torch.tensor([], dtype=torch.bool)
    
    device = boxes_points[0].device
    dtype = boxes_points[0].dtype
    
    if not isinstance(pos, torch.Tensor):
        pos = torch.tensor(pos, dtype=dtype, device=device)
    else:
        pos = pos.to(device=device, dtype=dtype)
    
    # 计算每个box中所有点到pos的最小距离
    min_distances = []
    for points in boxes_points:
        if len(points.shape) == 1: # shape may be (3,)
            points = points.unsqueeze(0)

        point_distances = torch.norm(points - pos, dim=1)  # [K,]
        min_dist = point_distances.min()
        min_distances.append(min_dist)
    
    distances = torch.stack(min_distances)  # [N,]
    
    mask = distances <= dis
    return mask




# 将核心计算逻辑提取为可编译的函数
@torch.compile(mode='max-autotune')
def compute_child_indices(batch_points, batch_parent_centers):
    """编译优化：计算子节点索引"""
    rel = batch_points - batch_parent_centers
    local_child_idx = (rel[:, 0] >= 0).long()
    local_child_idx += (rel[:, 1] >= 0).long() * 2
    local_child_idx += (rel[:, 2] >= 0).long() * 4
    return local_child_idx

@torch.compile(mode='max-autotune')
def compute_offsets(offsets_template, local_idx, quarter_sizes_batch):
    """编译优化：计算中心偏移"""
    return offsets_template[local_idx] * quarter_sizes_batch.unsqueeze(1)

def build_octree_v36_compiled(points_tensor, max_level=4, min_voxel_size=5, max_points_per_node=4):
    """
    修复版本：在识别叶子节点时立即保存点索引
    """
    device = points_tensor.device
    N = points_tensor.shape[0]
    
    # 初始化
    if hasattr(torch, 'aminmax'):
        min_bound, max_bound = torch.aminmax(points_tensor, dim=0)
    else:
        min_bound = points_tensor.min(dim=0)[0]
        max_bound = points_tensor.max(dim=0)[0]
    
    root_center = (min_bound + max_bound) * 0.5
    root_size = (max_bound - min_bound).max()
    
    offsets_template = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1]
    ], dtype=torch.float32, device=device)
    
    all_centers = [root_center.unsqueeze(0)]
    all_sizes = [root_size.unsqueeze(0)]
    all_node_ranges = [torch.tensor([[0, N]], dtype=torch.long, device=device)]
    point_indices = torch.arange(N, device=device)
    
    all_leaf_boxes = []
    all_leaf_point_indices = []  # ← 改为直接保存点索引列表
    
    for level in range(max_level + 1):
        level_centers = all_centers[level]
        level_sizes = all_sizes[level]
        num_nodes = level_centers.shape[0]
        
        if num_nodes == 0:
            break
        
        ranges = all_node_ranges[level]
        node_point_counts = ranges[:, 1] - ranges[:, 0]
        
        is_leaf = (level >= max_level) | (level_sizes <= min_voxel_size) | (node_point_counts <= max_points_per_node)
        
        # ← 关键修复：立即提取叶子节点的实际点索引
        if is_leaf.any():
            leaf_centers = level_centers[is_leaf]
            leaf_sizes = level_sizes[is_leaf]
            all_leaf_boxes.append(torch.cat([leaf_centers, leaf_sizes.unsqueeze(1)], dim=1))
            
            # 立即保存当前叶子节点的点索引（克隆以避免后续修改影响）
            leaf_ranges = ranges[is_leaf]
            for start, end in leaf_ranges:
                leaf_pts = point_indices[start:end].clone()
                all_leaf_point_indices.append(leaf_pts)
        
        if is_leaf.all():
            all_centers.append(torch.empty((0, 3), device=device))
            all_sizes.append(torch.empty(0, device=device))
            all_node_ranges.append(torch.empty((0, 2), dtype=torch.long, device=device))
            break
        
        # 处理非叶子节点
        non_leaf_mask = ~is_leaf
        non_leaf_ranges = ranges[non_leaf_mask]
        non_leaf_counts = node_point_counts[non_leaf_mask]
        non_leaf_centers = level_centers[non_leaf_mask]
        non_leaf_sizes = level_sizes[non_leaf_mask]
        total_points = non_leaf_counts.sum().item()
        
        counts_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            non_leaf_counts.cumsum(0)
        ])
        
        batch_parent_ids = torch.repeat_interleave(
            torch.arange(len(non_leaf_ranges), device=device),
            non_leaf_counts
        )
        
        within_offsets = torch.arange(total_points, device=device) - counts_cumsum[batch_parent_ids]
        global_indices = non_leaf_ranges[batch_parent_ids, 0] + within_offsets
        batch_point_indices = point_indices[global_indices]
        
        batch_points = points_tensor[batch_point_indices]
        batch_parent_centers = non_leaf_centers[batch_parent_ids]
        
        local_child_idx = compute_child_indices(batch_points, batch_parent_centers)
        global_child_idx = batch_parent_ids * 8 + local_child_idx
        
        unique_children, inverse_indices = torch.unique(global_child_idx, return_inverse=True)
        num_children = unique_children.size(0)
        
        sorted_inverse, sort_idx = torch.sort(inverse_indices)
        sorted_points = batch_point_indices[sort_idx]
        
        child_counts = torch.bincount(inverse_indices, minlength=num_children)
        child_cumsum = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            child_counts.cumsum(0)
        ])
        
        next_ranges = torch.stack([child_cumsum[:-1], child_cumsum[1:]], dim=1)
        point_indices[:total_points] = sorted_points
        
        parent_idx = unique_children // 8
        local_idx = unique_children % 8
        
        half_sizes = non_leaf_sizes * 0.5
        quarter_sizes = half_sizes * 0.5
        
        parent_centers_batch = non_leaf_centers[parent_idx]
        quarter_sizes_batch = quarter_sizes[parent_idx]
        
        offsets = compute_offsets(offsets_template, local_idx, quarter_sizes_batch)
        next_centers = parent_centers_batch + offsets
        next_sizes = half_sizes[parent_idx]
        
        all_centers.append(next_centers)
        all_sizes.append(next_sizes)
        all_node_ranges.append(next_ranges)
    
    # 整合叶子节点信息
    if all_leaf_boxes:
        leaf_boxes = torch.cat(all_leaf_boxes, dim=0)
        
        # ← 从保存的点索引列表构建输出
        leaf_point_indices_flat = torch.cat(all_leaf_point_indices, dim=0)
        leaf_point_counts = torch.tensor([len(pts) for pts in all_leaf_point_indices], 
                                         dtype=torch.long, device=device)
        
        leaf_info = (leaf_point_indices_flat, leaf_point_counts)
    else:
        leaf_boxes = torch.empty((0, 4), device=device)
        leaf_info = (torch.empty(0, dtype=torch.long, device=device), 
                     torch.empty(0, dtype=torch.long, device=device))
    
    return leaf_boxes, leaf_info


def build_point_to_leaf_mapping(leaf_info, num_points):
    leaf_point_indices_flat, leaf_point_counts = leaf_info
    device = leaf_point_indices_flat.device
    
    # 初始化为-1（表示不在任何叶子节点中）
    point_to_leaf = torch.full((num_points,), -1, dtype=torch.long, device=device)
    num_leaves = leaf_point_counts.shape[0]
    leaf_ids = torch.repeat_interleave(
        torch.arange(num_leaves, device=device),
        leaf_point_counts
    )
    # 建立映射
    point_to_leaf[leaf_point_indices_flat] = leaf_ids
    return point_to_leaf
