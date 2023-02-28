import torch

from changenetwork.modules.ops.index_select import index_select

def nearest_upsample(x, upsample_indices):

  """Pools features from the closest neighbors.
  WARNING: this function assumes the neighbors are ordered.
  Args:
      x: [n1, d] features matrix
      upsample_indices: [n2, max_num] Only the first column is used for pooling
  Returns:
      x: [n2, d] pooled features matrix
  """
  # Add a last row with minimum features for shadow pools
  x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
  # Get features for each pooling location [n2, d]
  x = index_select(x, upsample_indices[:, 0], dim=0)
  return x


def maxpool(x, neighbor_indices):

  """Max pooling from neighbors.
  Args:
      x: [n1, d] features matrix
      neighbor_indices: [n2, max_num] pooling indices
  Returns:
      pooled_feats: [n2, d] pooled features matrix
  """
  x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
  neighbor_feats = index_select(x, neighbor_indices, dim=0)

  pooled_feats = neighbor_feats.max(1)[0]

  return pooled_feats

