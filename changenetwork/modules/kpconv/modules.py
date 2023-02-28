
import torch
from changenetwork.modules.kpconv.functional import nearest_upsample,maxpool
from changenetwork.modules.kpconv.kpconv import KPConv

class MaxPool(torch.nn.Module):
  @staticmethod
  def forward(x_feats,neighbor_indices):
    return maxpool(x_feats,neighbor_indices)


class NearestUpsample(torch.nn.Module):
  @staticmethod
  def forward(x_feats, upsample_indices):
    return nearest_upsample(x_feats, upsample_indices)


class GroupNorm(torch.nn.Module):
  def __init__(self,num_groups,num_channels):
    super(GroupNorm,self).__init__()
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.norm = torch.nn.GroupNorm(self.num_groups,self.num_channels)

  def forward(self,x):
    x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B, C, N)
    x = self.norm(x)
    x = x.squeeze(0).transpose(0, 1)  # (B, C, N) -> (N, C)
    return x.squeeze()


class UnaryBlock(torch.nn.Module):
  def __init__(self,in_channels,out_channels,num_groups, include_relu = True, include_bias = True ):
    super(UnaryBlock,self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.mlp = torch.nn.Linear(in_channels ,out_channels, bias = include_bias)
    self.norm = GroupNorm(num_groups,out_channels)

    if include_relu:
      self.leaky_relu = torch.nn.LeakyReLU(0.1)
    else:
      self.leaky_relu = None


  def forward(self,x):
    x = self.mlp(x)
    x = self.norm(x)

    if self.leaky_relu is not None:
      x  = self.leaky_relu(x)

    return x

class ConvBlock(torch.nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size, radius, sigma, num_groups, negative_slope = 0.1, bias = True):
    super(ConvBlock,self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kpconv = KPConv(in_channels,out_channels, kernel_size, radius, sigma, bias = bias)

    self.group_norm = GroupNorm(num_groups, out_channels)
    self.leaky_relu = torch.nn.LeakyReLU(negative_slope = negative_slope)

  def forward(self,s_feats, q_points, s_points, neighbor_indices):
    x = self.kpconv(s_feats, q_points, s_points, neighbor_indices)
    x = self.group_norm(x)
    x = self.leaky_relu(x)

    return x


class ResidualBlock(torch.nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size, radius, sigma, num_groups,strided = False, bias = True):
    r"""Initialize a ResNet bottleneck block.
        Args:
            in_channels: dimension input features
            out_channels: dimension input features
            kernel_size: number of kernel points
            radius: convolution radius
            sigma: influence radius of each kernel point
            group_norm: group number for GroupNorm
            strided: strided or not
            bias: If True, use bias in KPConv
            layer_norm: If True, use LayerNorm instead of GroupNorm
        """
    
    super(ResidualBlock, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.strided = strided

    if in_channels != out_channels // 4:
      self.unary1 = UnaryBlock(in_channels, out_channels // 4, num_groups, include_bias = bias)
    else:
      self.unary1 = torch.nn.Identity()
    
    self.conv = ConvBlock(out_channels // 4, out_channels // 4, kernel_size, radius, sigma, num_groups, bias = bias)

    self.unary2 = UnaryBlock(out_channels // 4, out_channels , num_groups, include_relu = False, include_bias = bias)

    if in_channels != out_channels:
      self.unary_shortcut = UnaryBlock(in_channels, out_channels , num_groups, include_relu = False, include_bias = bias)
    else:
      self.unary_shortcut = torch.nn.Identity()

    self.leaky_relu = torch.nn.LeakyReLU(0.1)
    

  def forward(self,s_feats, q_points, s_points, neighbor_indices):
    x = self.unary1(s_feats)

    x = self.conv(x, q_points, s_points, neighbor_indices)

    x = self.unary2(x)

    if self.strided:
      shortcut = maxpool(s_feats,neighbor_indices)
    else:
      shortcut = s_feats

    shortcut = self.unary_shortcut(shortcut)

    out = self.leaky_relu(x + shortcut)

    return out
