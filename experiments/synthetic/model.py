import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from einops.layers import torch as ELT



from backbone import KPConvFPN

class ChangeNetwork(nn.Module):
  def __init__(self, cfg):
    super(ChangeNetwork, self).__init__()

    self.backbone =  KPConvFPN(cfg.backbone.input_dim,
                               cfg.backbone.output_dim,
                               cfg.backbone.init_dim,
                               cfg.backbone.kernel_size,
                               cfg.backbone.init_radius,
                               cfg.backbone.init_sigma,
                               cfg.backbone.group_norm)
    
    self.global_max_pooling = ELT.Reduce("N d -> d", "max")
    self.sequential = nn.Sequential(nn.Linear(cfg.backbone.output_dim, 512),nn.Dropout(), nn.ReLU(),
                                    nn.Linear(512, 256), nn.Dropout(),nn.ReLU(),
                                    nn.Linear(256, 5))
  
    
  def forward(self, data_dict):
    output_dict = {}

    feats = data_dict['features'].detach()

    ref_length = data_dict['lengths'][-1][0].item()

    feats = self.backbone(feats, data_dict)

    ref_feats = feats[:ref_length]
    src_feats = feats[ref_length:]

    ref_feats = self.global_max_pooling(ref_feats)
    src_feats = self.global_max_pooling(src_feats)

    diff = (ref_feats - src_feats)
    
    out = self.sequential(diff)

    out = torch.flatten(out)

    output_dict['output'] = out

    return output_dict


def create_model(cfg):
    model = ChangeNetwork(cfg)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)

if __name__ == '__main__':
    main()
