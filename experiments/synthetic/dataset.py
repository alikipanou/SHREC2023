from changenetwork.datasets.synthetic.dataset import SyntheticDataset

from changenetwork.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode
)


def train_valid_data_loader(cfg):
  train_dataset = SyntheticDataset(subset='train')
                    
  neighbor_limits = calibrate_neighbors_stack_mode(train_dataset, registration_collate_fn_stack_mode, 
                                                   cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius)

  train_loader = build_dataloader_stack_mode(train_dataset,registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=True)
  
  valid_dataset = SyntheticDataset(subset='val')

  valid_loader = build_dataloader_stack_mode(valid_dataset, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=cfg.train.batch_size, num_workers=cfg.train.num_workers, shuffle=False)
  
  return train_loader, valid_loader, neighbor_limits, train_dataset.weights


def test_data_loader(cfg):
    train_dataset = SyntheticDataset(subset='train')
    
    neighbor_limits = calibrate_neighbors_stack_mode(train_dataset, registration_collate_fn_stack_mode, 
                                                   cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius)

    test_dataset = SyntheticDataset(subset='test')
    
    test_loader = build_dataloader_stack_mode(test_dataset, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=1, num_workers=1, shuffle=False)

    return test_loader, neighbor_limits
