from changenetwork.datasets.changedataset.dataset import ChangeDataset

from changenetwork.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode
)


def train_valid_data_loader(cfg):
  train_dataset = ChangeDataset(subset = 'train')
                    
  #neighbor_limits = calibrate_neighbors_stack_mode(train_dataset, registration_collate_fn_stack_mode, 
                                                   #cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius)
  
  #print(neighbor_limits)
  neighbor_limits = [ 7 ,13, 20, 23]
  train_loader = build_dataloader_stack_mode(train_dataset,registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=1, num_workers=1, shuffle=True)
  
  valid_dataset = ChangeDataset(subset = 'val')

  valid_loader = build_dataloader_stack_mode(valid_dataset, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=1, num_workers=1, shuffle=False)
  
  return train_loader, valid_loader, neighbor_limits, train_dataset.weights


def test_data_loader(cfg):
    train_dataset = ChangeDataset(subset='train')
    
    #neighbor_limits = calibrate_neighbors_stack_mode(train_dataset, registration_collate_fn_stack_mode, 
                                                   #cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius)
    neighbor_limits = [ 7 ,13, 20, 23]
    test_dataset = ChangeDataset(subset='test')
    
    test_loader = build_dataloader_stack_mode(test_dataset, registration_collate_fn_stack_mode, cfg.backbone.num_stages,
                                             cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits,
                                             batch_size=1, num_workers=1, shuffle=False)

    return test_loader, neighbor_limits



