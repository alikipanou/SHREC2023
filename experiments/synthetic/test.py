import argparse
import os.path as osp
import time

import numpy as np

from changenetwork.engine.single_tester import SingleTester
from changenetwork.utils.common import ensure_dir, get_log_string
from changenetwork.utils.torch import release_cuda

from config import make_cfg
from dataset import test_data_loader
from trainval import Evaluator
from model import create_model


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)
        message = f'Calibrate neighbors: {neighbor_limits}.'
        self.logger.info(message)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.evaluator = Evaluator().cuda()

        # preparation
        self.output_dir = osp.join(cfg.feature_dir)
        ensure_dir(self.output_dir)

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.evaluator(data_dict, output_dict)
        return result_dict

    def after_test_epoch(self, summary_board):
        dict_accuracies = summary_board.summary(['correct', 'correct_added', 'correct_removed', 'correct_nochange', 'correct_change', 'correct_color_change'])
        dict_gts = summary_board.summary_len(['correct_added', 'correct_removed', 'correct_nochange', 'correct_change', 'correct_color_change'])
    

        dict_tps = summary_board.summary_sum(['correct_added', 'correct_removed', 'correct_nochange', 'correct_change', 'correct_color_change'])
    
        dict_preds = {}
        labels_exist = []
        for label in ['added', 'removed', 'nochange', 'change', 'color_change']:
            if label not in summary_board.meter_names:
                dict_preds[label] = 0
            else:
                labels_exist.append(label)
        dict_preds2 = summary_board.summary_sum(labels_exist)
        dict_preds2.update(dict_preds)
    
        summary_dict = {}
        summary_dict['overall_accuracy'] = dict_accuracies['correct']
        summary_dict['added_accuracy'] = dict_accuracies['correct_added']
        summary_dict['removed_accuracy'] = dict_accuracies['correct_removed']
        summary_dict['nochange_accuracy'] = dict_accuracies['correct_nochange']
        summary_dict['change_accuracy'] = dict_accuracies['correct_change']
        summary_dict['color_change_accuracy'] = dict_accuracies['correct_color_change']
        
        summary_dict['added_iou'] = dict_tps['correct_added'] / (dict_gts['correct_added'] + dict_preds2['added'] - dict_tps['correct_added'])
        summary_dict['removed_iou'] = dict_tps['correct_removed'] / (dict_gts['correct_removed'] + dict_preds2['removed'] - dict_tps['correct_removed'])
        summary_dict['nochange_iou'] = dict_tps['correct_nochange'] / (dict_gts['correct_nochange'] + dict_preds2['nochange'] - dict_tps['correct_nochange'])
        summary_dict['change_iou'] = dict_tps['correct_change'] / (dict_gts['correct_change'] + dict_preds2['change'] - dict_tps['correct_change'])
        summary_dict['color_change_iou'] = dict_tps['correct_color_change'] / (dict_gts['correct_color_change'] + dict_preds2['color_change'] - dict_tps['correct_color_change'])
        summary_dict['mean_iou'] = (summary_dict['added_iou'] + summary_dict['removed_iou'] + summary_dict['nochange_iou'] + summary_dict['change_iou'] +  summary_dict['color_change_iou']) / 5

        return summary_dict


def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == '__main__':
    main()
