from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from loader import *
from engine_org import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings

warnings.filterwarnings("ignore")

def main(config):
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'best.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    logger = get_logger('test', log_dir)
    log_config_info(config, logger)

    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    model = Model()
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    best_weight = torch.load(resume_model, map_location=torch.device('cuda'))
    model.module.load_state_dict(best_weight, strict=False)
    loss = test_one_epoch(test_loader, model, criterion, logger, config)

if __name__ == '__main__':
    config = setting_config
    main(config)