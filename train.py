from torch.cuda.amp import autocast, GradScaler
import warnings
from torch.utils.data import DataLoader
from loader import *
from utils import *
from engine_org import *
from model import Model
import os
import sys
import torch
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(config):
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)
    
    logger = get_logger('train', log_dir)
    log_config_info(config, logger)

    set_seed(config.seed)
    gpu_ids = [0]
    torch.cuda.empty_cache()

    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=config.num_workers)
    
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)
    
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, drop_last=True)

    model_cfg = config.model_config
    model = Model()
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
    
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()
    min_loss = 999
    start_epoch = 1
    min_epoch = 1
    
    if os.path.exists(resume_model):
        checkpoint = torch.load(resume_model, map_location=torch.device('cuda'))
        model.module.load_state_dict(checkpoint, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        start_epoch = 1  

    save_interval = 5
    
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config, scaler=scaler)

        loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config)
        
        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        if epoch % save_interval == 0:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, f'{epoch}.pth'))
        
        torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        best_weight = torch.load(config.work_dir + '/checkpoints/best.pth', map_location=torch.device('cuda'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(test_loader, model, criterion, logger, config)

if __name__ == '__main__':
    config = setting_config
    main(config)
