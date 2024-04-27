import os
import signal
import random
import json
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from data import Dataset
from network import DeepMorphoTM
import workspace
import utils


def train(experiment_directory, exp_No):

    signal.signal(signal.SIGINT, utils.signal_handler)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(False)
    # torch.autograd.profiler.profile(False)
    # torch.autograd.profiler.emit_nvtx(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(os.path.join(experiment_directory, str(exp_No)))
    specs = workspace.load_experiment_specifications(experiment_directory, exp_No)
    if specs["Reproducibility"]:
        seed = specs["RandomSeed"]
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        random.seed(seed)
        numpy.random.seed(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        g = torch.Generator()
        g.manual_seed(seed)
        worker_init_fn = utils.seed_worker
    else:
        g = None
        worker_init_fn = None

    cnn_init_dim = specs['CNNInitDim']
    train_batch_size = specs['TrainBatchSize']
    lr_schedule = utils.get_learning_rate_schedules(specs)
    num_workers = specs['NumWorkers']
    num_epochs = specs['NumEpochs']
    data_source = specs['DataSource']
    with open(specs['TrainDataInfo'], "r") as f:
        train_data_info = json.load(f)
    with open(specs['ValidDataInfo'], "r") as f:
        valid_data_info = json.load(f) 
    num_frames_per_seg = specs['NumFramesPerSeg']
    num_sampled_frames_per_seg = specs['NumSampledFramesPerSeg']
    log_frequency = utils.get_spec_with_default(specs, "LogFrequency", 10)
    checkpoints = list(
            range(
                specs["SnapshotFrequency"] - 1,
                specs["NumEpochs"] + 1,
                specs["SnapshotFrequency"],
            )
        )

    cnn = DeepMorphoTM(num_sampled_frames_per_seg, cnn_init_dim).to(device)
    cnn = torch.compile(cnn)
    if specs['Optimizer'] == 'Adam':
        optimizer = optim.Adam(cnn.parameters(), lr=lr_schedule.get_learning_rate(0))
    elif specs['Optimizer'] == 'AdamW':
        optimizer = optim.AdamW(cnn.parameters(), lr=lr_schedule.get_learning_rate(0), weight_decay=specs['WeightDecay'])
    criterion = nn.L1Loss()

    training_set = Dataset(data_source, train_data_info, num_frames_per_seg, num_sampled_frames_per_seg)
    training_dataloader = DataLoader(training_set, 
                                    batch_size=train_batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True,
                                    worker_init_fn=worker_init_fn,
                                    generator=g
                                    )
    num_batches_per_training_epoch = len(training_dataloader)
    validation_set = Dataset(data_source, valid_data_info, num_frames_per_seg, num_sampled_frames_per_seg, augmentation=False)
    valid_batch_size = utils.get_spec_with_default(specs, 'ValidBatchSize', len(validation_set) // 5)
    validation_dataloader = DataLoader(validation_set,
                                    batch_size=valid_batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=True,
                                    worker_init_fn=worker_init_fn,
                                    generator=g)
    num_batches_per_validation_epoch = len(validation_dataloader)

    for epoch in range(num_epochs):

        lr = utils.adjust_learning_rate(lr_schedule, optimizer, epoch)
        writer.add_scalar("lr", lr, epoch)

        loss_train = 0
        for sdf, displacement, modulus in training_dataloader:
            sdf, displacement, modulus = sdf.to(device, non_blocking=True), displacement.to(device, non_blocking=True), modulus.to(device, non_blocking=True)[..., None, None, None]
            out = cnn(sdf, modulus)
            batch_loss_train = criterion(out, displacement)
            batch_loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_train += batch_loss_train.detach()
        loss_train /= num_batches_per_training_epoch
        writer.add_scalar("loss/train", loss_train, epoch)
        
        with torch.no_grad():
            loss_valid = 0
            for sdf, displacement, modulus in validation_dataloader:
                sdf, displacement, modulus = sdf.to(device, non_blocking=True), displacement.to(device, non_blocking=True), modulus.to(device, non_blocking=True)[..., None, None, None]
                out = cnn(sdf, modulus)
                loss_valid += criterion(out, displacement)
            loss_valid /= num_batches_per_validation_epoch
            writer.add_scalar("loss/valid", loss_valid, epoch)

        if epoch in checkpoints:
            workspace.save_checkpoints(experiment_directory, exp_No, epoch, cnn, optimizer)
        if epoch % log_frequency == 0:
            workspace.save_latest(experiment_directory, exp_No, epoch, cnn, optimizer)

    writer.flush()
    writer.close()



if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--exp_No",
        "-n",
        nargs="*",
        type=int,
        dest="exp_No",
        required=True,
    )

    args = arg_parser.parse_args()

    for exp_No in args.exp_No:
        train(args.experiment_directory, exp_No)