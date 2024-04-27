import os
import signal
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from data import Dataset
from network import DeepMorphoTM
import workspace
import utils


def infer(experiment_directory, exp_No):

    signal.signal(signal.SIGINT, utils.signal_handler)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # torch.backends.cudnn.benchmark = True
    # torch.autograd.set_detect_anomaly(False)
    # torch.autograd.profiler.profile(False)
    # torch.autograd.profiler.emit_nvtx(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = workspace.load_experiment_specifications(experiment_directory, exp_No)

    cnn_init_dim = specs['CNNInitDim']
    num_workers = specs['NumWorkers']
    data_source = specs['DataSource']
    with open(specs['TestDataInfo'], "r") as f:
        test_data_info = json.load(f) 
    num_frames_per_seg = specs['NumFramesPerSeg']
    num_sampled_frames_per_seg = specs['NumSampledFramesPerSeg']

    cnn = DeepMorphoTM(num_sampled_frames_per_seg, cnn_init_dim).to(device)
    cnn = torch.compile(cnn)
    _ = workspace.load_model(experiment_directory, exp_No, 'latest', cnn)

    test_set = Dataset(data_source, test_data_info, num_frames_per_seg, num_sampled_frames_per_seg, augmentation=False)
    test_dataloader = DataLoader(test_set,
                                    batch_size=specs['TestBatchSize'],
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=False,
                                    )
    inference = []
    with torch.no_grad():
        for sdf, displacement, modulus in test_dataloader:
            sdf, modulus = sdf.to(device, non_blocking=True), modulus.to(device, non_blocking=True)[..., None, None, None]
            out = cnn(sdf, modulus)
            inference.append(out.detach().cpu().numpy())
    
    inference = np.concatenate(inference, axis=0)
    np.save(os.path.join(experiment_directory, 'Inference_' + str(exp_No), 'displacement.npy'), inference)




if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Inference.")
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
        os.makedirs(os.path.join(args.experiment_directory, 'Inference_' + str(exp_No)), exist_ok=True)
        infer(args.experiment_directory, exp_No)