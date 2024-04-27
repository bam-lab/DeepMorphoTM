import torch
import numpy as np
import concurrent.futures
import os


def read_dataset_info(info):
    sdf_frames = []
    displacement_frames = []
    moduli = []
    num_frames = []
    seq_counter = 0
    for modulus in info:
        for seq in info[modulus]:
            num_frames.append(len(info[modulus][seq]))
            moduli.append(int(modulus))
            sdf_frames.append([])
            displacement_frames.append([])
            for instance_name in info[modulus][seq]:
                sdf_instance_filename= os.path.join(
                    modulus, seq, 'SDF', instance_name + ".npy"
                )
                displacement_instance_filename= os.path.join(
                    modulus, seq, 'Displacement', instance_name + ".npy"
                )
                sdf_frames[seq_counter] += [sdf_instance_filename]
                displacement_frames[seq_counter] += [displacement_instance_filename]
            seq_counter += 1
    return sdf_frames, displacement_frames, moduli, num_frames

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_source, data_info, num_frames_per_seg, num_sampled_frames_per_seg, augmentation=True):
        
        sdf_frame_names, displacement_frame_names, moduli, num_frames = read_dataset_info(data_info)
        self.num_sampled_frames_per_seg = num_sampled_frames_per_seg
        self.augmentation = augmentation
        self.segs = []
        self.segs_seq = []
        self.segs_modulus = []        
        num_following_frames_per_seg = num_frames_per_seg - 1

        for seq in range(len(sdf_frame_names)):
            init_seg_frames = list(range(0, num_frames[seq] - num_following_frames_per_seg))
            for init_seg_frame in init_seg_frames:
                self.segs.append(np.linspace(init_seg_frame, init_seg_frame + num_following_frames_per_seg, self.num_sampled_frames_per_seg).astype(int))
                self.segs_seq.append(seq)
                self.segs_modulus.append(moduli[seq])
        
        def np_load(dir):
            return np.load(os.path.join(data_source, dir))
        def np_load_displacement(dir):
            try:
                return np.load(os.path.join(data_source, dir))
            except:
                return np.zeros((2, 256, 256))
        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            self.sdf_ram = [list(executor.map(np_load, seq)) for seq in sdf_frame_names]
            self.displacement_ram = [list(executor.map(np_load_displacement, seq)) for seq in displacement_frame_names]
            
    def __len__(self):
            return len(self.segs)

    def __getitem__(self, idx):

        sampled_seg_seq = self.segs_seq[idx]
        sampled_modulus = self.segs_modulus[idx]
        sampled_sdf = np.array([self.sdf_ram[sampled_seg_seq][fr] for fr in self.segs[idx]])
        # sampled_displacement = np.array([self.displacement_ram[sampled_seg_seq][fr] for fr in self.segs[idx]])
        sampled_displacement = self.displacement_ram[sampled_seg_seq][self.segs[idx][3]]

        if self.augmentation:
            self.augmentation = np.random.randint(6)
        else:
            self.augmentation = 0

        if self.augmentation == 1:
            sampled_sdf = np.rot90(sampled_sdf, k=1, axes=(1, 2))
            sampled_displacement = np.rot90(sampled_displacement, k=1, axes=(1, 2))[::-1]
            sampled_displacement[1] *= -1
        elif self.augmentation == 2:
            sampled_sdf = np.rot90(sampled_sdf, k=2, axes=(1, 2))
            sampled_displacement = - np.rot90(sampled_displacement, k=2, axes=(1, 2))
        elif self.augmentation == 3:
            sampled_sdf = np.rot90(sampled_sdf, k=3, axes=(1, 2))
            sampled_displacement = np.rot90(sampled_displacement, k=3, axes=(1, 2))[::-1]
            sampled_displacement[0] *= -1
        elif self.augmentation == 4:
            sampled_sdf = np.flip(sampled_sdf, axis=2)
            sampled_displacement = np.flip(sampled_displacement, axis=2)
            sampled_displacement[0] *= -1
        elif self.augmentation == 5:
            sampled_sdf = np.flip(sampled_sdf, axis=1)
            sampled_displacement = np.flip(sampled_displacement, axis=1)
            sampled_displacement[1] *= -1

        return sampled_sdf.astype(np.float32), sampled_displacement.astype(np.float32), sampled_modulus
