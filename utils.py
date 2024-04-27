import sys
import random
import torch
import numpy


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def signal_handler(sig, frame):
    sys.exit(0)


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class ConstantLearningRateSchedule(LearningRateSchedule):
    def __init__(self, value):
        self.value = value

    def get_learning_rate(self, epoch):
        return self.value

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor, max_steps):
        self.initial = initial
        self.interval = interval
        self.factor = factor
        self.max_steps = max_steps

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** min((epoch // self.interval), self.max_steps))

class WarmupLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, warmed_up, length):
        self.initial = initial
        self.warmed_up = warmed_up
        self.length = length

    def get_learning_rate(self, epoch):
        if epoch > self.length:
            return self.warmed_up
        return self.initial + (self.warmed_up - self.initial) * epoch / self.length

def get_learning_rate_schedules(specs):

    schedule_specs = specs["LearningRateSchedule"]

    for schedule_specs in schedule_specs:

        if schedule_specs["Type"] == "Step":
            schedule = StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                        schedule_specs["MaxSteps"],
                    )
            
        elif schedule_specs["Type"] == "Warmup":
            schedule = WarmupLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Final"],
                        schedule_specs["Length"],
                    )
            
        elif schedule_specs["Type"] == "Constant":
            schedule = ConstantLearningRateSchedule(schedule_specs["Value"])

        else:
            raise Exception(
                'no known learning rate schedule of type "{}"'.format(
                    schedule_specs["Type"]
                )
            )

    return schedule


def adjust_learning_rate(lr_schedule, optimizer, epoch):
    lr = lr_schedule.get_learning_rate(epoch)
    optimizer.param_groups[0]["lr"] = lr
    return lr