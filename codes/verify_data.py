import argparse
import matplotlib.pyplot as plt

from data.dstl_dataset.dataset import DstlDataset
import options.options as option


parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=True)
opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

train_set = DstlDataset(opt['datasets']['train'])

for i in range(100):
    data = train_set[i]
