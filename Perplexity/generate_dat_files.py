'''
Creates two folders:

1) For original sentences
2) For attacked sentences

In each folder, there will be created a .dat file per data point.
There will also be created a flist file, storing a list of all the .dat files in each folder

The purpose of creating this is to be used by a run_gdlm.sh script to find the perplexity
per data point
'''

import sys
import os
import argparse
import json

def load_data(base_dir):
    '''
    Returns data as list from attacked data directory
    '''

    data_list = []
    for i in range(2000):
        fname = base_dir + '/'+str(i)+'.txt'
        try:
            with open(fname, 'r') as f:
                item = json.load(f)
            data_list.append(item)
        except:
            print("Failed to load", i)
    return data_list

def keep_only_success(data_list):
    '''
    data_list: list
        [item1, item2, ...]
        where,
            item: dict
                with keys:
                    sentence
                    updated sentence
                    original prob
                    updated prob
                    true label
    Filter list to keep only items where adv attack worked
    '''
    new_data_list = []
    for item in data_list:
        original_prob = item['original prob']
        updated_prob = item['updated prob']
        label = item['true label']

        original_pred = original_prob.index(max(original_prob))
        updated_pred = updated_prob.index(max(updated_prob))

        if original_pred != label:
            continue

        if updated_pred != original_pred:
            new_data_list.append(item)

    return new_data_list

if __name__ == '__main__':
	# Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('DATA_DIR', type=str, help='Attacked Data Directory')
    commandLineParser.add_argument('OUT_ORIG_DIR', type=str, help='Dir to hold original .dat files')
    commandLineParser.add_argument('OUT_ADV_DIR', type=str, help='Dir to hold adversarial .dat files')

    args = commandLineParser.parse_args()
    data_dir = args.DATA_DIR
    out_orig_dir = args.OUT_ORIG_DIR
    out_adv_dir = args.OUT_ADV_DIR

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/generate_dat_files.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Create the .dat out directories
    if not os.path.isdir(out_orig_dir):
        os.mkdir(out_orig_dir)
    if not os.path.isdir(out_adv_dir):
        os.mkdir(out_adv_dir)

    # Load all the data from attacked data directory
    data_list = load_data(data_dir)

    # Keep only successful adversarial attack data points
    data_list = keep_only_success(data_list)

    # Create the tflist.txt files
    with open(f'{out_orig_dir}/tflist.txt', 'w') as f:
        f.truncate(0)
    with open(f'{out_adv_dir}/tflist.txt', 'w') as f:
        f.truncate(0)

    # Generate the .dat files
    for num, item in enumerate(data_list):
        orig = item['sentence']
        adv = item['updated sentence']

        # Original
        filename = f'{out_orig_dir}/{num}.dat'
        with open(filename, 'w') as f:
            f.write(f'4 <s> {orig} </s>')
        with open(f'{out_orig_dir}/tflist.txt', 'a+') as f:
            f.write(filename+'\n')

        # Adversarial
        filename = f'{out_adv_dir}/{num}.dat'
        with open(filename, 'w') as f:
            f.write(f'4 <s> {adv} </s>')
        with open(f'{out_adv_dir}/tflist.txt', 'a+') as f:
            f.write(filename+'\n')
