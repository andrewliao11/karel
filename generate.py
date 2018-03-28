#!/usr/bin/env python
import os
import argparse
import numpy as np
import h5py

from karel import KarelWithCurlyParser, KarelForSynthesisParser
from karel import str2bool, makedirs, pprint, beautify, TimeoutError

try:
    from tqdm import trange
except:
    trange = range


if __name__ == '__main__':
    data_arg = argparse.ArgumentParser()
    data_arg.add_argument('--num_code_train', type=int, default=50000)
    data_arg.add_argument('--num_code_test', type=int, default=5000)
    data_arg.add_argument('--num_code_val', type=int, default=5000)
    data_arg.add_argument('--num_trace_train', type=int, default=10)
    data_arg.add_argument('--num_trace_test', type=int, default=10)
    data_arg.add_argument('--num_trace_val', type=int, default=10)
    data_arg.add_argument('--num_examples', type=int, default=2)
    data_arg.add_argument('--parser_type', type=str, default='curly', choices=['curly', 'synthesis'])
    data_arg.add_argument('--data_dir', type=str, default='data')
    data_arg.add_argument('--max_depth', type=int, default=5)
    data_arg.add_argument('--mode', type=str, default='token', choices=['text', 'token'])
    data_arg.add_argument('--beautify', type=str2bool, default=False)
    data_arg.add_argument('--world_height', type=int, default=10, help='Height of square grid world')
    data_arg.add_argument('--world_width', type=int, default=10, help='Width of square grid world')
    config = data_arg.parse_args()

    # Make directories
    makedirs(config.data_dir)
    datasets = ['train', 'test', 'val']

    # Generate datasets
    if config.parser_type == "curly":
        parser = KarelWithCurlyParser()
    elif config.parser_type == "synthesis":
        parser = KarelForSynthesisParser()

    hf_path = os.path.join(config.data_dir, 'karel_{}_{}_{}_world_{}x{}.h5'.format(config.max_depth, 
                                                                                   config.num_code_train, 
                                                                                   config.num_code_test, 
                                                                                   config.world_width, 
                                                                                   config.world_height))
    hf = h5py.File(hf_path, 'w')
    word2idx = hf.create_group('word2idx')
    idx2word = hf.create_group('idx2word')
    action2idx = hf.create_group('action2idx')
    idx2action = hf.create_group('idx2action')
    for k, v in parser.token_to_idx_details.items():
        word2idx[k] = str(v)
        idx2word[str(v)] = k
    for k, v in parser.get_action_dict().items():
        action2idx[k] = str(v)
        idx2action[str(v)] = k

    if config.mode == 'text':
        for name in datasets:
            data_num = getattr(config, "num_{}".format(name))

            text = ""
            text_path = os.path.join(config.data_dir, "{}.txt".format(name))

            for _ in trange(data_num):
                code = parser.random_code(stmt_max_depth=config.max_depth)
                if config.beautify:
                    code = beautify(code)
                text += code  + "\n"

            with open(text_path, 'w') as f:
                f.write(text)
    else:
        for name in datasets:
            code_num = getattr(config, "num_code_{}".format(name))
            trace_num = getattr(config, "num_trace_{}".format(name))
            grp = hf.create_group(name)

            data = []
            i = 0
            while True:
                subgrp = grp.create_group('{}'.format(i))
                while True:
                    code = parser.random_code(stmt_max_depth=config.max_depth)
                    inputs, outputs, state_sequences, state_actions = [], [], [], []
                    try:
                        for _ in range(trace_num):
                            parser.new_game(world_size=(config.world_width, config.world_height))
                            input = parser.get_state()
                            parser.run(code)
                            output = parser.get_state()
                            # get state sequence
                            state_sequence = parser.get_state_sequence()
                            state_action = parser.get_state_action()
                            inputs.append(input)
                            state_sequences.append(np.array(state_sequence))
                            state_actions.append(np.array(state_action, dtype=np.uint8))
                            outputs.append(output)
                    except TimeoutError:
                        continue
                    except IndexError:
                        continue

                    if sum([(len(state_action) < 1)+0 for state_action in state_actions]) > 0:
                        continue

                    token_idxes = parser.lex_to_idx(code, details=True)
                    inputs = np.array(inputs)
                    outputs = np.array(outputs)

                    subgrp['input'] = inputs
                    subgrp['output'] = outputs
                    for j, (arr1, arr2) in enumerate(zip(state_sequences, state_actions)):
                        subgrp['state_sequence_{}'.format(j)] = arr1
                        subgrp['state_action_{}'.format(j)] = arr2
                    subgrp['code'] = token_idxes
                    subgrp['code_length'] = len(token_idxes)
                    i += 1
                    break
                if i % 20 == 0:
                    print('{}/{} [{:.3f}%]'.format(i, code_num, 100.*i/code_num))
                if i > code_num:
                    break
