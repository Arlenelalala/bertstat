# coding=utf-8
# Copyright 2018 The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BERT checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import torch
import numpy as np

from bert.modeling import BertConfig, BertModel

parser = argparse.ArgumentParser()

# Required parameters
b_path = '/Users/liangsong/Desktop/refcode/pretrained_LM/uncased_L-24_H-1024_A-16'
parser.add_argument("--tf_checkpoint_path",
                    default=b_path+'/bert_model.ckpt',
                    type=str,
                    help="Path the TensorFlow checkpoint path.")
parser.add_argument("--bert_config_file",
                    default=b_path+'/bert_config.json',
                    type=str,
                    help="The config json file corresponding to the pre-trained BERT model. \n"
                         "This specifies the model architecture.")
parser.add_argument("--pytorch_dump_path",
                    default=b_path+"/pytorch_model.bin",
                    type=str,
                    help="Path to the output PyTorch model.")

args = parser.parse_args()


def convert(args):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(args.bert_config_file)
    model = BertModel(config)

    # Load weights from TF model
    path = args.tf_checkpoint_path
    print("Converting TensorFlow checkpoint from {}".format(path))

    init_vars = tf.train.list_variables(path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading {} with shape {}".format(name, shape))
        array = tf.train.load_variable(path, name)
        print("Numpy array shape {}".format(array.shape))
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name[5:]  # skip "bert/"
        print("Loading {}".format(name))
        name = name.split('/')
        if name[0] in ['redictions', 'eq_relationship']:
            print("Skipping")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel':
                pointer = getattr(pointer, 'weight')
            else:
                if l[0] != 'l_step':
                    pointer = getattr(pointer, l[0], name)
                else:
                    print(l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        except AttributeError:
            continue
        pointer.data = torch.from_numpy(array)

    # Save pytorch-model
    torch.save(model.state_dict(), args.pytorch_dump_path)


if __name__ == "__main__":
    convert(args)


# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import os
# import re
# import argparse
# import tensorflow as tf
# import torch
# import numpy as np
#
# from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
#
#
# def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
#     # Initialise PyTorch model
#     config = BertConfig.from_json_file(bert_config_file)
#     print("Building PyTorch model from configuration: {}".format(str(config)))
#     model = BertForPreTraining(config)
#
#     # Load weights from tf checkpoint
#     load_tf_weights_in_bert(model, tf_checkpoint_path)
#
#     # Save pytorch-model
#     print("Save PyTorch model to {}".format(pytorch_dump_path))
#     torch.save(model.state_dict(), pytorch_dump_path)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # Required parameters
#     b_path = '/Users/liangsong/Desktop/publish'
#     parser.add_argument("--tf_checkpoint_path",
#                         default=b_path+'/bert_model.ckpt',
#                         type=str,
#                         help="Path the TensorFlow checkpoint path.")
#     parser.add_argument("--bert_config_file",
#                         default=b_path+'/bert_config.json',
#                         type=str,
#                         help="The config json file corresponding to the pre-trained BERT model. \n"
#                              "This specifies the model architecture.")
#     parser.add_argument("--pytorch_dump_path",
#                         default=b_path+"/pytorch_model.bin ",
#                         type=str,
#                         help="Path to the output PyTorch model.")
#     args = parser.parse_args()
#
#     convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path,
#                                      args.bert_config_file,
#                                      args.pytorch_dump_path)
