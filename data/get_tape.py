#!/usr/bin/env python
# coding: utf-8

import os
import torch
from tape import ProteinBertModel, TAPETokenizer
from tqdm import tqdm
import numpy as np
import glob
import argparse

# 設定參數解析器
parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="the path of input fasta file")
parser.add_argument("-out", "--path_output", type=str, help="the path of output esm file")

def main(input_folder, out_folder, miss_txt):
    input_files = glob.glob(os.path.join(input_folder, "*"))
    
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
    
    for path in tqdm(input_files, desc="Processing", unit="file"):
        try:
            with open(path) as f:
                fasta = f.readlines()

            if len(fasta) < 2:
                print(f"文件 {path} 格式錯誤，跳過")
                continue

            title = fasta[0][1:].strip()
            sequence = fasta[1].strip()
            out_path = os.path.join(out_folder, title)

            # Tokenize the sequence
            token_ids = torch.tensor([tokenizer.encode(sequence)])
            
            # Process sequence with the model
            output = model(token_ids)
            sequence_output = output[0][:, 1:-1, :].cpu().detach().numpy()
            
            # Save the processed sequence output
            np.save(out_path + ".npy", sequence_output)
        
        except Exception as e:
            log_mode = 'a' if os.path.exists(miss_txt) else 'w'
            with open(miss_txt, log_mode) as tape_miss:
                tape_miss.write(title + ".fasta\n")
            print(f"處理 {title} 時發生錯誤：{e}")
            continue

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.path_input, args.path_output, "tape_miss")
