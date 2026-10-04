import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-in", "--path_input", type=str, help="the path of input file")
parser.add_argument("-out", "--path_output", type=str, help="the path of output file")
parser.add_argument("-dt", "--data_type", type=str, help="the data type of feature")
parser.add_argument("-maxseq", "--max_sequence", type=int, default=0, help="the maxseq of feature")

def loadData(path):
    if path.endswith(".npy"):
        data = np.load(path)
        print(f"Loading {path}, shape: {data.shape}")
        # 如果維度是 (1, N, M)，則壓縮第一維
        if len(data.shape) == 3 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)
        return data
    else:
        data = np.loadtxt(path)
        print(f"Loading {path}, shape: {data.shape}")
        return data

def saveData(path, data):
    # 如果路徑沒帶 .npy，np.save 會自動補上，但為了明確我們手動處理
    if not path.endswith(".npy"):
        path = path + ".npy"
    print(f"Saving combined data to {path}, final shape: {data.shape}")
    np.save(path, data)

def get_series_feature(org_data, maxseq, length):
    data = np.zeros((maxseq, length), dtype=np.float16)
    
    # 獲取 org_data 的長度 (序列長度)
    data_len = len(org_data)
    if data_len < maxseq:
        # 如果長度不足，填入現有的，剩餘部分為零 (padding)
        data[:data_len, :] = org_data
    else:
        # 如果超出，則截斷
        data[:, :] = org_data[:maxseq, :]
    
    # 調整維度為 (1, 1, maxseq, length) 方便後續 concatenate
    data = data.reshape((1, 1, maxseq, length))    
    return data

def main(path_input, path_output, data_type, maxseq, length):
    result = []
    file_names = [] 
    
    if not os.path.exists(path_input):
        print(f"Error: Input path {path_input} does not exist.")
        return

    input_files = os.listdir(path_input)
    # 排序檔案名稱，確保順序穩定
    input_files.sort() 

    for i in input_files:
        if i.endswith(data_type):
            file_name = i.split(".")[0]
            data = loadData(os.path.join(path_input, i))
            result.append(get_series_feature(data, maxseq, length))
            file_names.append(file_name) 

    if not result:
        print("No files found with the specified data type.")
        return

    # 1. 儲存特徵矩陣 (.npy)
    data_all = np.concatenate(result, axis=0)
    saveData(path_output, data_all)

    # 2. 儲存對應的 ID 清單 (.txt)
    # 邏輯：直接拿 path_output 並加上 .txt 副檔名
    # 這樣如果 -out 是 folder_001_15，就會產出 folder_001_15.txt
    id_output_path = path_output + ".txt"
    
    # 確保輸出目錄存在
    output_dir = os.path.dirname(id_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(id_output_path, 'w') as f:
        for name in file_names:
            f.write(f"{name}\n")
            
    print(f"IDs saved to: {id_output_path}")

if __name__ == "__main__":
    args = parser.parse_args()
    
    # 根據資料類型決定特徵長度
    if args.data_type == ".prottrans":
        length = 1024
    elif args.data_type == ".esm":
        length = 2560
        #length = 1280
    elif args.data_type == ".npy":
        length = 768
    else:
        length = 20
        
    main(args.path_input, args.path_output, args.data_type, args.max_sequence, length)