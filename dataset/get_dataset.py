import argparse
import numpy as np
import os
parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input file")
parser.add_argument("-out","--path_output", type=str, help="the path of output file")
parser.add_argument("-dt","--data_type", type=str, help="the data type of feature")
parser.add_argument("-maxseq","--max_sequence", type=int, default=0,help="the maxseq of feature")


def loadData(path):
    if path.endswith(".npy"):
        data = np.load(path)
        print(data.shape)
        if data.ndim == 3 and data.shape[0] == 1:
            data = np.squeeze(data, axis=0)   # (1, L, 768) → (L, 768)
        # 已經是 (L, 768) 就不動
        print(data.shape)
        return data
    else:
        data = np.loadtxt(path)
        print(data.shape)
        return data

    
    

def saveData(path,data):
    #data= data[:, np.newaxis, :, :]
    print(data.shape)
    np.save(path, data)

def get_series_feature(org_data, maxseq ,length):
    data = np.zeros((maxseq, length), dtype=np.float16)
    
    # 获取org_data的长度
    data_len = len(org_data)
    if data_len < maxseq:
        # 如果org_data的长度小于maxseq，只复制org_data的数据到新数组中
        data[:data_len, :] = org_data
    else:
        # 如果org_data的长度大于等于maxseq，只复制前maxseq个数据到新数组中
        data[:, :] = org_data[:maxseq, :]
    print(data.shape)
    data = data.reshape((1, 1, maxseq, length))    
    return data

def main(path_input, path_output, data_type,maxseq,length):
    result=[]
    input=os.listdir(path_input)
    for i in input:
        if i.endswith(data_type):
            file_name=i.split(".")[0]
            data = loadData(path_input+"/"+file_name+data_type)
            result.append( get_series_feature(data, maxseq,length))
    data = np.concatenate(result, axis=0)
    saveData(path_output, data)
        

if __name__ == "__main__":
    args = parser.parse_args()
    path_input = args.path_input
    path_output = args.path_output
    data_type=args.data_type
    maxseq=args.max_sequence
    if args.data_type == ".prottrans":
        length = 1024
    elif args.data_type == ".esm":
        length = 1280
        #length = 2560
    elif args.data_type == ".npy":
        length = 768
    else:
        length = 20
    main(path_input, path_output,data_type,maxseq,length)




