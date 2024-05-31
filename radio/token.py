import torch
import csv
model_dir = "../model/chatTTS"
std, mean = torch.load(model_dir + '/asset/spk_stat.pt').chunk(2)
rand_spk = torch.randn(768) * std + mean

def writeToCsv(csv_file_path,data):
  with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入数据
    writer.writerow(data.tolist())

writeToCsv(f"saved.csv",rand_spk.detach().numpy())