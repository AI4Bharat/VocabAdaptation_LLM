import os

folder_fil = ["eng_Latn-hin_Deva", "eng_Latn-tam_Taml" ]
# for folder in os.listdir("/data-3/nandini/vocab_adapt/codes/bpcc_combine/"):
for folder in folder_fil:
    print(folder)
    # folder = 'eng_Latn-eng_Latn'
    # text_path = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/eng_Latn-hin_Deva/train.{folder[-8:]}"
    text_path_in = f"/data/nandini/vocab_adapt/codes/bpcc_combine/{folder}/train.{folder[-8:]}"
    text_path_en = f"/data/nandini/vocab_adapt/codes/bpcc_combine/{folder}/train.eng_Latn"
    with open(text_path_in,'r', encoding='utf-8' ) as file:
        data_in = file.readlines()
    
    with open(text_path_en,'r', encoding='utf-8' ) as file:
        data_en = file.readlines()
    
    if len(data_in) == len(data_en):
        print("yes sie is same")
    else:
        print("no it is not")
    print(len(data_en))
    
    
    data_target=[]
    for i in range(10000000):
        text_in = data_in[i].strip()
        text_en = data_en[i].strip()
        # print("indian language text is ", text_in)
        # print("indian language text is ", text_en)
        t = text_en + " ||| " + text_in + "\n"   #according to the input format of fastAlign
        data_target.append(t)
      

    train_path = f"/data/nandini/vocab_adapt/codes/fast_align_data_new/merge.{folder[-8:]}"
    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(data_target)

    
