import os

to_write = open(f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/combine_bilingual_dict.txt", 'w', encoding='utf-8')
unique= set()
for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/align_dict/"):
    print(folder)
    dic_file = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/align_dict/{folder}"
    with open(dic_file,'r', encoding='utf-8' ) as file:
        bi_dict = file.readlines()
    
    count =0
    i =0
    # for i in range(100):
    while i < len(bi_dict) and count<500:
        bi_dict[i] = bi_dict[i].strip()
        # print(bi_dict[i], " size is: ", len(bi_dict[i]))
        words = bi_dict[i].split('\t')
        i +=1
        if words[0] not in unique:
            if len(words[1]) !=0:
                unique.add(words[0])
                to_write.write( words[0]+ "\t" + words[1]+"\n")
                count +=1
    print(count)
    
