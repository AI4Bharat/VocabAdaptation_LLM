import sys
import random, os

## Read in a source language file, a target language file, and an alignment file.
## In the target language file, replace some words with words in the source file using the alignments.
## Write the resulting file to stdout.
for folder in os.listdir("/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/"):
    src_file = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.eng_Latn"
    tgt_file = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/BPCC-seed/{folder}/train.{folder[-8:]}"
    align_file = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/fast_align/build/{folder[-8:]}.align"
    code_mixed_file = f"/nlsasfs/home/ai4bharat/nandinim/nandini/vocab_adap/code_mix_data/en_{folder[-8:]}.txt"
    code_mixed_file = open(code_mixed_file, 'w')
    # max_span_to_replace = float(sys.argv[5])
    max_span_to_replace = float(2)

    for src_line, tgt_line, align_line in zip(open(src_file), open(tgt_file), open(align_file)):
        src_line = src_line.strip()
        tgt_line = tgt_line.strip()
        align_line = align_line.strip()
        if src_line == "" or tgt_line == "" or align_line == "":
            continue
        src_words = src_line.split()
        tgt_words = tgt_line.split()
        align_words = align_line.split()
        align_dict = {}
        for align_word in align_words:
            align_word = align_word.split("-")
            align_word[0] = int(align_word[0])
            align_word[1] = int(align_word[1])
            if align_word[0] not in align_dict:
                align_dict[align_word[0]] = [align_word[1]]
            else:
                align_dict[align_word[0]].append(align_word[1])
        # Get a source word span to replace.
        src_span = random.randint(0, len(src_words) - 1)
        src_span_start = src_span
        max_span_to_replace_curr = max(1, int(max_span_to_replace*len(src_words)))
        src_span_end = src_span + random.randint(1, max_span_to_replace_curr)
        if src_span_end >= len(src_words):
            src_span_end = len(src_words) - 1
        # Get aligned target words.
        tgt_span = []
        for i in range(src_span_start, src_span_end + 1):
            if i in align_dict:
                tgt_span.extend(align_dict[i])
        #print(" ".join(src_words[src_span_start:src_span_end + 1]), tgt_span, align_dict, src_words, tgt_words)
        if len(tgt_span) == 0:
            code_mixed_file.write(" ".join(tgt_words) + "\n")
            continue
        max_pos = max(tgt_span)
        min_pos = min(tgt_span)
        # Replace target span with source span.
        tgt_words = tgt_words[:min_pos] + src_words[src_span_start:src_span_end + 1] + tgt_words[max_pos + 1:]
        # Write to file.
        code_mixed_file.write(" ".join(tgt_words) + "\n")
    
        
