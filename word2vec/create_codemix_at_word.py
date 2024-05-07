import sys
import random

## Read in a source language file, a target language file, and an alignment file.
## In the target language file, replace some words with words in the source file using the alignments.
## Write the resulting file to stdout.

# src_file = sys.argv[1]
# tgt_file = sys.argv[2]
# align_file = sys.argv[3]
# code_mixed_file = sys.argv[4]
# code_mixed_file = open(code_mixed_file, 'w')
max_span_to_project = float(0.4) ## This indicates a maximum span to project from the source to the target at a time. Value should be between 0 and 1. If 0, then we project only one word at a time. If 1, then at max we project the entire source sentence to the target sentence.
max_percentage_to_project = float(0.5) ## This indicates a maximum total percentage of the source sentence to project to the target sentence. Counted as total number of words in the source sentence projected to the target sentence divided by the total number of words in the source sentence. If 0, then we project only one word at a time. If 1, then at max we project the entire source sentence to the target sentence.

max_retries = 1000

folder_fil = ["merge.hin_Deva" ]
for folder in folder_fil:
    src_tgt_file = f"/data/nandini/vocab_adapt/codes/fast_align_data_new/{folder}"
    align_file = f"/data/nandini/vocab_adapt/codes/fast_align/build/{folder[-8:]}_symmetric.align"
    code_mixed_file = f"/data/nandini/vocab_adapt/codes/code_mix_data_new/word_en_{folder[-8:]}_1.txt"
    code_mixed_file = open(code_mixed_file, 'w')
    
    cnt_to_write = 0
    for src_tgt_line, align_line in zip(open(src_tgt_file), open(align_file)):
        curr_retries = 0
        src_tgt_line = src_tgt_line.strip()
        src_line , tgt_line = src_tgt_line.split(" ||| ")
        align_line = align_line.strip()
        print("source line is: ", src_line)
        print("target line is: ", tgt_line)
        print("align line is: ", align_line)
        code_mixed_file.write(src_line + "\n" + tgt_line + "\n" + align_line + "\n")
        if src_line == "" or tgt_line == "" or align_line == "":
            continue
        src_words = src_line.split()
        tgt_words = tgt_line.split()
        align_words = align_line.split()
        print("source word is: ", src_words)
        print("target word is: ", tgt_words)
        print("align word is: ", align_words)
        max_span_diff = 0
        align_dict = {}                          #src to target that is english to target
        for align_word in align_words:
            align_word = align_word.split("-")
            align_word[0] = int(align_word[0])    #align_word[0] is of english language (source language)
            align_word[1] = int(align_word[1])    ##align_word[1] is of target language
            max_span_diff = max(max_span_diff, align_word[1])
            if align_word[0] not in align_dict:
                align_dict[align_word[0]] = [align_word[1]]
            else:
                align_dict[align_word[0]].append(align_word[1])
        # Get a source word span to replace.
        curr_src_sentence_projected = 0
        max_current_src_sentence_projected = int(max_percentage_to_project*len(src_words))
        src_spans_covered = []
        tgt_spans_covered = []
        continue_to_next = False
        while True:
            curr_retries += 1
            if curr_retries >= max_retries:
                continue_to_next = True
                print("breaking in curr_retries ")
                break
            if curr_src_sentence_projected >= max_current_src_sentence_projected:
                continue_to_next = True
                print("breaking in curr_src_sentence projected ")
                break
            src_span_start = random.randint(0, len(src_words) - 1) ## Choose a random point in the source sentence to start projecting from.
            max_span_to_project_curr = max(1, int(max_span_to_project*len(src_words))) ## We require that a span of 1 be the minimum.
            src_span_end = src_span_start + random.randint(1, max_span_to_project_curr)-1 ## Choose a random span to project from.
            if src_span_end >= len(src_words):
                src_span_end = len(src_words) - 1
            
            print("src_span start is: ", src_span_start, " max span to project curr is: ", max_span_to_project_curr)
            print("src_span_end is: ",src_span_end )

            print("current curr_retries is: ", curr_retries)
            print("src span covered is:: ", src_spans_covered)
            
            ## Check overlap with previous spans.
            overlap = False
            for span in src_spans_covered:
                if (src_span_start >= span[0] and src_span_start <= span[1]) or (src_span_end >= span[0] and src_span_end <= span[1]):
                    overlap = True
                    break
            
            ## If there is overlap then choose another span.
            if overlap:
                continue
            # Get aligned target words.
            tgt_span = []
            break_flag = False
            for i in range(src_span_start, src_span_end+1): 
                if i in align_dict:
                    tgt_span.extend(align_dict[i])                  ########why? target words should not be there in tgt_span
                    # print("align_dixt is: ", align_dict[i])
                    diff_src_tgt = abs(i- max(align_dict[i]))
                    if diff_src_tgt > max_span_diff * 0.75:
                        break_flag = True
                        break
            if break_flag:
                print("in break flag")
                continue

            if len(tgt_span) == 0:
                continue
            max_pos = max(tgt_span)                               #######ye target mei
            min_pos = min(tgt_span)                               ########ye target mei
            # Check if the target span overlaps with any previous target span.
            overlap = False
            for span in tgt_spans_covered:
                if (min_pos >= span[0] and min_pos <= span[1]) or (max_pos >= span[0] and max_pos <= span[1]):
                    overlap = True
                    break
            if overlap:
                continue
            src_spans_covered.append((src_span_start, src_span_end))
            tgt_spans_covered.append((min_pos, max_pos))
            curr_src_sentence_projected += src_span_end - src_span_start + 1

        if src_spans_covered == []:
            tgt_words_final = tgt_words
        else:
            src_tgt_spans = zip(src_spans_covered, tgt_spans_covered)
            print("src_span_covered is: ", src_spans_covered)
            print("tgt_spans_covered is: ", tgt_spans_covered)
            print("src tgt span before sort is: ", src_tgt_spans )
            ## Now sort according to the target span.
            src_tgt_spans = sorted(src_tgt_spans, key = lambda x: x[1][0])
            print("src tgt span after sort is: ", src_tgt_spans )
            ## Now we do the replacements
            prev_tgt_span_end = 0
            tgt_words_final = []
            for src_span, tgt_span in src_tgt_spans:
                print("in for loop src_span is::: ", src_span)
                print("in for loop tgt_span is::: ", tgt_span)
                src_span_start, src_span_end = src_span
                tgt_span_start, tgt_span_end = tgt_span
                min_pos = tgt_span_start
                max_pos = tgt_span_end
                tgt_words_final += tgt_words[prev_tgt_span_end:min_pos] + src_words[src_span_start:src_span_end + 1]
                prev_tgt_span_end = max_pos + 1
            tgt_words_final += tgt_words[prev_tgt_span_end:]

        # Write to file.
        code_mixed_file.write(" ".join(tgt_words_final) + "\n" + "\n")
        cnt_to_write += 1
        if cnt_to_write == 10:
            break
    
