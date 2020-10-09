import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument(
        '--input-path', type=str, required=True,
    )
    parser.add_argument(
        '--output-path', type=str, required=True,
    )
    parser.add_argument(
        '--use-date', type=str, default=None,
    )

    return parser.parse_args()


def load_tokens(data_dir):

    tokens = []
    for file in tqdm(os.listdir(data_dir)):
        if not file.endswith('.dag'):
            continue

        with open(os.path.join(data_dir, file), 'r') as f:
            sent_idx = 0

            for i, line in enumerate(f):
                line = line.strip('\n')

                if i == 0:
                    year = line
                elif line == '':
                    sent_idx += 1
                else:
                    ls = line.split('\t')
                    ls[0] = int(ls[0])
                    ls[1] = int(ls[1])
                    tokens.append([file, year, sent_idx] + ls)
    
    return tokens


def aggregate_tokens_into_words(tokens):

    prev_file = None
    prev_sent = None
    current_end = 0

    word_idx = -1
    words = []
    for token in tokens:
        file, sent, beg, end = token[0], token[2], token[3], token[4]

        assert beg <= current_end, 'jest przerwa'

        if prev_file != file or prev_sent != sent or beg == current_end:
            current_end = end
            word_idx += 1
            words.append([])
        else:
            current_end = max(current_end, end)

        prev_file = file
        prev_sent = sent
        words[-1].append(token + [str(word_idx)])

    return words


def extract_singular_subtokens(word):
    # note: sometimes words has longer span than one, but no segmentation
    # this might be a leak

    if len(set([t[5] for t in word])) == 1:
        return {word[0][3]: word[0][5]}
    
    subtokens = {}
    for idx1, token1 in enumerate(word):
        for idx2, token2 in enumerate(word):
            beg1, end1, seg1 = token1[3], token1[4], token1[5]
            beg2, end2, seg2 = token2[3], token2[4], token2[5]

            if idx1 == idx2 and (end1 - beg1) == 1:
                subtokens[beg1] = seg1
            elif beg1 == beg2 and end1 + 1 == end2:
                subtokens[end1] = seg2[len(seg1):]
            elif beg1 == beg2 + 1 and end1 == end2:
                subtokens[beg2] = seg2[:-len(seg1)]
    
    if len(subtokens) != (max(subtokens.keys()) - min(subtokens.keys()) + 1):
        # in case of missing subtoken try to extract using existing subtokens

        for token in word:
            beg, end, seg = token[3], token[4], token[5]
            for idx, subtoken in sorted(subtokens.items()):
                if beg >= end:
                    break
                if idx != beg:
                    continue
                else:
                    seg = seg[len(subtoken):]
                    beg += 1

            for idx, subtoken in sorted(subtokens.items(), reverse=True):
                if beg >= end:
                    break
                if idx + 1 != end:
                    continue
                else:
                    seg = seg[:-len(subtoken)]
                    end -= 1

            assert end - beg <= 1, 'nadal zle'
            if end - beg == 1:
                assert len(seg) > 0
                subtokens[beg] = seg

    assert len(subtokens) == (max(subtokens.keys()) - min(subtokens.keys()) + 1), f'brakuje indeksu: {word}, {subtokens}'

    return subtokens


def extract_disamb_subtokens(word, subtokens):
    
    output_subtokens = []
    for token in word:
        if token[9] not in ['disamb', 'disamb_manual']:
            continue

        for i in range(token[3], token[4]):
            output_subtoken = token.copy()
            output_subtoken[3] = i
            output_subtoken[4] = i + 1
            output_subtoken[5] = subtokens[i]
            output_subtokens.append(output_subtoken)

    return output_subtokens


def extract_all_subtokens(word, word_idx, subtokens):
    file, year, sent_idx = word[0][:3]
    
    output_subtokens = []
    for beg_idx, subtoken in subtokens.items():
        output_subtokens.append([
            file, year, sent_idx,
            beg_idx, beg_idx + 1, subtoken,
            '_', '_', '_', '_', str(word_idx),
        ])
    
    return output_subtokens


def create_dataset(tokens, disamb=True):
    words = aggregate_tokens_into_words(tokens)
    
    output = []
    for word_idx, word in tqdm(enumerate(words)):
        subtokens = extract_singular_subtokens(word)
        if disamb:
            correct_subtokens = extract_disamb_subtokens(word, subtokens)
        else:
            correct_subtokens = extract_all_subtokens(word, word_idx, subtokens)
        output += correct_subtokens
        
    return output


def to_conll(dataset, path, use_date=None):
    
    with open(path, 'w') as f:
        prev_sent_id = None
        
        for token in dataset:
            file, year, sent, beg, end, seg, lemma, tag, nps, target, word_idx = token
            sent_id = (file, sent)      
            
            if use_date == 'as_upostag':
                upostag = year[1:].split(':')[0][:3]
            else:
                upostag = '_'
            
            # new sentence
            if sent_id != prev_sent_id:
                if prev_sent_id is not None:
                    f.write('\n')
                f.write(f'# {file}\n')
                if use_date == 'as_token':
                    y3 = year[1:].split(':')[0][:3]
                    f.write('\t'.join([
                        '_',      # id
                        y3,       # form
                        '_',      # lemma
                        '_',      # upostag
                        '_',      # xpostag
                        '_',      # feats
                        '0',      # head
                        'root',   # deprel
                        '_',      # deps
                        '_',      # misc
                    ]))
                    f.write('\n')
                    
                    
            prev_sent_id = sent_id
            
            # write token
            f.write('\t'.join([
                str(beg), # id
                seg,      # form
                lemma,    # lemma
                upostag,  # upostag
                tag,      # xpostag
                word_idx, # feats
                '0',      # head
                'root',   # deprel
                '_',      # deps
                '_',      # misc
            ]))
            f.write('\n')
                    
        f.write('\n')


if __name__ == '__main__':
    args = parse_args()
    
    tokens = load_tokens(args.input_path)
    data = create_dataset(tokens, disamb='disamb' in args.input_path)
    to_conll(data, args.output_path, use_date=args.use_date)
