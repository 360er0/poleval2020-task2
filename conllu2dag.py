import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Task
    parser.add_argument(
        '--input-path', type=str, required=True,
    )
    parser.add_argument(
        '--output-path', type=str, required=True,
    )

    return parser.parse_args()


def conllu2dag(conllu_file, dag_dir):

    outputs = []
    with open(conllu_file, 'r') as f:        
        for line in f:
            line = line.strip()
            
            if line.startswith('#'):
                outputs.append({
                    'filename': dag_dir + line[2:],
                    'data': [],
                })
            elif line == '':
                continue
            else:
                beg, seg, lemma, year, tag, word = line.split('\t')[:6]
                if beg == '_':  # as_token
                    continue
                
                data = outputs[-1]['data']
                if len(data) > 0 and data[-1]['tag'] == tag and data[-1]['word'] == word:
#                     data[-1]['end'] = str(int(data[-1]['end']) + 1)
                    data[-1]['seg'] += seg
                else:
                    data.append({
                        'beg': beg,
#                         'end': end,
                        'seg': seg, 
                        'lemma': lemma, 
                        'tag': tag, 
                        'word': word,
                    })

    if os.path.isdir(dag_dir):
        shutil.rmtree(dag_dir)
    else:
        os.mkdir(dag_dir)
        
    for output in outputs:
        with open(output['filename'], 'a') as f:
            for i, row in enumerate(output['data']):
                if (i + 1) < len(output['data']):
                    end = output['data'][i + 1]['beg']
                else:
                    end = str(int(row['beg']) + 1)
                
                f.write('\t'.join([
                    row['beg'], end,
                    row['seg'], row['lemma'],
                    row['tag'],
                    '', 'disamb',
                ]))
                f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    args = parse_args()

    conllu2dag(args.input_path, args.output_path)
