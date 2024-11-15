## we handle each of the common ancestor
import sys
import csv
import os
import gzip
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def main(ancestor,singleton_dir,eggnog_dir,save_dir):

    group_file = f'{eggnog_dir}/{ancestor}.tsv.gz'

    singletons = dict()

    ## load all the singletons as a dict
    print('loading singletons')
    for f in tqdm(list(os.listdir(singleton_dir))):
        f = os.path.join(singleton_dir,f)
        s = open(f).read().strip().split('\n')
        species = f.split('/')[-1].split('.')[0]
        singletons[species] = set(s)

    records = list()

    print('parsing orthologs')
    ## get the length of the file
    length = os.popen(f'zcat {group_file} | wc -l').read().strip()
    length = int(length)

    print('length:',length)

    with gzip.open(group_file,'rt') as f:

        for line in tqdm(f,total=length):
            line = line.strip().split('\t')
            orthologs = set(line[-1].split(','))
            og_name = line[1]
            species = line[4].split(',')

            og_singletons = list()
            
            ## singleton in this og
            for o in orthologs:
                s = o.split('.')[0]
                ## check if this is a singleton
                if o in singletons[s]:
                    og_singletons.append(o)
            
            if len(og_singletons) == 0:
                continue    

            ## check if all the orthologs are singletons
            if len(og_singletons) == len(orthologs):
                records.append((og_name,','.join(species),','.join(og_singletons),'','non-interaction'))
            else:
                non_singleton_og = orthologs - set(og_singletons)
                records.append((og_name,','.join(species),','.join(og_singletons),','.join(non_singleton_og),'partial-interaction'))
    print('saving')

    ## save as a gzipped file
    with gzip.open(f'{save_dir}/{ancestor}.tsv.gz','wt') as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerow(['og_name','species','singletons','non_singleton_orths','interaction'])
        writer.writerows(records)
    print('done')
                


if __name__ == '__main__':

    ancestor = int(sys.argv[1])

    singleton_dir = sys.argv[2]

    eggnog_dir = sys.argv[3]

    save_dir = sys.argv[4]

    main(ancestor,singleton_dir,eggnog_dir,save_dir)