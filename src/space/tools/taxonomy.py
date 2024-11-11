import os
import pandas as pd
import gzip

## for some species, the taxid used in STRING is different from the one used in ncbi
__TAXID_CONVERTER__ = {339724:2059318,     
    1745343:2082293,
    1325735:2604345,
    1658172:1955775,
    56484:2754530,
    1266660:1072105,
    1229665:2502994,
    944018:1913371,
    743788:2126942
}


class Lineage:
    
    def __init__(self,node_dmp_zip,group_dir) -> None:

        self.df = pd.read_csv(node_dmp_zip,sep='|',compression='zip',header=None)
        self.eggnog_ancestors = {f.split('.')[0] for f in os.listdir(group_dir) }
        self.group_dir = group_dir

    def get_lineage(self,taxid):
        taxid = int(taxid)
        l = [taxid]
        while taxid != 1:
            taxid = self.df[self.df.iloc[:,0]==taxid].iloc[0,1]
            l = [taxid] + l
        return l

    def common_ancestor(self,taxid_1,taxid_2):


        taxid_1 = int(taxid_1)
        taxid_2 = int(taxid_2)

        use_taxid_1 = __TAXID_CONVERTER__.get(taxid_1,taxid_1)
        use_taxid_2 = __TAXID_CONVERTER__.get(taxid_2,taxid_2)

        l_1 = self.get_lineage(use_taxid_1)
        l_2 = self.get_lineage(use_taxid_2)
        for idx,taxid in enumerate(l_1):
            if taxid in l_2:
                common_ancestor = taxid
            else:
                break
        
        ## make sure eggNOG has the common ancestor, and the orthologs are not empty 
        while True:
            if self.check_ortholog_group(taxid_1,taxid_2,common_ancestor):
                break
            idx -= 1
            common_ancestor = l_1[idx-1]
        
        return str(taxid_1),str(taxid_2),int(common_ancestor)
    

    def check_ortholog_group(self,taxid_1,taxid_2,ancestor):

        group_file = f'{self.group_dir}/{ancestor}.tsv.gz'

        if not os.path.exists(group_file):
            return False

        with gzip.open(group_file,'rt') as f:

            for line in f:
                line = line.strip().split('\t')
                species_list = line[-2].split(',')

                if str(taxid_1) in species_list and str(taxid_2) in species_list:
                    return True
        
        return False
