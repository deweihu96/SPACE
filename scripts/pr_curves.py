import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    custom_dashes = [5, 8]
    title_size = 14
    tick_size = 14
    label_size = 14
    legend_fontsize = 14


    df_cv = pd.read_csv("data/benchmark_results/sub_loc/cv_pr_curves.csv")
    df_hpa = pd.read_csv("data/benchmark_results/sub_loc/hpa_pr_curves.csv")

    ## number of positive samples
    pos_cv = pd.read_csv('data/benchmark_results/sub_loc/cv_data.csv')
    pos_num = pos_cv.iloc[:,1:-1].sum().values.sum()
    pos_num/(pos_cv.iloc[:,1:-1].shape[0]*len(pos_cv.columns[1:-1]))



    plt.figure(figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3,hspace=0.37)  # Increase vertical space between plots

    plt.subplot(2, 3, 1)
    colors = ['#1e1e1e','#bcbcbc',"#e60049",]
    plt.plot(df_cv['space_recall'], df_cv['space_prec'], label='SPACE', color=colors[2], linestyle='-', linewidth=2)
    plt.plot(df_cv['aligned_recall'], df_cv['aligned_prec'], label='Aligned', color=colors[0], linestyle='-', linewidth=2)
    plt.plot(df_cv['t5_recall'], df_cv['t5_prec'], label='ProtT5', color=colors[1], linestyle='-', linewidth=2)


    plt.yticks(np.arange(0,1.1,0.2),fontsize=tick_size)
    plt.xticks(np.arange(0,1.1,0.2),fontsize=tick_size)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall',fontsize=label_size)
    plt.ylabel('Precision',fontsize=label_size)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.title("SwissProt Cross Validation Set",fontsize=title_size)

    ## number of positive samples
    pos_hpa = pd.read_csv('data/benchmark_results/sub_loc/hpa_testset_mapped.csv')
    pos_num = pos_hpa.iloc[:,1:-1].sum().values.sum()
    pos_num/(pos_hpa.iloc[:,1:-1].shape[0]*len(pos_hpa.columns[1:-1]))

    plt.subplot(2, 3, 2)
    plt.plot(df_hpa['space_recall'], df_hpa['space_prec'], label='SPACE', color=colors[2], linestyle='-', linewidth=2)
    plt.plot(df_hpa['aligned_recall'], df_hpa['aligned_prec'], label='Aligned', color=colors[0], linestyle='-', linewidth=2)
    plt.plot(df_hpa['t5_recall'], df_hpa['t5_prec'], label='ProtT5', color=colors[1], linestyle='-', linewidth=2)
    # deeploc2 at the best cutoffs per label
    # Precision:  0.5763269140441521
    # Recall:  0.6147294589178357
    plt.scatter(0.6147294589178357, 0.5763269140441521, s=120, color="#0bb4ff", marker='*',)

    plt.yticks(np.arange(0,1.1,0.2),fontsize=tick_size)
    plt.xticks(np.arange(0,1.1,0.2),fontsize=tick_size)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Recall',fontsize=label_size)
    plt.ylabel('Precision',fontsize=label_size)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.annotate('DeepLoc 2.0',  # Text
                xy=(0.6147294589178357, 0.5763269140441521),  # Point to annotate
                xytext=(0.5, 0.5),  # Text position
                fontsize=12,
                )

    # have a horizontal legend in the middle bottom, no border
    plt.legend(loc='lower center', bbox_to_anchor=(-0.25, -0.3), 
            ncol=4, frameon=False, fontsize=legend_fontsize,
            )

    plt.title("HPA Test Set",fontsize=title_size)

    # plt.tight_layout()
    # plt.savefig("../manuscript_figs/fig3_a.png", dpi=300, bbox_inches='tight')

    pr_rc_dir = 'data/benchmark_results/func_pred/cafa-eval-index'
    aspects = ['mf', 'bp','cc']
    netgo_scores = pd.read_csv('data/benchmark_results/func_pred/scores.csv')
    # plt.figure(figsize=(15, 8))
    label_dict = {'aligned':"Aligned", 'seq':"ProtT5", 'space':"SPACE"}
    aspect_dict = {'cc':"Cellular Component", 'bp':"Biological Process", 'mf':"Molecular Function"}
    for aspect in aspects:
        plt.subplot(2, 3, aspects.index(aspect)+4)
        colors = ["#e60049",'#1e1e1e','#bcbcbc',]
        for idx,emb in enumerate(['space','aligned','seq',]):

            if emb == 'space':
                emb_ = 'seq_concat_aligned'
            else:
                emb_ = emb

            df = pd.read_csv(f'{pr_rc_dir}/{aspect}_{emb_}_merged.csv')
            rc,pr = df['rc'].values, df['pr'].values
            ## make sure both rc and pr are 
            
            plt.plot(df['rc'], df['pr'], label=label_dict[emb], 
                    linestyle='-', linewidth=2, color=colors[idx])

            pr,rc = netgo_scores[netgo_scores['entry']==aspect+'_'+emb_][['pr','rc']].values[0]
            ## annotate this point, with 'x'
            plt.scatter(rc, pr, s=80, color=colors[idx], marker='*')
            plt.xlabel('Recall',fontsize=label_size)
            plt.ylabel('Precision',fontsize=label_size)
            plt.title(aspect_dict[aspect],fontsize=title_size)  
            plt.xticks(np.arange(0,1.1,0.2),fontsize=tick_size)
            plt.yticks(np.arange(0,1.1,0.2),fontsize=tick_size)
            plt.xlim(0,1)
            plt.ylim(0,1)
            ## get rid of the up and right spines
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            if aspect == 'bp':
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
                        fontsize=legend_fontsize)

    # plt.tight_layout()
    plt.savefig('results/pr_curves.png',dpi=300, bbox_inches='tight')
    plt.show()
