import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    
    
    argparser = argparse.ArgumentParser(description="Plot PR curves for SPACE and other methods.")
    argparser.add_argument('--cv_data', type=str, default='data/benchmark_results/sub_loc/cv_data.csv',
                        help='Path to the cross-validation data CSV file.')
    argparser.add_argument('--cv_pr_curves', type=str, default='data/benchmark_results/sub_loc/cv_pr_curves.csv',
                        help='Path to the cross-validation PR curves CSV file.')
    argparser.add_argument('--hpa_pr_curves', type=str, default='data/benchmark_results/sub_loc/hpa_pr_curves.csv',
                        help='Path to the HPA PR curves CSV file.')
    argparser.add_argument('--cafa_eval_index', type=str, default='data/benchmark_results/func_pred/cafa-eval-index',
                        help='Path to the CAFA evaluation index directory.')
    argparser.add_argument('--scores_csv', type=str, default='data/benchmark_results/func_pred/scores.csv',
                        help='Path to the scores CSV file for functional prediction.')
    argparser.add_argument('--output', type=str, default='results/pr_curves.png',
                        help='Output path for the PR curves plot.')
    args = argparser.parse_args()
    

    custom_dashes = [5, 8]
    title_size = 14
    tick_size = 14
    label_size = 14
    legend_fontsize = 14

    df_cv = pd.read_csv(args.cv_pr_curves)
    df_hpa = pd.read_csv(args.hpa_pr_curves)

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


    aspects = ['mf', 'bp','cc']

    netgo_scores = pd.read_csv(args.scores_csv)
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

            df = pd.read_csv(f'{args.cafa_eval_index}/{aspect}_{emb_}_merged.csv')
            rc,pr = df['rc'].values, df['pr'].values
            ## make sure both rc and pr are 
            
            plt.plot(df['rc'], df['pr'], label=label_dict[emb], 
                    linestyle='-', linewidth=2, color=colors[idx])
            
            pr,rc = netgo_scores[netgo_scores['entry']==aspect+'_'+emb_+'_merged'][['pr','rc']].values[0]
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

    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"PR curves saved to {args.output}")

