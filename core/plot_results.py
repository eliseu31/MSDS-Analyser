import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os


class PlotClassificationResults:

    def __init__(self, file_name='classifier_results.csv'):
        # get the file path
        file_path = os.path.join(os.path.dirname(sys.path[0]), 'resources', 'results', file_name)
        # reads the file path
        self.df = pd.read_csv(file_path)
        # set y axis on the right
        # ax[1].yaxis.tick_right()
        # set the grid
        # ax[1].grid(axis='x')

    def plot_classification_metrics(self):
        # plot the results
        fig = plt.figure(1, figsize=(6, 5))
        ax = fig.subplots(2, 1, sharex='col', sharey='row')

        x_axis = np.array([1, 1.5, 2.5, 3, 4, 4.5])

        # plot the f1-score and roc graphs
        f1_bars = ax[0].bar(x_axis - 0.1, self.df['f1-score'],
                            width=0.2, color='yellow', label='F1-Score')
        roc_bars = ax[0].bar(x_axis + 0.1, self.df['roc'],
                             width=0.2, color='orange', label='AUC')
        # add the labels on the bars
        self.auto_label(f1_bars, ax[0], self.df['f1-score'])
        self.auto_label(roc_bars, ax[0], self.df['roc'])
        # sets legend and limits the y axe
        ax[0].legend()
        ax[0].set_ylim(0, 1)

        # plot the recall and precision graphs
        recall_bars = ax[1].bar(x_axis - 0.1, self.df['recall'],
                                width=0.2, color='blue', label='Recall')
        precision_bars = ax[1].bar(x_axis + 0.1, self.df['precision'],
                                   width=0.2, color='skyblue', label='Precision')
        # add the labels on the bars
        self.auto_label(recall_bars, ax[1], self.df['recall'])
        self.auto_label(precision_bars, ax[1], self.df['precision'])
        # sets legend and limits the y axe
        ax[1].legend()
        ax[1].set_ylim(0, 1)

        # sets the x axes
        ax[1].set_xticks(x_axis)
        ax[1].set_xticklabels(self.df['model'])
        # makes the extra x axe
        ax2 = ax[1].twiny()
        ax2.set_xticks([1.25, 2.75, 4.25])
        ax2.set_xticklabels(['Fire', 'Disposal', 'Storage'])
        ax2.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_label_position('bottom')
        ax2.spines['bottom'].set_position(('outward', 20))
        ax2.set_xlim(ax[1].get_xlim())
        ax2.set_xlabel('Classification Models & Procedures', weight='bold')

        # general graph info
        ax[0].set_title('Classifier Metrics Comparision', weight='bold')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def auto_label(bar_plot, ax, bar_labels):
        for idx, rect in enumerate(bar_plot):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 0.4 * height,
                    bar_labels[idx],
                    ha='center', va='bottom', rotation=90)


pcr = PlotClassificationResults()
pcr.plot_classification_metrics()
