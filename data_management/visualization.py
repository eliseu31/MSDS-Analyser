from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class PlotGraphics:

    def __init__(self, data, pipelines):
        # text extraction pipelines
        self.text_extraction_pipes = pipelines
        # get the data
        self.x_train, self.x_test, self.y_train, self.y_test = data

    def plot_bag_words(self, class_report):
        fig = plt.figure(1)
        # create the subplot
        axes = fig.subplots(nrows=2, ncols=1)
        # plot the features in the button
        count_vectorizer = self.text_extraction_pipes['feature']['vectorizer']
        self.plot_word_freq(count_vectorizer, self.x_train, axes[1])
        plt.title('Target Data')
        # plot the targets in the top
        self.plot_metrics(class_report, axes[0])
        plt.title('Feature Data')
        # shows the plot
        plt.show()

    def plot_metrics(self, class_report, ax1):
        # plot the bar word frequency
        count_vectorizer = self.text_extraction_pipes['target']['vectorizer']
        # TEST ALSO Y_TEST
        word_labels, _ = self.plot_word_freq(count_vectorizer, self.y_train, ax=ax1)
        # get the metrics lists
        metrics_list = []
        # iterate over the word labels
        for label in word_labels:
            # stores the values in ordered list
            metrics_tuple = (class_report[label]['precision'],
                             class_report[label]['recall'],
                             class_report[label]['f1-score'])
            metrics_list.append(metrics_tuple)

        # plot the metrics graph
        ax2 = ax1.twinx()
        precision, recall, f1_sco = zip(*metrics_list)
        ax2.plot(word_labels, precision, label='precision', marker='.', color='red')
        ax2.plot(word_labels, recall, label='recall', marker='o', color='green')
        ax2.plot(word_labels, f1_sco, label='f1-score', marker='v', color='yellow')
        ax2.set_ylim(bottom=0, top=1)
        # shows the legend
        plt.legend()

    @staticmethod
    def plot_word_freq(vectorizer, string_df, ax=plt):
        # plot the vocabulary freq
        # plt.rcParams.update({'font.size': 17})
        bag_of_words = vectorizer.transform(string_df)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, i]) for word, i in vectorizer.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        words, total = zip(*words_freq)
        ax.bar(words[0:50], total[0:50])
        # plt.xticks(rotation=90)
        ax.set_visible(True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        # return the word labels and values
        return words, total

    @staticmethod
    def plot_pca(x_data, y_data, class_labels):
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(x_data.todense())
        print(pca_data)
        print(pca_data.shape)
        print(y_data.shape)
        print(y_data.todense())

        # plot the pca
        # for class_label in class_labels:
        #     plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y_data)
        #     plt.show()
