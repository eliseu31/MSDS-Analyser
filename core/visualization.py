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
        plt.title('Feature Data', weight='bold')
        # plot the targets in the top
        self.plot_metrics(class_report, axes[0])
        plt.title('Target Data', weight='bold')
        # shows the plot
        plt.show()

    def plot_metrics(self, class_report, bag_words_ax):
        # plot the bar word frequency
        count_vectorizer = self.text_extraction_pipes['target']['vectorizer']
        # TEST ALSO Y_TEST
        word_labels, _ = self.plot_word_freq(count_vectorizer, self.y_train, ax=bag_words_ax)
        # get the metrics lists
        metrics_list = []
        # iterate over the word labels
        for label in word_labels:
            # stores the values in ordered list
            metrics_tuple = (class_report[label]['precision'],
                             class_report[label]['recall'],
                             class_report[label]['f1-score'])
            metrics_list.append(metrics_tuple)

        # format the word labels
        word_labels = list(word_labels)
        for i, label in enumerate(word_labels):
            # splits the label
            label_parts = label.split(' ')
            # slices all the words in groups of 2
            slices = [' '.join(label_parts[i:i+2]) for i in range(0, len(label_parts), 2)]
            # joins all the slices
            word_labels[i] = '\n'.join(slices)

        # plot the metrics graph
        twin_ax = bag_words_ax.twinx()
        precision, recall, f1_sco = zip(*metrics_list)
        # plots the metrics
        twin_ax.plot(word_labels, precision, label='precision', marker='.', color='red')
        twin_ax.plot(word_labels, recall, label='recall', marker='o', color='green')
        twin_ax.plot(word_labels, f1_sco, label='f1-score', marker='v', color='yellow')
        twin_ax.set_ylim(bottom=0, top=1)
        # shows the legend
        plt.legend()

    @staticmethod
    def plot_word_freq(vectorizer, string_df, ax=plt):
        # plot the vocabulary freq
        # plt.rcParams.update({'font.size': 15})
        bag_of_words = vectorizer.transform(string_df)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, i]) for i, word in enumerate(vectorizer.get_feature_names())]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        words, total = zip(*words_freq)
        # plots the histogram
        ax.bar(words[0:30], total[0:30])
        # alternative method to rotate the graphics
        # plt.xticks(rotation=90)
        # method to rotate the graphics
        ax.set_visible(True)
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        # return the word labels and values
        return words, total
