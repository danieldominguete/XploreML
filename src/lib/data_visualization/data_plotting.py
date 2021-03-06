'''
===========================================================================================
Data Plotting Package
===========================================================================================
Script Reviewed by COGNAS
===========================================================================================
'''
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

sns.set(color_codes=True)
sns.set(style="darkgrid")

MAX_CATEGORIES_FOR_PLOT = 500

class DataPlotting:

    def __init__(self, dataframe=None, view_plots=False, save_plots=False, folder_path=None, prefix=None):
        '''Constructor for this class'''
        self.data = dataframe
        self.view_plots=view_plots
        self.save_plots=save_plots
        self.folder_path=folder_path
        self.prefix=prefix

    def plot_line_steps(self, y_column):

        # fig with single axes
        fig, ax = plt.subplots(figsize=(12,7))
        ax = sns.lineplot(x=self.data.index, y=self.data[y_column])
        ax.set_xlabel('steps')
        ax.set_ylabel('values')
        ax.set_title(y_column)
        ax.legend()
        plt.tight_layout()

        # registering results
        full_path=None
        if self.save_plots:
            full_path= self.folder_path + self.prefix + y_column + '_line_steps.png'
            plt.savefig(full_path)

        if self.view_plots:
            plt.show()

        plt.close()
        return full_path

    def plot_multiple_lines(self, x_column:str, y_columns:list, title="", xlabel="", ylabel=""):

        full_path = None

        if self.save_plots:
            fig, ax = plt.subplots(figsize=(12,7))
            for i in range(len(y_columns)):
                ax = sns.lineplot(x=self.data[x_column], y=self.data[y_columns[i]])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(y_columns)
            plt.tight_layout()

            full_path = self.folder_path + self.prefix + title + '.png'
            plt.savefig(full_path)

            plt.close()

        if self.view_plots:
            fig, ax = plt.subplots(figsize=(12, 7))
            for i in range(len(y_columns)):
                ax = sns.lineplot(x=self.data[x_column], y=self.data[y_columns[i]])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(y_columns)
            plt.tight_layout()
            plt.show()

        return full_path

    def plot_count_cat_histogram(self, y_column):

        full_path = None

        if self.save_plots:
            if self.data[y_column].unique().shape[0] < MAX_CATEGORIES_FOR_PLOT:
                # fig with single axes
                fig, ax = plt.subplots(figsize=(12,7))
                ax = sns.catplot(data=self.data, x=y_column, kind="count")
                plt.tight_layout()

                full_path= self.folder_path + self.prefix + y_column + '_count_cat.png'
                plt.savefig(full_path)
                plt.close()
            else:
                logging.info('Count categorical histogram to ' + y_column + ' exceed the maximum limit of ' + str(MAX_CATEGORIES_FOR_PLOT) + ' categories.')

        if self.view_plots:
            if self.data[y_column].unique().shape[0] < MAX_CATEGORIES_FOR_PLOT:
                # fig with single axes
                fig, ax = plt.subplots(figsize=(12, 7))
                ax = sns.catplot(data=self.data, x=y_column, kind="count")
                plt.tight_layout()
                plt.show()

        return full_path

    def plot_numerical_histogram(self, y_column):

        # fig with single axes
        fig, ax = plt.subplots(figsize=(12,7))
        ax = sns.distplot(self.data[y_column])
        plt.tight_layout()

        # registering results
        full_path=None
        if self.save_plots:
            full_path= self.folder_path + self.prefix + y_column + '_hist.png'
            plt.savefig(full_path)

        if self.view_plots:
            plt.show()

        plt.close()
        return full_path


    def plot_scatter_2d(self, X_name=None, Y_name=None, title=None, marginal_distribution=False):

        if self.save_plots:
            # Visualising
            if marginal_distribution:
                sns.jointplot(x=X_name, y=Y_name, data=self.data, kind="reg")
            else:
                sns.lmplot(x=X_name, y=Y_name, data=self.data, fit_reg=False)

            plt.title(title)

            filename = self.folder_path + self.prefix + "scatter.png"
            plt.savefig(filename)
            plt.close()

        if self.view_plots:
            # Visualising
            if marginal_distribution:
                sns.jointplot(x=X_name, y=Y_name, data=self.data, kind="reg")
            else:
                sns.lmplot(x=X_name, y=Y_name, data=self.data, fit_reg=False)

            plt.title(title)

            plt.show()

        return True

    def plot_variables_hist(self):

        self.data.hist()
        plt.show()

        return True

    def plot_variables_density(self):

        self.data.plot(kind='density', subplots=True, sharex=False)
        plt.show()

        return True

    def plot_variables_boxplot(self):

        self.data.plot(kind='box', subplots=True, sharex=False, sharey=False)
        plt.show()

        return True

    def plot_correlation_matrix(self):

        correlations=self.data.corr()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax=ax.matshow(correlations,vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks=np.arange(0,9,1)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        plt.show()

        return True

    def plot_scatter_matrix(self, categ=None):

        if categ:
            sns.pairplot(data=self.data, hue=categ)
        else:
            sns.pairplot(data=self.data)
        plt.show()

        return True

    def plot_image(self, img):

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)

        ax.imshow(img, cmap='gray')
        width, height = img.shape
        thresh = img.max()/2.5
        for x in range(width):
            for y in range(height):
                ax.annotate(str(round(img[x][y],2)), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if img[x][y]<thresh else 'black')
        plt.show()

        return True

    def plot_confusion_matrix(self, cm, names, title='Confusion matrix', cmap=plt.cm.Blues):

        if self.save_plots:
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(names))
            plt.xticks(tick_marks, names, rotation=45)
            plt.yticks(tick_marks, names)
            plt.tight_layout()
            plt.ylabel('Label Observado')
            plt.xlabel('Label Previsto')

            filename = self.folder_path + self.prefix + "confusion.png"
            plt.savefig(filename)
            plt.close()

        if self.view_plots:

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(names))
            plt.xticks(tick_marks, names, rotation=45)
            plt.yticks(tick_marks, names)
            plt.tight_layout()
            plt.ylabel('Label Observado')
            plt.xlabel('Label Previsto')
            plt.show()

        return True

    def plot_roc(self, pred, y):

        if self.save_plots:
            fpr, tpr, _ = roc_curve(y, pred)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label='Curva ROC (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falso Positivo')
            plt.ylabel('Taxa de Verdadeiro Positivo')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")

            filename = self.folder_path + self.prefix + "roc.png"
            plt.savefig(filename)
            plt.close()

        if self.view_plots:
            fpr, tpr, _ = roc_curve(y, pred)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label='Curva ROC (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falso Positivo')
            plt.ylabel('Taxa de Verdadeiro Positivo')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            plt.show()

        return True

    def plot_chart_sort_regression(self, pred, y, sort=True):

        if self.save_plots:

            # t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
            if sort:
                self.data.sort_values(by=[y], inplace=True)

            b = plt.plot(self.data[pred].tolist(), label='prediction')
            a = plt.plot(self.data[y].tolist(), label='expected')
            plt.ylabel('output')
            plt.legend()

            filename = self.folder_path + self.prefix + "sort_reg.png"
            plt.savefig(filename)
            plt.close()

        if self.view_plots:

            # t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
            if sort:
                self.data.sort_values(by=[y], inplace=True)

            b = plt.plot(self.data[pred].tolist(), label='prediction')
            a = plt.plot(self.data[y].tolist(), label='expected')
            plt.ylabel('output')
            plt.legend()
            plt.show()

        return True

    def plot_history_training(self, history=None):

        if self.save_plots:

            for metric in history['metrics']:

                fig, ax = plt.subplots(figsize=(12, 7))
                ax = plt.plot(history['metrics'].get(metric))
                plt.title(metric)
                plt.ylabel('value')
                plt.xlabel('epoch')

                filename = self.folder_path + self.prefix + metric + ".png"
                plt.savefig(filename)
                plt.close()

        if self.view_plots:

            for metric in history['metrics']:
                fig, ax = plt.subplots(figsize=(12, 7))
                ax = plt.plot(history['metrics'].get(metric))
                plt.title(metric)
                plt.ylabel('value')
                plt.xlabel('epoch')

                filename = self.folder_path + self.prefix + metric + ".png"
                plt.savefig(filename)
                plt.show()

        return True

    def plot_tokens_freq(self, frequency_dist, var:str=None) -> bool:

        filename = None
        if self.save_plots:
            frequency_list = frequency_dist.most_common()
            # file histogram
            filename = self.folder_path + self.prefix + var + "_tks_hist.csv"
            with open(filename, 'w') as f:
                f.write('token; samples\n')
                for token, frequency in frequency_list:
                    f.write("%s;%s\n" % (token, frequency))

        if self.view_plots:
            frequency_dist.plot(50, cumulative=False)

        return filename

    def plot_tokens_cloud(self, frequency_dist) -> bool:
        from wordcloud import WordCloud
        wcloud = WordCloud().generate_from_frequencies(frequencies=frequency_dist)
        filename = None

        if self.save_plots:
            plt.figure()
            plt.imshow(wcloud, interpolation='bilinear')
            plt.axis("off")

            filename = self.folder_path + self.prefix + "tk_cloud.png"
            plt.savefig(filename)
            plt.close()

        if self.view_plots:
            plt.figure()
            plt.imshow(wcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()

        return filename