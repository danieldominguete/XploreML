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

    def plot_count_cat_histogram(self, y_column):

        if self.data[y_column].unique().shape[0] < MAX_CATEGORIES_FOR_PLOT:
            # fig with single axes
            fig, ax = plt.subplots(figsize=(12,7))
            ax = sns.catplot(data=self.data, x=y_column, kind="count")
            plt.tight_layout()

            # registering results
            full_path=None
            if self.save_plots:
                full_path= self.folder_path + self.prefix + y_column + '_count_cat.png'
                plt.savefig(full_path)

            if self.view_plots:
                plt.show()

            plt.close()
            return full_path
        else:
            logging.info('Count categorical histogram to ' + y_column + ' exceed the maximum limit of ' + str(MAX_CATEGORIES_FOR_PLOT) + ' categories.')
            return False

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

        # Visualising
        if marginal_distribution:
            sns.jointplot(x=X_name, y=Y_name, data=self.data, kind="reg")
        else:
            sns.lmplot(x=X_name, y=Y_name, data=self.data)

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

        #t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
        if sort:
            self.data.sort_values(by=[y], inplace=True)

        b = plt.plot(self.data[pred].tolist(), label='prediction')
        a = plt.plot(self.data[y].tolist(), label='expected')
        plt.ylabel('output')
        plt.legend()
        plt.show()

        return True

    #TODO Plot time x pred x target for forecast (index colum)

    def plot_history_training(self, history=None, loss=True, accuracy=False):

        # loss value x epochs
        if loss:
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.show()

        if accuracy:
            plt.plot(history.history['sparse_categorical_accuracy'])
            plt.plot(history.history['val_sparse_categorical_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Valid'], loc='upper left')
            plt.show()

        return True