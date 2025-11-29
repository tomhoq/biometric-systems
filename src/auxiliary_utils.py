from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import describe, gaussian_kde
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, 
    auc, precision_score, recall_score, f1_score, det_curve,
    accuracy_score
)
#import seaborn as sns
from DET import DET


mated_colour = "green"
mated_label = "Mated scores"
nonmated_colour = "red"
nonmated_label = "Non-mated scores"

figure_size = (6,4)
alpha_shade = 0.25
alpha_fill = 1.0
linewidth = 2
legend_loc = "upper left"
legend_anchor = (1.0, 1.02)
legend_cols = 1
legend_fontsize = 12
label_fontsize = 16

threshold_colour = "black"
threshold_style = "--"
round_digits = 5
plt.rc("axes", axisbelow=True)

class Auxiliary:
    """
    A class to hold auxiliary functions.
    """
    """Draw a confusion matrix from mated and non-mated scores."""
    
    
    def __init__(self, true_male_scores, true_female_scores, labels_list, total_scores, threshold=0.5, mode="percent", out_path=None):
        """
        Initialize the Auxiliary class with mated scores.

        Args:
            true_male_scores (array-like): Scores for genuine (mated) comparisons.
        """
        self.true_male_scores = true_male_scores  # scores of male when male is true
        self.true_female_scores = true_female_scores # scores of female when female is true
        self.labels_list = labels_list # true labels
        self.total_scores = total_scores # all the obtained scores where 1 is man and 0 is woman  
        self.threshold = threshold  
        self.mode = mode
        self.out_path = out_path

    def print_classification_metrics(self):
        """
        Computes and prints Precision, Recall (Sensitivity), Specificity, and F1-Score.
        Assumes binary classification with labels 1 (Male) and 0 (Female).
        """
        y_true = np.array(self.labels_list)
        y_scores = np.array(self.total_scores)
        y_pred = (y_scores >= self.threshold).astype(int)

        self.better_confusion_matrix()
        auc = self.plot_roc_curve()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        eer = self.plot_det()
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)  # aka Sensitivity
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        f1 = f1_score(y_true, y_pred)

        # Print results
        print("Classification Metrics")
        print(f"Accuracy     : {accuracy:.3f}")
        print(f"Precision     : {precision:.3f}")
        print(f"Recall        : {recall:.3f} (Sensitivity)")
        print(f"Specificity   : {specificity:.3f}")
        print(f"F1 Score      : {f1:.3f}")
        print(f"AUC           : {auc:.3f}")
        print(f"EER:          : {eer:.3f}")

        if self.out_path is not None:
            with open(f'{self.out_path}/metrics.txt', 'w') as f:
                print("Classification Metrics", file= f)
                print(f"Accuracy     : {accuracy:.3f}", file=f)
                print(f"Precision     : {precision:.3f}", file= f)
                print(f"Recall        : {recall:.3f} (Sensitivity)", file= f)
                print(f"Specificity   : {specificity:.3f}", file= f)
                print(f"F1 Score      : {f1:.3f}", file= f)
                print(f"AUC      : {auc:.2f}", file= f)
                print(f"EER:          : {eer:.3f}", file= f)



    def plot_roc_curve(self):
        y_true = np.array(self.labels_list)
        y_scores = np.array(self.total_scores)

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plotting
        plt.figure(figsize=figure_size)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Male Rate')
        plt.ylabel('True Male Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        if self.out_path is not None:
            plt.savefig(f"{self.out_path}/roc.png")
        #plt.show()
        return roc_auc

    """ Model2 and Model3 are a 2d array like [np_male_list, np_female_list]"""
    def plot_det(self, model2 = None, model3 = None):
        det = DET(biometric_evaluation_type='algorithm', abbreviate_axes=True, plot_eer_line=True, plot_title="1 system example")
        det.x_limits = np.array([1e-4, .5])
        det.y_limits = np.array([1e-4, .5])
        det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
        det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
        det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
        det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
        det.create_figure()

        tp_man = self.total_scores[np.array(self.labels_list) == 1]
        fn_man = self.total_scores[np.array(self.labels_list) == 0]
        det.plot(tar=tp_man, non=fn_man, label="Model 1")

        def compute_eer(tp_man, fn_man, model):
            # Flatten lists if needed
            tar_scores = np.array(tp_man).flatten()
            non_scores = np.array(fn_man).flatten()

            # Generate labels and scores
            y_true = np.concatenate([np.ones_like(tar_scores), np.zeros_like(non_scores)])
            y_scores = np.concatenate([tar_scores, non_scores])

            # Compute false positive rate, false negative rate
            fpr, fnr, thresholds = det_curve(y_true, y_scores)

            # Compute EER as point where FPR ≈ FNR
            eer_index = np.nanargmin(np.absolute(fnr - fpr))
            eer = (fpr[eer_index] + fnr[eer_index]) / 2
            return eer

        eer = compute_eer(tp_man, fn_man, "Model 1")

        if model2 is not None:
            det.plot(tar=model2[0], non= model2[1], label="Model 2")
            compute_eer(model2[0], model2[1], "Model 2")

        if model3 is not None:
            det.plot(tar=model3[0], non= model3[1], label="Model 3")
            compute_eer(model3[0], model3[1], "Model 3")
        if self.out_path is not None:
            det.save(f"{self.out_path}/det", type="png")
        det.legend_on()
        return eer

    def better_confusion_matrix(self):
        # Assume you used the first model to compute predictions
        y_true = self.labels_list
        y_pred = []

        for probs in self.total_scores:
            # If Male (index 1) probability > 0.5, predict Male (1); else Female (0)
            predicted_label = 1 if probs > 0.5 else 0
            y_pred.append(predicted_label)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])  # [1, 0] for Male, Female

        if self.mode == "percent":
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0], normalize="all")  # [1, 0] for Male, Female

        # Display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
        disp.plot(cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if self.out_path is not None:
            plt.savefig(f"{self.out_path}/cm.png")
        #plt.show()


    """Plotting functions for KDE with a decision self.threshold."""
    def get_kde(self, distribution, linspace_items=101):
        dist_min, dist_max = distribution.min(), distribution.max()
        dist_kde = gaussian_kde(distribution)
        dist_pos = np.linspace(dist_min, dist_max, linspace_items)
        return dist_kde, dist_pos, dist_min, dist_max
    

    """Plotting functions for KDE with a decision self.threshold."""
    def kde_with_threshold(self, savename=None): 
        linspace_items = 101
        mated_kde, mated_pos, mated_min, mated_max = self.get_kde(self.true_male_scores, linspace_items)
        nonmated_kde, nonmated_pos, nonmated_min, nonmated_max = self.get_kde(self.true_female_scores, linspace_items)
        
        plt.figure(figsize=figure_size)
        plt.xlabel("Comparison Score", size=label_fontsize)
        plt.ylabel("Probability Density", size=label_fontsize)
        
        def compute_fills( mated_min, mated_max, nonmated_min, nonmated_max, linspace_items):
            if mated_min < self.threshold:
                mated_shade = np.linspace(self.threshold, mated_max, linspace_items)
                mated_fill = np.linspace(mated_min, self.threshold, linspace_items) 
            else:
                mated_shade = np.linspace(mated_min, mated_max, linspace_items)
                mated_fill = None
            if nonmated_max > self.threshold:
                nonmated_shade = np.linspace(nonmated_min, self.threshold, linspace_items)
                nonmated_fill = np.linspace(self.threshold, nonmated_max, linspace_items)
            else:
                nonmated_shade = np.linspace(nonmated_min, nonmated_max, linspace_items)
                nonmated_fill = None
            return mated_shade, mated_fill, nonmated_shade, nonmated_fill
        
        plt.plot(mated_pos, mated_kde(mated_pos), linewidth=linewidth, color=mated_colour, label=mated_label)
        plt.plot(nonmated_pos, nonmated_kde(nonmated_pos), linewidth=linewidth, color=nonmated_colour, label=nonmated_label)
        
        mated_shade, mated_fill, nonmated_shade, nonmated_fill = compute_fills( mated_min, mated_max, nonmated_min, nonmated_max, linspace_items)
        
        plt.fill_between(mated_shade, mated_kde(mated_shade), alpha=alpha_shade, color=mated_colour) 
        plt.fill_between(nonmated_shade, nonmated_kde(nonmated_shade), alpha=alpha_shade, color=nonmated_colour) 
        
        if mated_fill is not None:
            plt.fill_between(mated_fill, mated_kde(mated_fill), alpha=alpha_fill, color=mated_colour)
        if nonmated_fill is not None:
            plt.fill_between(nonmated_fill, nonmated_kde(nonmated_fill), alpha=alpha_fill, color=nonmated_colour)

        plt.axvline(self.threshold, linewidth=linewidth, linestyle=threshold_style, color=threshold_colour, label="Decision self.threshold")
        
        plt.legend(loc=0)
        red_patch = mpatches.Patch(color=nonmated_colour, alpha=alpha_fill, label='False Male')
        green_patch = mpatches.Patch(color=mated_colour, alpha=alpha_fill, label='False Female')
        shaded_red_patch = mpatches.Patch(color=nonmated_colour, alpha=alpha_shade, label='True Female')
        shaded_green_patch = mpatches.Patch(color=mated_colour, alpha=alpha_shade, label='True Male')
        current_handles, _ = plt.gca().get_legend_handles_labels()
        
        plt.grid(True)
        plt.legend(loc=legend_loc, bbox_to_anchor=legend_anchor, ncol=legend_cols, fontsize=legend_fontsize, handles=[green_patch, red_patch, shaded_green_patch, shaded_red_patch]+current_handles)
        plt.xlim(0, 1)
        plt.ylim(0, None)
        if self.out_path is not None:
            plt.savefig(f"{self.out_path}/kde.png")
        #plt.show()