import numpy as np
from numpy.polynomial.polynomial import Polynomial
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle

class PerformanceEvaluation:
    def __init__(self, tn, fp, fn, tp):
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.tp = tp
    def true_positive_rate(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) != 0 else 0
    
    def true_negative_rate(self):
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) != 0 else 0
    
    def false_positive_rate(self):
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) != 0 else 0
    
    def false_negative_rate(self):
        return self.fn / (self.fn + self.tp) if (self.fn + self.tp) != 0 else 0


class PerformanceMeasure:
    def __init__(self, df, col, batch_size):
        self.df = df
        self.col = col
        self.batch_size = batch_size
        
    def measure(self):
        df_sorted = self.df.sort_values(by=self.col)
        total_rows = len(df_sorted)
        
        acc = []
        tpr = []
        tnr = []
        fpr = []
        fnr = []
        d_mean = []
        
        batch_data= ""
        
        for start_idx in range(0, total_rows, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_data = df_sorted.iloc[start_idx:end_idx]
            y_true = batch_data["labels"]
            y_pred = batch_data["predictions"]
            
            conf_matrix = confusion_matrix(y_true, y_pred, normalize="true")
            num_classes = conf_matrix.shape[0]
            if num_classes == 2:
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, normalize="true").ravel()
            else:
                # 1x1
                tn = conf_matrix[0, 0]
                fp = 0
                fn = 0
                tp = 0

            pe = PerformanceEvaluation(tn, fp, fn, tp)
            d_mean.append(batch_data[self.col].mean())
            acc.append(accuracy_score(y_true, y_pred))
            tpr.append(pe.true_positive_rate())
            tnr.append(pe.true_negative_rate())
            fpr.append(pe.false_positive_rate())
            fnr.append(pe.false_negative_rate())
        
        res = {self.col: d_mean, "ACC": acc, "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr}
        df_res = pd.DataFrame(res)
        return df_res

metrics = ["ACC", "TPR", "TNR", "FPR", "FNR"]
colors = ["blue", "orange", "green", "red", "purple"]


class PerformanceOptimisation:
    def __init__(self, df, db_name, model_name):
        self.df = df
        self.db_name = db_name
        self.model_name = model_name
        self.df_performance = None
        self.optimal_model = None

    # Noiseを除去して平均化するためのバッチサイズの検証が必要
    def check_batch_size(self, ax, b_start = 10, b_end = 210, b_step = 10):
        
        df_batch_correlation = pd.DataFrame()
        
        for i in range(b_start, b_end, b_step):
            # i is batch size
            pm = PerformanceMeasure(self.df, "distance", i)
            df_conclusion = pm.measure()
            correlation_matrix = df_conclusion.corr()
        
            correlation_matrix["batch size"] = i
            df_batch_correlation = pd.concat([df_batch_correlation, correlation_matrix])
        
        df_batch_correlation = df_batch_correlation.loc["distance"]
        df_batch_correlation.reset_index(inplace=True)
    
        for m in metrics:
            ax.plot(df_batch_correlation["batch size"], df_batch_correlation[m], marker="o", label=m)
            
        ax.set_title(f"Batch Size Justification {self.db_name}-{self.model_name}")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Correlation")
        ax.legend()    
        return ax

    # 最適なバッチサイズでの相関を出力
    def check_optimal_batch_correlation(self, batch_size):
        pm = PerformanceMeasure(self.df, "distance", batch_size)
        df_conclusion = pm.measure()
        self.df_performance = df_conclusion # 最適なバッチサイズでの距離と性能のデータフレームを保存
        print(df_conclusion.corr())

    # 最適なバッチサイズでの距離と性能を表示
    def display_distance_performance(self, ax):
        if self.df_performance is not None:
            for i, m in enumerate(metrics):
                ax.scatter(self.df_performance["distance"], self.df_performance[m], color=colors[i], label=m)
                ax.axhline(y=self.df_performance[m].mean(), color=colors[i], linestyle="--", label=f"{m} MEAN")
            
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax.set_title(f"Performance vs Distance {self.db_name}-{self.model_name}")
            ax.set_xlabel("Distance")
            ax.set_ylabel("Performance")
            ax.legend()
            return ax
        else:
            print("execute check_optimal_batch_correlation")

    # 最適な多項数の確認
    def check_optimised_degree(self, metric, degrees, ax):

        print(f"{self.db_name} - {self.model_name}")
        
        x = self.df_performance["distance"]
        y = self.df_performance[metric]
        
        # x の範囲に対して y を予測するための新しい x
        x_new = np.linspace(x.min(), x.max(), 1000)
        
        # To see the best degree
        ax.scatter(x, y, label="Observed Data", color="gray", alpha=0.8)
        
        for deg in degrees:
            p = Polynomial.fit(x, y, deg)
        
            # To plot
            y_new = p(x_new)
            ax.plot(x_new, y_new, label=f"Degree {deg}")
        
            # To see errors
            y_pred = p(x)
            mse = mean_squared_error(y, y_pred)
            print(f"Degree {deg}: MSE = {mse}")

        ax.set_title(f"{self.db_name} - {self.model_name}")
        ax.set_xlabel("Distance")
        ax.set_ylabel(metric)
        ax.legend()
        return ax


    # 最適な多項の表示
    def diaplay_optimal_degree(self, metric, optimal_deg, ax):

        x = self.df_performance["distance"]
        y = self.df_performance[metric]
        
        x_new = np.linspace(x.min(), x.max(), 1000)
        self.optimal_model = Polynomial.fit(x, y, optimal_deg)
        y_optimal_pred = self.optimal_model(x_new)
        
        ax.scatter(x, y, label="Observed Data", color="gray", alpha=0.8)
        ax.plot(x_new, y_optimal_pred, color="blue", label=f"Optimal Degree {optimal_deg}")
        ax.axhline(y.mean(), color="red", linestyle="--", label="Zero Difference")
        ax.set_xlabel("Distance")
        ax.set_ylabel(metric)
        ax.legend()
    
        return ax

    # 最適な多項のモデルを返す
    def get_optimal_polynomial_model(self):
        if self.optimal_model is not None:
            return self.optimal_model
        else:
            print("execute diaplay_optimal_degree")

    # 最適な多項のモデルを保存
    def save_optimal_polynomial_model(self, polynomial_save_file):
        if self.optimal_model is not None:
            with open(polynomial_save_file, "wb") as f:
                pickle.dump(self.optimal_model, f)
                print(f"Modle saved")

    def get_prior_data(self, metric):
        x = self.df_performance["distance"]
        y = self.df_performance[metric]
        return x, y

class PerformanceEstimation:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def estimation(self):
        x = np.array(self.x).reshape(-1, 1)
        y = np.array(self.y)
        
        reg = linear_model.BayesianRidge()

        try:
            reg.fit(x, y)
        except Exception as e:
            raise ValueError(f"An error occurred while fitting the model: {e}")
        
        pred = reg.predict(x)
        
        # Compute performance metrics
        mse = mean_squared_error(y, pred)
        mae = mean_absolute_error(y, pred)
        rmse = mean_squared_error(y, pred, squared=False)
        
        return reg, (mse, mae, rmse)