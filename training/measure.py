import pandas as pd
import cv2
from skincolors import IndividualTypologyAngle
from distance import DistanceMeasure

class MeasureSkin:
    def __init__(self):
        self.baseline_filepath = None
        self.baseline_mean_ita = None
        self.baseline_nuance_ita = None
    def select_baseline_skin(self, df, col_filepath="masked filepath", random_state=12):
        
        baseline = df.sample(n=1, random_state=random_state)
        self.baseline_filepath = baseline[col_filepath].values[0]
        print(f"BaseLine File Name: {self.baseline_filepath}")

        img = cv2.imread(self.baseline_filepath)
        baseline_skin_pixels_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ita = IndividualTypologyAngle(baseline_skin_pixels_image)
        self.baseline_mean_ita = ita.get_mean_ita()
        print(f"Conventional ITA values: {self.baseline_mean_ita}")
        self.baseline_nuance_ita = ita.get_nuance_ita()

    def measure(self, df, col_filepath="masked filepath"):
        means = []
        distances = []

        for _, row in df.iterrows():

            img = cv2.imread(row[col_filepath])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if img is None:
                # print(_)
                means.append(np.nan)
                distances.append(np.nan)
            else:
                ita = IndividualTypologyAngle(img)
                means.append(ita.get_mean_ita())
                nuance_ita = ita.get_nuance_ita()

                if len(nuance_ita) == 0:
                    print(row[col_filepath])
                    
                dm = DistanceMeasure(self.baseline_nuance_ita, nuance_ita)
                distance = dm.sign_wasserstein_distance()
                distances.append(distance)

        df["means"] = means
        df["distance"] = distances
        print(f"Completed: {len(distances)}")
        return df