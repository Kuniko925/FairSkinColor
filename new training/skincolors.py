from PIL import Image
from skimage import io, color
from derm_ita import derm_ita
import numpy as np
import math

class IndividualTypologyAngle:
    def __init__(self, RGB):
        #self.filepath = filepath
        #self.RGB = Image.open(filepath)
        self.RGB = Image.fromarray(RGB)
        #self.RGB = RGB
        self.CIELAB = np.array(color.rgb2lab(self.RGB))

    # Calculate mean ITA value
    def get_mean_ita(self):
        #print(f"Type of self.RGB: {type(self.RGB)}")  # Debugging statement
        ita = derm_ita.get_ita(self.RGB)
        return ita

    def get_nuance_ita(self):
        ita = []
        
        L = self.CIELAB[:, :, 0]
        L = np.where(L != 0, L, np.nan)
    
        B = self.CIELAB[:, :, 2]
        B = np.where(B != 0, B, np.nan)
    
        # If both nan, remove the pixel
        nan_indices = np.isnan(L) & np.isnan(B)
        L = L[~nan_indices]
        B = B[~nan_indices]
    
        has_nan_in_L = np.isnan(L).any()
    
        if has_nan_in_L:
            print("L contains NaN values.")
    
        has_nan_in_B = np.isnan(B).any()
        
        if has_nan_in_B:
            print("B contains NaN values.")
        
        # Calculate ITA
        for j in range(len(L)):
            ITA = math.atan2(L[j] - 50, B[j]) * (180 / np.pi)
            ita.append(ITA)
        return ita

    def map_skin_tone(self, ita): # Based on Fitzpatrick
        if ita > 55:
            return "1"
        elif ita > 41:
            return "2"
        elif ita > 28:
            return "3"
        elif ita > 10:
            return "4"
        elif ita > -30:
            return "5"
        else:
            return "6"
