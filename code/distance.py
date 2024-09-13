import numpy as np
import statistics
from scipy.stats import wasserstein_distance

class DistanceMeasure:
    def __init__(self, XX, YY):
        self.XX = XX
        self.YY = YY

    def check_sign(self):
        gap = statistics.median(self.XX) - statistics.median(self.YY)
        
        if gap < 0:
            return 1
        else:
            return -1

    # Want to know positive/negative distibution shift
    def sign_wasserstein_distance(self):
        wd = wasserstein_distance(self.XX, self.YY)
        sign = self.check_sign()
        return wd * sign
    
    def kuiper_distance(self):

        nx = len(self.XX)
        ny = len(np.array(self.YY))
        n = nx + ny
    
        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        up = 0
        down = 0
        Res = 0
        E_CDF = 0
        F_CDF = 0
        height = 0
        power = 1
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            if XY_Sorted[ii+1] != XY_Sorted[ii]: height = F_CDF-E_CDF
            if height > up: up = height
            if height < down: down = height
    
        K_Dist = abs(down)**power + abs(up)**power
    
        return K_Dist

    def Anderson_Darling_Dist(self):
    
        nx = len(self.XX)
        ny = len(np.array(self.YY))
        n = nx + ny
    
        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        Res = 0
        E_CDF = 0
        F_CDF = 0
        G_CDF = 0
        height = 0
        SD = 0
        power = 1
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            G_CDF = G_CDF + 1/n
            SD = (n * G_CDF * (1-G_CDF))**0.5
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii+1] != XY_Sorted[ii]:
                if SD>0:
                    Res = Res + (height/SD)**power
    
        AD_Dist = Res
    
        return AD_Dist

    def CVM_Dist(self):
    
        nx = len(self.XX)
        ny = len(self.YY)
        n = nx + ny
    
        XY = np.concatenate([self.XX, self.YY])
        X2 = np.concatenate([np.repeat(1/nx, nx), np.repeat(0, ny)])
        Y2 = np.concatenate([np.repeat(0, nx), np.repeat(1/ny, ny)])
    
        S_Ind = np.argsort(XY)
        XY_Sorted = XY[S_Ind]
        X2_Sorted = X2[S_Ind]
        Y2_Sorted = Y2[S_Ind]
    
        Res = 0;
        E_CDF = 0;
        F_CDF = 0;
        power = 1;
    
        for ii in range(0, n-2):
            E_CDF = E_CDF + X2_Sorted[ii]
            F_CDF = F_CDF + Y2_Sorted[ii]
            height = abs(F_CDF - E_CDF)
            if XY_Sorted[ii+1] != XY_Sorted[ii]: Res = Res + height**power
    
        CVM_Dist = Res
    
        return CVM_Dist