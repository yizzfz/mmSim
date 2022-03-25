import numpy as np
import scipy

class Evaluator:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.nx = X.shape[0]
        self.ny = Y.shape[0]
        assert len(X.shape) == len(Y.shape) == 2
        assert X.shape[1] == Y.shape[1] == 3
        if not (self.nx == 0 or self.ny == 0):
            self.dis = scipy.spatial.distance.cdist(X, Y)
            self.X_to_Y = np.min(self.dis, axis=1) 
            self.Y_to_X = np.min(self.dis, axis=0) 

    def run(self, th=0.1, print_result=False):
        """ Return: IoU, precision and sensitivity
        """
        if self.nx == 0 or self.ny == 0:
            return [0, 0, 0]
        precision = self.precision(th)
        sensitivity = self.sensitivity(th)
        IoU = self.IoU(th)
        if print_result:
            print(f'precision {precision*100:.2f}%, sensitivity {sensitivity*100:.2f}%, IoU {IoU*100:.2f}%.')
        return IoU, precision, sensitivity

    def precision(self, th):
        """Precision - among all the point in X, how many of them are in Y. 
        """
        return np.sum(self.X_to_Y <= th)/self.nx
        
    def sensitivity(self, th):
        """sensitivity - among all the point in Y, how many of them are in X. 
        """
        return np.sum(self.Y_to_X <= th)/self.ny

    def IoU(self, th):
        """Intersection over union.
        """
        union = self.nx + self.ny
        intersection1 = np.sum(self.X_to_Y <= th)
        intersection2 = np.sum(self.Y_to_X <= th)
        return (intersection1+intersection2)/union
        
    



