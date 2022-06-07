import numpy as np
import scipy

N_METRICS = 4
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
            

    def run(self, th=0.1, print_result=False, norm=False):
        """ Return: VIoU, NIoU, precision and sensitivity
        """
        if self.nx == 0 or self.ny == 0:
            return [0, 0, 0, 0]
        precision = self.precision(th)
        sensitivity = self.sensitivity(th)
        IoU = self.IoU(th, norm=norm)
        VIoU = self.VIoU(th)
        if print_result:
            print(f'precision {precision*100:.2f}%, sensitivity {sensitivity*100:.2f}%, IoU {IoU*100:.2f}%, VIoU {VIoU*100:.2f}%.')
        return VIoU, IoU, precision, sensitivity

    def precision(self, th):
        """Precision - among all the point in X, how many of them are in Y. 
        """
        return np.sum(self.X_to_Y <= th)/self.nx
        
    def sensitivity(self, th):
        """sensitivity - among all the point in Y, how many of them are in X. 
        """
        return np.sum(self.Y_to_X <= th)/self.ny

    def IoU(self, th, norm=False):
        """Intersection over union.
        """
        if self.nx > self.ny and norm:
            union = self.ny*2
            res = []
            for _ in range(10):
                idx = np.random.choice(self.nx, size=self.ny, replace=False)
                X = self.X[idx]
                Y = self.Y
                dis = scipy.spatial.distance.cdist(X, Y)
                X_to_Y = np.min(dis, axis=1)
                Y_to_X = np.min(dis, axis=0)
                intersection1 = np.sum(X_to_Y <= th)
                intersection2 = np.sum(Y_to_X <= th)
                res.append((intersection1+intersection2)/union)
            return np.mean(res)
        else:
            union = self.nx + self.ny
            intersection1 = np.sum(self.X_to_Y <= th)
            intersection2 = np.sum(self.Y_to_X <= th)
        return (intersection1+intersection2)/union

    def VIoU(self, th):
        VX = np.unique((self.X/th).astype(int), axis=0)
        VY = np.unique((self.Y/th).astype(int), axis=0)
        I = np.sum((VX[:, None] == VY).all(-1).any(-1))
        U = (VX.shape[0]+VY.shape[0]) - I
        VIoU = I/U
        return VIoU
        

class VoxelEvaluator:
    def __init__(self, X, Y, voxel_size=0.1):
        assert len(X.shape) == len(Y.shape) == 2
        assert X.shape[1] == Y.shape[1] == 3
        if not (X.shape[0] == 0 or X.shape[0] == 0):
            self.VX = np.unique((X/voxel_size).astype(int), axis=0)
            self.VY = np.unique((Y/voxel_size).astype(int), axis=0)
            self.nx = self.VX.shape[0]
            self.ny = self.VY.shape[0]
            self.I = np.sum((self.VX[:, None] == self.VY).all(-1).any(-1))
        
    def run(self, print_result=False):
        """ Return: IoU, precision and sensitivity
        """
        if self.nx == 0 or self.ny == 0:
            return (0, 0, 0)
        precision = self.precision()
        sensitivity = self.sensitivity()
        IoU = self.IoU()
        if print_result:
            print(f'precision {precision*100:.2f}%, sensitivity {sensitivity*100:.2f}%, IoU {IoU*100:.2f}%.')
        return IoU, precision, sensitivity

    def precision(self):
        """Precision - among all the point in X, how many of them are in Y. 
        """
        return self.I/self.nx

    def sensitivity(self):
        """sensitivity - among all the point in Y, how many of them are in X. 
        """
        return self.I/self.ny

    def IoU(self):
        """Intersection over union.
        """
        return self.I/(self.nx+self.ny-self.I)



