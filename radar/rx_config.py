import numpy as np
import sys

ti_layout = ['1443', '1642', '6843isk', '6843oks', 'isk', 'ods']
class RxConfig:
    # data coming from right hand side will have a increasing phase
    # data coming from top will have a decreasing phase
    def __init__(self, layout):
        if layout in ti_layout:
            self.init_ti_layout(layout)
        else:
            try:
                p = layout.split('x')
                na = int(p[0])
                ne = int(p[1])
            except Exception as e:
                print(e)
                sys.abort(1)
            self.init_square_array(na, ne)

    def init_square_array(self, na, ne):
        xs = np.tile(np.arange(0, -na, -1), ne)
        zs = np.repeat(np.arange(ne), na) 
        ys = np.zeros(na*ne)
        rx = np.array((xs, ys, zs)).T
        self.azimuth_rx = np.arange(na, dtype=int)
        self.elevation_rx = self.azimuth_rx + na
        self.rx = rx
        self.n_rx = na*ne
        self.phase_offset = 0
        self.row = ne
        self.col = na

    def init_ti_layout(self, name):
        if name in ['1443', '6843isk', 'isk']:
            xs1 = np.arange(0, -8, -1)
            xs2 = np.arange(-2, -6, -1)
            ys1 = np.zeros(8)
            ys2 = np.zeros(4)
            zs1 = np.zeros(8)
            zs2 = np.zeros(4) + 1

            azimuth = np.array((xs1, ys1, zs1))
            elevation = np.array((xs2, ys2, zs2))
            rx = np.concatenate((azimuth, elevation), axis=-1).T
            
            self.azimuth_rx = np.arange(0, 8, dtype=int)
            self.elevation_rx = np.arange(8, 12, dtype=int)
            self.phase_offset = 2
            self.row = 8
            self.col = 2
        else:
            raise NotImplementedError

        self.rx = rx
        self.n_rx = rx.shape[0]

        