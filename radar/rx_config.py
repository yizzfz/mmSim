import numpy as np
import sys

# supported TI receiver layouts
ti_layout = ['1443', '1642', '6843isk', '6843oks', 'isk', 'ods']
class RxConfig:
    """
    Defines the layout of the radar receiver array.

    Data coming from right hand side will have a decreasing index but a increasing (positive) phase.

    Data coming from top will have a increasing index but a decreasing (negative) phase.

    Some code in pointcloud_processor relies on this behaviour.
    """
    def __init__(self, layout):
        self.shape = None
        self.name = layout
        if layout in ti_layout:         # TI radar layout
            self.init_ti_layout(layout)
        else:                           # square (AxB) layout
            try:
                p = layout.split('x')
                na = int(p[0])
                ne = int(p[1])
            except Exception as e:
                print(e)
                sys.abort(1)
            self.init_square_array(na, ne)

    def init_square_array(self, na, ne):
        """Square reciever array of size `na` (azimuth) by `ne` (elevation)."""
        if na <= 1 or ne <= 1:
            raise ValueError(f'Sqaure array dimension {ne}x{na} is not 2D')
        xs = np.tile(np.arange(0, -na, -1), ne)
        zs = np.repeat(np.arange(ne), na) 
        ys = np.zeros(na*ne)
        rx = np.array((xs, ys, zs)).T
        # azimuth rx is the first row
        self.azimuth_rx = np.arange(na, dtype=int)
        # elevation rx is the second row (old TI sdk) or the first column
        self.elevation_rx = [self.azimuth_rx + na, np.arange(ne) * na]
        self.rx = rx
        self.n_rx = na*ne
        self.phase_offset = 0
        self.row = ne
        self.col = na
        self.shape = (self.row, self.col)

    def init_ti_layout(self, name):
        if name in ['1443', '6843isk', 'isk']:
            """
                  11 10 09 08
            07 06 05 04 03 02 01 00
            """
            xs1 = np.arange(0, -8, -1)
            xs2 = np.arange(-2, -6, -1)
            ys1 = np.zeros(8)
            ys2 = np.zeros(4)
            zs1 = np.zeros(8)
            zs2 = np.zeros(4) + 1

            azimuth = np.array((xs1, ys1, zs1))
            elevation = np.array((xs2, ys2, zs2))
            self.rx = np.concatenate((azimuth, elevation), axis=-1).T
            # azimuth rx is the first row
            self.azimuth_rx = np.arange(0, 8, dtype=int)
            # elevation rx is the second row
            self.elevation_rx = [np.arange(8, 12, dtype=int)]
            self.phase_offset = 2
            self.row = 8
            self.col = 2
        elif name in ['6843ods', 'ods']:
            """
                  11 10
                  09 08
            07 06 05 04
            03 02 01 00
            """
            cols = [4, 4, 2, 2]
            rx = []
            for row in range(4):
                col = cols[row]
                xs = np.arange(0, -col, -1)
                ys = np.zeros(col)
                zs = np.zeros(col) + row
                rx.append(np.asarray((xs, ys, zs)).T)
            self.rx = np.concatenate(rx)
            
            self.azimuth_rx = np.arange(0, 4, dtype=int)
            self.elevation_rx = [self.azimuth_rx + 4, np.asarray([0, 4, 8, 10])]
            self.row = 4
            self.col = 4
            self.phase_offset = 0
            self.shape = (4, 4)
        else:
            raise NotImplementedError
        self.n_rx = self.rx.shape[0]

    def prepare_2d_fft_data(self, data):
        """Find which receivers to use for a 2D angle-FFT"""
        if self.shape is None:
            raise RuntimeError('Calling 2D angle FFT, but no array shape speicified in RxConfig')
        if self.name in ['6843ods', 'ods']:
            data = np.insert(data, 10, 0)
            data = np.insert(data, 11, 0)
            data = np.insert(data, 14, 0)
            data = np.insert(data, 15, 0)
        data = data.reshape(self.shape)
        data = data[::-1]                             # flip so that object from top has a positive phase
        return data

if __name__ == '__main__':
    rxcfg = RxConfig('ods')