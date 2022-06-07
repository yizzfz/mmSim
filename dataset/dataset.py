import torch_geometric
from torch_geometric.transforms import BaseTransform, Compose, SamplePoints, FixedPoints
from enum import Enum
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

computer_loc = {
    'K-X360': 'e:/study/datasets/mmSim/',
    'K-Shuangzi': 'f:/datasets/mmSim/',
    'IT077979': 'd:/datasets/mmSim/',
    'IT090254': 'c:/datasets/mmSim/',
}

class RadarTransform(BaseTransform):
    r"""swap the y and z axis of the point cloud to align with radar coordinate system,
    then place the point cloud 2m away from the radar.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.pos[:, [1,2]] = data.pos[:, [2,1]]                             # swap y and z
        data.pos[:, [0,1]] = -data.pos[:, [0,1]]                            # rotate to face the camera
        data.pos[:, 1] = data.pos[:, 1] - min(data.pos[:, 1])
        return data

class Translate(BaseTransform):
    def __init__(self, dis):
        super().__init__()
        self.dis = dis

    def __call__(self, data):
        data.pos[:, 1] += self.dis       # move away
        return data

class Scale(BaseTransform):
    r"""scale the point cloud by s.
    """
    def __init__(self, s):
        self.s = s
        super().__init__()

    def __call__(self, data):
        data.pos = data.pos * self.s
        return data

class DF(BaseTransform):
    r"""To serielize DynamicFaust somehow.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data.pos = data.pos[-1]
        data.num_nodes = None           # overwrite this value as it causes bugs in FixedPoints
        return data

datasets = {
    'ShapeNet': {},
    'S3DIS': {},
    'ModelNet':{
        'pretransform': Compose([
            SamplePoints(4096),
            Scale(0.01)
        ])
    }, 
    'FAUST':{
        'pretransform': SamplePoints(4096),
    },
    'DynamicFAUST':{
        'pretransform': Compose([
            DF(), 
            SamplePoints(4096),
        ])
    }
}

mesh_datasets = {
    'ModelNet':{
        'pretransform': Scale(0.01)
    }, 
    'FAUST': {},
    'DynamicFAUST': {
        'pretransform': DF()
    },
}
# https://github.com/Aaron-Zhao123/Sylveon/blob/master/sylveon/datasets/factory.py



class Dataset:
    """supported dataset: 'ShapeNet', 'ModelNet', 'S3DIS', 'FAUST', 'DynamicFAUST'
    """
    datasets = datasets
    def __init__(self, name, n_samples=None, location=None, distance=2, train=False):
        if name not in self.datasets:
            raise ValueError('Incorrect dataset name')
        
        cls = getattr(torch_geometric.datasets, name)
        pre_transform_func = self.datasets[name].get('pretransform')
        if pre_transform_func is not None:
            pre_transform_func = Compose([pre_transform_func, RadarTransform()])
        else:
            pre_transform_func = RadarTransform()
        if location is None:
            computer_name = os.environ['COMPUTERNAME']
            location = computer_loc.get(computer_name)      # random sample a number of points on every access
            if location is None:
                raise ValueError
        transform_func = []
        if n_samples is not None:
            transform_func.append(FixedPoints(n_samples, replace=False))
        if distance != 0:
            transform_func.append(Translate(distance))
        transform_func = Compose(transform_func) if transform_func != [] else None
        
        dataset_args = {
            'pre_transform': pre_transform_func,
            'transform': transform_func,
        }
        if name in ['ShapeNet']:
            dataset_args['split'] = 'trainval' if train else 'test'
        elif name in ['ModelNet', 'S3DIS', 'FAUST']:
            dataset_args['train'] = train

        self.load(cls, location, name, **dataset_args)

        if name == 'DynamicFAUST':
            if train:
                idx = [i for i in range(len(self.dataset)) if i%5!=0]
            else:
                idx = range(0, len(self.dataset), 5)
            self.dataset = self.dataset[idx]

    def load(self, cls, location, name, **args):
        self.dataset = cls(location + '/pointcloud/'+ name, **args)

    def __iter__(self):
        yield from self.dataset

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)

class DatasetMesh(Dataset):
    """supported dataset: 'ModelNet', 'FAUST'
    """
    datasets = mesh_datasets
    def __init__(self, name, location=None, distance=2, train=False):
        super().__init__(name, location=location, distance=distance, train=train)

    def load(self, cls, location, name, **args):
        self.dataset = cls(location + '/mesh/'+ name, **args)


def view_room(i, ds):
    d1 = ds[i].pos.numpy()
    color = ds[i].x.numpy()[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d1)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])

def view_pc(i, ds):
    print(i)
    d1 = ds[i].pos.numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d1)
    o3d.visualization.draw_geometries([pcd])

def view_mesh(i, ds):
    print(i)
    d2 = ds[i]
    vertice = d2.pos.numpy()
    face = d2.face.T.numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertice)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    o3d.visualization.draw_geometries([mesh])

def view_both(i, ds1, ds2):
    print(i)
    d1 = ds1[i].pos.numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(d1)

    d2 = ds2[i]
    vertice = d2.pos.numpy()
    face = d2.face.T.numpy()
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertice)
    mesh.triangles = o3d.utility.Vector3iVector(face)
    o3d.visualization.draw_geometries([pcd, mesh])

def save_one_fig(obj, name):
    param = o3d.io.read_pinhole_camera_parameters('../o3d-camera.json')
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=640)
    vis.add_geometry(obj)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(name)
    vis.destroy_window()

def save_figs(name):
    ds1 = Dataset(name, n_samples=1024, train=True)
    ds2 = DatasetMesh(name, train=True)
    for i in range(len(ds1)):
        d1 = ds1[i].pos.numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(d1)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 1, 0]), (d1.shape[0], 1)))
        figname = f"img/{name}-pc-{i}.png"
        save_one_fig(pcd, figname)

        d2 = ds2[i]
        vertice = d2.pos.numpy()
        face = d2.face.T.numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertice)
        mesh.triangles = o3d.utility.Vector3iVector(face)
        figname = f"img/{name}-mesh-{i}.png"
        save_one_fig(mesh, figname)

if __name__ == "__main__":
    # save_figs('FAUST')
    # save_figs('DynamicFAUST')
    # ds1 = Dataset('FAUST', 512, train=True, distance=3)
    # ds2 = DatasetMesh('FAUST', train=True, distance=1)
    # for i in range(len(ds2)):
    #     view_both(i, ds1, ds2)
    # ds1 = Dataset('ModelNet')
    ds1 = Dataset('FAUST', train=True)
    # ds2 = Dataset('DynamicFAUST', train=True)
    import pdb; pdb.set_trace()
    # view_pc(0, ds1)

    # ds = Dataset('S3DIS')
    # import pdb; pdb.set_trace()