import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import datetime
import open3d as o3d
import hashlib
import json
import cv2
import pandas as pd

def get_FFT_peak(data):
    """find the highest value in an FFT, return a list of (idx, mag, phase)"""
    mag = np.abs(data)
    # mag[:20] = 0            # ignore signal too close
    phases = np.angle(data)
    phases = (phases+2*np.pi) % (2*np.pi)
    peak_idx = np.argmax(mag)
    return (peak_idx, mag[peak_idx], phases[peak_idx])

def log(message):
    """Log a message to console with timestamp"""
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] {message}')

def prepare_display(data, data_gt=None, mesh=None, origin=[0, 0, 0], show_radar=False, voxel=False):
    """Use open3d to display point cloud or mesh models"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (data.shape[0], 1)))
    display = [pcd]
    if show_radar:
        radar = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=origin)
        display.append(radar)

    if data_gt is not None:
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(data_gt)
        pcd_gt.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 1, 0]), (data_gt.shape[0], 1)))
        display.append(pcd_gt)

    if mesh is not None:
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.pos)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.face.T)
        display.append(o3d_mesh)

    if voxel:
        pcd_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.01)
        display.append(pcd_voxel)
    return display

def plot_point_cloud_o3d(data, data_gt=None, mesh=None, voxel=False, origin=[0, 0, 0], show_radar=False, video=False, view='front'):
    """Plot a point cloud. The x, y, z axis will be rendered as red, green, and blue arrows respectively
    """
    display = prepare_display(data, data_gt, mesh, origin, show_radar, voxel=voxel)
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    for d in display:
        vis.add_geometry(d)
    if view == 'left':
        cfg = 'o3d-camera-left.json'
    else:
        cfg = 'o3d-camera.json'
    param = o3d.io.read_pinhole_camera_parameters(cfg)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    
    if video:
        o3d_video(vis, video)
    else:
        vis.run()
        # ts = datetime.datetime.now().strftime('%m%d%H%M%S')
        # image = vis.capture_screen_float_buffer(False)
        # plt.imsave(f"{ts}.png", np.asarray(image), dpi=1)
        vis.destroy_window()

def o3d_video(vis, videoname):
    """Plot 3D models and make a video"""
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    o3d_video.vw = cv2.VideoWriter(f'{videoname}.avi', fourcc, 30.0, (1920, 1080), True)
    o3d_video.cnt = 0
    def rotate_and_capture(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        image = vis.capture_screen_float_buffer(False)
        image = np.asarray(image)
        image = np.asarray(image*255, dtype=np.uint8)
        o3d_video.vw.write(image)
        o3d_video.cnt += 1
        if o3d_video.cnt == 108*4:
            o3d_video.vw.release()
            o3d_video.vis.register_animation_callback(None)
        return False
    o3d_video.vis = vis
    vis.register_animation_callback(rotate_and_capture)
    vis.run()
    vis.destroy_window()

def config_to_id(config, name=None):
    """Compute unique hash ID from a dict"""
    if name is not None:
        return name + '-' + hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()

def read_csv(datafile):
    header = ['Name', 'VIoU', 'NIoU', 'Precision', 'Sensitivity']
    try:
        data = pd.read_csv(datafile, header=None, names=header)
    except FileNotFoundError as e:
        print(f'Search result {datafile} not found, using default')
        return 'CA-6-80-64'
    data['FMI'] = np.sqrt(data['Sensitivity']*data['Precision'])
    data = data.sort_values('FMI', ascending=False)
    first = data.iloc[0]
    return first[0]

def save_one_fig(figname, pcd, mesh=None, view='front'):
    """Plot a point cloud and make a snapshot."""
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array([0, 0, 1]), (pcd.shape[0], 1)))
    if view == 'left':
        cfg = 'o3d-camera-left.json'
    else:
        cfg = 'o3d-camera.json'
    param = o3d.io.read_pinhole_camera_parameters(cfg)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=480, height=640)
    vis.add_geometry(pcd_o3d)
    if mesh is not None:
        vertice = mesh.pos.numpy()
        face = mesh.face.T.numpy()
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertice)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(face)
        vis.add_geometry(mesh_o3d)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.update_geometry(pcd_o3d)
    if mesh is not None:
        vis.update_geometry(mesh_o3d)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f'{figname}-{view}.png')
    vis.destroy_window()