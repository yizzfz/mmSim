import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import datetime
import open3d as o3d
import hashlib
import json
import cv2

# find the highest value in an FFT, return (idx, mag, phase)
def get_FFT_peak(data):
    mag = np.abs(data)
    # mag[:20] = 0            # ignore signal too close
    phases = np.angle(data)
    phases = (phases+2*np.pi) % (2*np.pi)
    peak_idx = np.argmax(mag)
    return (peak_idx, mag[peak_idx], phases[peak_idx])

def log(message):
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] {message}')

def prepare_display(data, data_gt=None, mesh=None, origin=[0, 0, 0], show_radar=False):
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
    return display

def plot_point_cloud_o3d(data, data_gt=None, mesh=None, origin=[0, 0, 0], show_radar=False, video=False):
    """The x, y, z axis will be rendered as red, green, and blue arrows respectively
    """
    display = prepare_display(data, data_gt, mesh, origin, show_radar)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for d in display:
        vis.add_geometry(d)
    param = o3d.io.read_pinhole_camera_parameters('o3d-camera.json')
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    
    if video:
        o3d_video(vis, video)
    else:
        vis.run()
        ts = datetime.datetime.now().strftime('%m%d%H%M%S')
        image = vis.capture_screen_float_buffer(False)
        plt.imsave(f"{ts}.png", np.asarray(image), dpi=1)
        vis.destroy_window()

def o3d_video(vis, video):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    o3d_video.vw = cv2.VideoWriter(f'tmp/simulation-pointcloud-{video}.avi', fourcc, 30.0, (1920, 1080), True)
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
    if name is not None:
        return name + '-' + hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()