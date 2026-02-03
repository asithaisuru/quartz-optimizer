import os
import struct
import numpy as np
import collections

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

def read_uint64(fid): 
    data = fid.read(8)
    if len(data) < 8: return None
    return struct.unpack("Q", data)[0]

def read_uint32(fid):
    data = fid.read(4)
    if len(data) < 4: return None
    return struct.unpack("I", data)[0]

def read_int32(fid):
    data = fid.read(4)
    if len(data) < 4: return None
    return struct.unpack("i", data)[0]

def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_uint64(fid)
        if num_cameras is None: return {}
        for _ in range(num_cameras):
            camera_id = read_uint32(fid)
            model_id = read_int32(fid)
            width = read_uint64(fid)
            height = read_uint64(fid)
            
            # Param lengths for common COLMAP models
            # 0: SIMPLE_PINHOLE (3), 1: PINHOLE (4), 2: SIMPLE_RADIAL (4), 3: RADIAL (5)
            # Default to 4 if unknown
            num_params = 4
            if model_id == 0: num_params = 3
            elif model_id == 2: num_params = 4
            
            params = np.fromfile(fid, dtype=np.float64, count=num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_id, width=width, height=height, params=params)
    return cameras

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_uint64(fid)
        if num_reg_images is None: return {}
        
        for _ in range(num_reg_images):
            binary_image_properties = fid.read(64)
            if len(binary_image_properties) < 64: break
            
            image_id = struct.unpack("i", binary_image_properties[0:4])[0]
            qvec = np.frombuffer(binary_image_properties[4:36], dtype=np.float64)
            tvec = np.frombuffer(binary_image_properties[36:60], dtype=np.float64)
            camera_id = struct.unpack("i", binary_image_properties[60:64])[0]
            
            name = ""
            while True:
                char = fid.read(1)
                if char == b"\x00" or not char: break
                name += char.decode("utf-8")
                
            num_points2D = read_uint64(fid)
            if num_points2D is None: break
            
            data = np.fromfile(fid, dtype=np.float64, count=num_points2D * 2).reshape((num_points2D, 2))
            point3D_ids = np.fromfile(fid, dtype=np.int64, count=num_points2D)
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=name, xys=data, point3D_ids=point3D_ids)
    return images

def read_points3D_binary(path_to_model_file):
    points3D = {}
    if not os.path.exists(path_to_model_file): return {}
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_uint64(fid)
        if num_points is None: return {}

        for _ in range(num_points):
            binary_point_line_properties = fid.read(43)
            if len(binary_point_line_properties) < 43: break
            
            point3D_id = struct.unpack("Q", binary_point_line_properties[0:8])[0]
            xyz = np.frombuffer(binary_point_line_properties[8:32], dtype=np.float64)
            rgb = np.frombuffer(binary_point_line_properties[32:35], dtype=np.uint8)
            error = struct.unpack("d", binary_point_line_properties[35:43])[0]
            
            track_length = read_uint64(fid)
            if track_length is None: break
            
            track_elems = read_uint32(fid) * 2
            fid.read(track_elems * 4) # Skip track content
            
            points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=None, point2D_idxs=None)
    return points3D