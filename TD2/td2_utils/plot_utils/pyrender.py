import numpy as np
import meshplot as mp
import matplotlib.colors
import trimesh
import matplotlib.cm as cm
from tqdm.auto import tqdm
from . import utils
import pyrender 
from PIL import Image, ImageDraw, ImageFont


def euler_to_matrix(rx, ry, rz, order='XYZ'):
    """
    Convert Euler angles (in radians) to a 3x3 rotation matrix.
    Default order is XYZ (rotation around x, then y, then z).
    """
    # Compute individual rotation matrices
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])

    if order == 'XYZ':
        R_x = np.array([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        R_y = np.array([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        R_z = np.array([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        R = R_z @ R_y @ R_x  # Combined rotation
    else:
        raise NotImplementedError(f"Order {order} not supported yet.")

    return R

class PyRenderer:

    def plot_one_side(vertices, faces, cmap=None, points=None, cmap_p=None, rotation=None, colormap='viridis',
                    flat=True, pretty=False, point_size=None, image_size=(800, 600)):
        # Build vertex colors
        if cmap is None:
            vertex_colors = np.ones((vertices.shape[0], 4)) * 0.7  # grey default
        else:
            if cmap.ndim == 1:  # scalar values -> apply colormap
                cmap_norm = (cmap - cmap.min()) / (cmap.ptp() + 1e-8)
                vertex_colors = cm.get_cmap(colormap)(cmap_norm)
            else:
                vertex_colors = np.hstack([cmap, np.ones((cmap.shape[0], 1))])  # add alpha

        # Build trimesh + pyrender mesh
        tm = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=False)
        # --- Compute camera placement ---
        center = tm.bounds.mean(axis=0)
        extents = tm.extents
        max_extent = np.max(extents)

        # Position camera so the mesh fits in frame
        camera_distance = max_extent * 1.8
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = center + np.array([0, 0, camera_distance])  # move camera along +Z
        render_mesh = pyrender.Mesh.from_trimesh(tm, smooth=not flat)

        scene = pyrender.Scene()
        scene.add(render_mesh)

        # Add point cloud as small spheres
        if points is not None:
            if len(points.shape) == 1:
                points = vertices[points]
            if rotation is not None:
                points = points @ rotation.T

            if cmap_p is None:
                colors_p = np.array([[1.0, 0.0, 0.0, 1.0]] * len(points))  # red default
            else:
                if cmap_p.ndim == 1:  # scalar values
                    cmap_norm = (cmap_p - cmap_p.min()) / (cmap_p.ptp() + 1e-8)
                    colors_p = cm.get_cmap(colormap)(cmap_norm)
                else:
                    colors_p = np.hstack([cmap_p, np.ones((cmap_p.shape[0], 1))])

            
            for i, pt in enumerate(points):
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=point_size)
                sphere.visual.vertex_colors = (colors_p[i] * 255).astype(np.uint8)
                sphere.vertices += pt[None, :]
                pm = pyrender.Mesh.from_trimesh(sphere, smooth=True)
                scene.add(pm, pose=np.eye(4, dtype=np.float32)[:3].tolist() + [[0,0,0,1]])
                #scene.nodes[-1].matrix[:3, 3] = pt

        # Add a camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light, pose=np.eye(4))

        # Render offscreen
        r = pyrender.OffscreenRenderer(viewport_width=image_size[0],
                                    viewport_height=image_size[1])
        color, depth = r.render(scene)
        return color, depth
        

    def plot_multiple_sides(mesh1, cmap=None, points=None, cmap_p=None, rotation=None, colormap='viridis',
                    flat=True, pretty=False, point_size=None, image_size=(800, 600),
                    margin=20, label_height=40, font_path=None, return_all=False):
        """
        Static rendering of a mesh (and optional point cloud) using pyrender.
        Renders from 6 standard viewpoints and merges the results.

        Returns
        -------
        merged_img : np.ndarray - combined RGBA image
        images     : list[np.ndarray] - list of individual rendered images
        """

        # Rotate mesh vertices if needed
        vertices = mesh1.vertices.copy()
        faces = mesh1.faces

        if point_size is None:
            mesh_area = mesh1.A.sum()
            if mesh_area is not None:
                point_size = .01*np.sqrt(mesh_area)
            else:
                point_size = 0.01
        tri = trimesh.base.Trimesh(vertices, faces, process=False)
        vertices = tri.vertices - tri.centroid
        if points is not None:
            if len(points.shape) == 2:
                points = points - tri.centroid
        rotations = np.array([
        [0.0, 0.0, 0.0],             # front view
        [0.0, np.pi/2, 0.0],         # side view (right)
        [0.0, -np.pi/2, 0.0],        # side view (left)
        [np.pi/4, np.pi/4, 0.0],     # top-front diagonal
        [-np.pi/4, np.pi/4, 0.0],    # bottom-front diagonal
        [0.0, np.pi, 0.0],           # back view
        ])

        labels = ["Front", "Right", "Left", "Top-Front", "Bottom-Front", "Back"]

        # Font
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", 28)
        except:
            font = ImageFont.load_default()

        rendered_images = []

        # Render each rotation
        for rot, label in zip(rotations, labels):
            R = euler_to_matrix(*rot)
            verts_rot = vertices @ R.T
            if points is not None:
                if len(points.shape) == 2:
                    points_rot = points @ R.T
                else:
                    points_rot = points.copy()
            else:
                points_rot = None
            img_rot, _ = PyRenderer.plot_one_side(verts_rot, faces, cmap=cmap, points=points_rot, cmap_p=cmap_p, colormap=colormap,
                    flat=flat, pretty=pretty, point_size=point_size, image_size=(400, 300))

        # Convert to PIL
            img = Image.fromarray(img_rot)
            draw = ImageDraw.Draw(img)

            # Label background
            w, h = img.size
            draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 180))

            # Text sizing (Pillow ≥10)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Draw centered label
            draw.text(((w - text_w) / 2, (label_height - text_h) / 2),
                    label, fill=(255, 255, 255, 255), font=font)

            rendered_images.append(img)

        # Compose grid (2 rows × 3 columns)
        w, h = rendered_images[0].size
        grid_w = 3 * w + 4 * margin
        grid_h = 2 * h + 3 * margin
        merged = Image.new("RGBA", (grid_w, grid_h), (30, 30, 30, 255))

        positions = []
        for i, img in enumerate(rendered_images):
            row, col = divmod(i, 3)
            x = margin + col * (w + margin)
            y = margin + row * (h + margin)
            merged.paste(img, (x, y))
            positions.append((x, y))

        if return_all:
            return merged.convert("RGB"), rendered_images
        else:
            return merged.convert("RGB")
        
    def plot(mesh1, cmap=None, multiple_sides=True, points=None, cmap_p=None, rotation=None, colormap='viridis',
                    flat=True, pretty=False, point_size=None, image_size=(800, 600),
                    margin=20, label_height=40, font_path=None, return_all=False):
        """
        Static rendering of a mesh (and optional point cloud) using pyrender.
        Renders from 6 standard viewpoints and merges the results.

        Returns
        -------
        merged_img : np.ndarray - combined RGBA image
        images     : list[np.ndarray] - list of individual rendered images
        """
        if multiple_sides:
            return PyRenderer.plot_multiple_sides(mesh1, cmap, points, cmap_p, rotation, colormap,
                    flat, pretty, point_size, image_size, margin=margin, label_height=label_height, font_path=font_path, return_all=return_all)
        else:
            return PyRenderer.plot_one_side(mesh1.vertices, mesh1.faces, cmap, points, cmap_p, rotation, colormap,
                    flat, pretty, point_size, image_size)
        

    def plot_texture_side(vertices, faces, uv, texture_img="texture.png", wireframe=False, resolution=(640, 480)):
        """
        Render a textured mesh offscreen using pyrender, with a white background and automatic framing.

        Parameters
        ----------
        vertices : (N, 3) array
            3D vertex coordinates.
        faces : (M, 3) array
            Triangular face indices.
        uv : (N, 2) array
            UV coordinates (range [0, 1]).
        wireframe : bool, optional
            If True, overlay a wireframe mesh.
        output_path : str, optional
            Output image filename.
        resolution : tuple, optional
            (width, height) of render.
        """
        uv = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
        # --- Load texture properly ---
        texture_image = Image.open(texture_img).convert("RGBA")
        texture_image = np.asarray(texture_image).astype(np.uint8)

        # --- Create textured mesh ---
        tex_visuals = trimesh.visual.texture.TextureVisuals(uv=uv, image=texture_image)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=tex_visuals, process=False)

        # --- Compute camera placement ---
        center = mesh.bounds.mean(axis=0)
        extents = mesh.extents
        max_extent = np.max(extents)

        # Position camera so the mesh fits in frame
        camera_distance = max_extent * 1.8
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = center + np.array([0, 0, camera_distance])  # move camera along +Z

        # --- Convert mesh to pyrender mesh ---
        render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

        # --- Scene setup ---
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255])  # white background
        scene.add(render_mesh)

        # --- Lighting ---
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
        scene.add(light, pose=cam_pose)  # align light with camera

        # Fill light (from above-left)
        light2_pose = np.eye(4)
        light2_pose[:3, 3] = center + np.array([-camera_distance / 2, camera_distance / 2, camera_distance])
        light2 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
        scene.add(light2, pose=light2_pose)

        # Back light (from behind)
        light3_pose = np.eye(4)
        light3_pose[:3, 3] = center + np.array([camera_distance / 2, -camera_distance / 2, -camera_distance])
        light3 = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)
        scene.add(light3, pose=light3_pose)

        # --- Camera ---
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        scene.add(camera, pose=cam_pose)

        # --- Optional wireframe overlay ---
        if wireframe:
            wf_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            wf_mesh = pyrender.Mesh.from_trimesh(wf_trimesh, wireframe=True)
            scene.add(wf_mesh)

        # --- Render offscreen ---
        r = pyrender.OffscreenRenderer(*resolution)
        color, depth = r.render(scene)
        r.delete()

        # --- Save result ---
        return Image.fromarray(color).convert("RGB")
    
    def plot_texture_multiple_sides(vertices, faces, uv, texture_img, wireframe=False, resolution=(640, 480),
                                    margin=20, label_height=40, font_path=None, return_all=False):
        """
        Static rendering of a mesh (and optional point cloud) using pyrender.
        Renders from 6 standard viewpoints and merges the results.

        Returns
        -------
        merged_img : np.ndarray - combined RGBA image
        images     : list[np.ndarray] - list of individual rendered images
        """

        # Rotate mesh vertices if needed
        tri = trimesh.base.Trimesh(vertices, faces, process=False)
        vertices = tri.vertices - tri.centroid
        rotations = np.array([
        [0.0, 0.0, 0.0],             # front view
        [0.0, np.pi/2, 0.0],         # side view (right)
        [0.0, -np.pi/2, 0.0],        # side view (left)
        [np.pi/4, np.pi/4, 0.0],     # top-front diagonal
        [-np.pi/4, np.pi/4, 0.0],    # bottom-front diagonal
        [0.0, np.pi, 0.0],           # back view
        ])

        labels = ["Front", "Right", "Left", "Top-Front", "Bottom-Front", "Back"]

        # Font
        try:
            font = ImageFont.truetype(font_path or "arial.ttf", 28)
        except:
            font = ImageFont.load_default()

        rendered_images = []

        # Render each rotation
        for rot, label in zip(rotations, labels):
            R = euler_to_matrix(*rot)
            verts_rot = vertices @ R.T

            img_rot = PyRenderer.plot_texture_side(verts_rot, faces, uv, texture_img, wireframe, resolution)

            # Convert to PIL
            draw = ImageDraw.Draw(img_rot)

            # Label background
            w, h = img_rot.size
            draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 180))

            # Text sizing (Pillow ≥10)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Draw centered label
            draw.text(((w - text_w) / 2, (label_height - text_h) / 2),
                    label, fill=(255, 255, 255, 255), font=font)

            rendered_images.append(img_rot)

        # Compose grid (2 rows × 3 columns)
        w, h = rendered_images[0].size
        grid_w = 3 * w + 4 * margin
        grid_h = 2 * h + 3 * margin
        merged = Image.new("RGBA", (grid_w, grid_h), (30, 30, 30, 255))

        positions = []
        for i, img in enumerate(rendered_images):
            row, col = divmod(i, 3)
            x = margin + col * (w + margin)
            y = margin + row * (h + margin)
            merged.paste(img, (x, y))
            positions.append((x, y))

        if return_all:
            return merged.convert("RGB"), rendered_images
        else:
            return merged.convert("RGB")
        
    def plot_texture(vertices, faces, uv, texture_img="texture.png", wireframe=False, multiple_sides=True, resolution=(640, 480),
                    margin=20, label_height=40, font_path=None, return_all=False):
        if multiple_sides:
            return PyRenderer.plot_texture_multiple_sides(vertices, faces, uv, texture_img=texture_img, wireframe=wireframe, resolution=(640, 480),
                                    margin=margin, label_height=label_height, font_path=font_path, return_all=return_all)
        else:
            return PyRenderer.plot_texture_side(vertices, faces, uv, texture_img=texture_img, wireframe=wireframe, resolution=resolution)