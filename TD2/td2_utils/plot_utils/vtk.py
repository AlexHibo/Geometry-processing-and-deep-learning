from plot_utils.vtkviz.vtkVisualization import VTKSurface, VTKPointCloudSphere, VTKVisualization, center_mesh_give_size
from PIL import Image, ImageFont, ImageDraw
from xvfbwrapper import Xvfb
import trimesh
import matplotlib.cm as cm
import numpy as np

# This is just that y and z may be inverted depending on what you are used to.
dir_x = np.array([1, 0, 0])
dir_y = np.array([0, 0, 1])
dir_z = np.array([0, 1, 0])

display = Xvfb(width=1280, height=740, colordepth=16)
display.start()


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


class vtkRenderer:

    def plot_one_side(vertices, faces, cmap=None, points=None, cmap_p=None, rotation=None, colormap='viridis',
                        flat=True, pretty=False, point_size=None, image_size=(800, 600)):
        tri = trimesh.base.Trimesh(vertices, faces, process=False)
        vertices = tri.vertices - tri.centroid
        p1, p2 = center_mesh_give_size(vertices)
        center_end = ((p2 + p1) / 2) * dir_z
        surf_actor = VTKSurface(vertices-center_end, faces, color=cmap)

        render = VTKVisualization()
        
        render.add_entity(surf_actor)
        
        if points is not None:
            if len(points.shape) == 1:
                points = vertices[points]
            if rotation is not None:
                points = points @ rotation.T

            if cmap_p is None:
                colors_p = np.array([[255., 0.0, 0.0]] * len(points))  # red default
            else:
                if cmap_p.ndim == 1:  # scalar values
                    cmap_norm = (cmap_p - cmap_p.min()) / (cmap_p.ptp() + 1e-8)
                    colors_p = cm.get_cmap(colormap)(cmap_norm)[:, :3]  * 255.#we work with RGB, no alpha hahahaha
                else:
                    colors_p = np.hstack([cmap_p, np.ones((cmap_p.shape[0], 1))])
                print(colors_p.shape)
            pt_actor = VTKPointCloudSphere(points - center_end, colors_p, scale=point_size)  
            render.add_entity(pt_actor)
        
        render.change_camera(dir_z, 2 * (p1 - p2) * dir_y, [0, 0, 0])
        img = render.toPIL(background=[1, 1, 1], size=image_size)
        return img

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
                point_size = .1*np.sqrt(mesh_area)
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
            img_rot = vtkRenderer.plot_one_side(verts_rot, faces, cmap=cmap, points=points_rot, cmap_p=cmap_p, colormap=colormap,
                    flat=flat, pretty=pretty, point_size=point_size, image_size=(400, 300))

            temp_draw = ImageDraw.Draw(img_rot)
            bbox = temp_draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            label_height = text_h + 10  # add some padding
        
            # Create a new image with extra space on top
            w, h = img_rot.size
            new_h = h + label_height
            new_img = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
        
            # Paste the original image *below* the new top margin
            new_img.paste(img_rot, (0, label_height))
        
            # Draw label area
            draw = ImageDraw.Draw(new_img)
            draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 180))
        
            # Center text
            text_x = (w - text_w) / 2
            text_y = (label_height - text_h) / 2
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)
        
            rendered_images.append(new_img)


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
            return vtkRenderer.plot_multiple_sides(mesh1, cmap, points, cmap_p, rotation, colormap,
                    flat, pretty, point_size, image_size, margin=margin, label_height=label_height, font_path=font_path, return_all=return_all)
        else:
            return vtkRenderer.plot_one_side(mesh1.vertices, mesh1.faces, cmap, points, cmap_p, rotation, colormap,
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

        tri = trimesh.base.Trimesh(vertices, faces, process=False)
        vertices = tri.vertices - tri.centroid
        p1, p2 = center_mesh_give_size(vertices)
        center_end = ((p2 + p1) / 2) * dir_z
        surf_actor = VTKSurface(vertices-center_end, faces, uv=uv, color=texture_img)
        render = VTKVisualization()
            
        render.add_entity(surf_actor)
        
        render.change_camera(dir_z, 2 * (p1 - p2) * dir_y, [0, 0, 0])
        img = render.toPIL(background=[1, 1, 1], size=resolution)
        return img

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
            img_rot = vtkRenderer.plot_texture_side(verts_rot, faces, uv, texture_img, wireframe, resolution)

            temp_draw = ImageDraw.Draw(img_rot)
            bbox = temp_draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            label_height = text_h + 10  # add some padding
        
            # Create a new image with extra space on top
            w, h = img_rot.size
            new_h = h + label_height
            new_img = Image.new("RGBA", (w, new_h), (0, 0, 0, 0))
        
            # Paste the original image *below* the new top margin
            new_img.paste(img_rot, (0, label_height))
        
            # Draw label area
            draw = ImageDraw.Draw(new_img)
            draw.rectangle([(0, 0), (w, label_height)], fill=(0, 0, 0, 180))
        
            # Center text
            text_x = (w - text_w) / 2
            text_y = (label_height - text_h) / 2
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255), font=font)
        
            rendered_images.append(new_img)


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
            return vtkRenderer.plot_texture_multiple_sides(vertices, faces, uv, texture_img=texture_img, wireframe=wireframe, resolution=(640, 480),
                                    margin=margin, label_height=label_height, font_path=font_path, return_all=return_all)
        else:
            return vtkRenderer.plot_texture_side(vertices, faces, uv, texture_img=texture_img, wireframe=wireframe, resolution=resolution)