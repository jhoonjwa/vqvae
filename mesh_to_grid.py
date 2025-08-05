import numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

def load_glb_mesh(glb_path):
    mesh = o3d.io.read_triangle_mesh(str(glb_path))
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    if mesh.triangle_uvs is not None and len(mesh.triangle_uvs) > 0:
        triangle_uvs = np.asarray(mesh.triangle_uvs)
        uvs = np.zeros((len(vertices), 2))
        uv_counts = np.zeros(len(vertices))
        
        for face_idx, face in enumerate(faces):
            for i in range(3):
                vertex_idx = face[i]
                uv_idx = face_idx * 3 + i
                if uv_idx < len(triangle_uvs):
                    uvs[vertex_idx] += triangle_uvs[uv_idx]
                    uv_counts[vertex_idx] += 1
        
        valid_vertices = uv_counts > 0
        uvs[valid_vertices] = uvs[valid_vertices] / uv_counts[valid_vertices, np.newaxis]
    else:
        min_coords = np.min(vertices[:, :2], axis=0)
        max_coords = np.max(vertices[:, :2], axis=0)
        coord_range = max_coords - min_coords
        coord_range[coord_range == 0] = 1.0
        uvs = (vertices[:, :2] - min_coords) / coord_range
    
    return vertices, faces, uvs

def calculate_triangle_midpoints_and_vertices(vertices, faces, uvs):
    triangle_vertices_3d = vertices[faces]
    triangle_uvs = uvs[faces]
    midpoints_uv = np.mean(triangle_uvs, axis=1)
    return midpoints_uv, triangle_vertices_3d

def create_uv_grid_mapping(midpoints_uv, triangle_vertices_3d, grid_size=512):
    min_u, min_v = np.min(midpoints_uv, axis=0)
    max_u, max_v = np.max(midpoints_uv, axis=0)
    
    grid = np.zeros((grid_size, grid_size), dtype=int)
    vertex_data = {}
    
    u_scale = (grid_size - 1) / (max_u - min_u) if max_u != min_u else 1
    v_scale = (grid_size - 1) / (max_v - min_v) if max_v != min_v else 1
    
    grid_coords = np.zeros((len(midpoints_uv), 2), dtype=int)
    grid_coords[:, 0] = np.clip(((midpoints_uv[:, 0] - min_u) * u_scale).astype(int), 0, grid_size - 1)
    grid_coords[:, 1] = np.clip(((midpoints_uv[:, 1] - min_v) * v_scale).astype(int), 0, grid_size - 1)
    
    for i, (grid_u, grid_v) in enumerate(grid_coords):
        grid[grid_u, grid_v] += 1
        key = f"{grid_u},{grid_v}"
        if key not in vertex_data:
            vertex_data[key] = []
        
        triangle_data = {
            'vertices_3d': triangle_vertices_3d[i].flatten(),
            'uv_coord': midpoints_uv[i],
            'triangle_idx': i
        }
        vertex_data[key].append(triangle_data)
    
    return grid, vertex_data, (min_u, min_v, max_u, max_v)

def return_push_direction(grid, i, j):
    rows, cols = grid.shape
    
    right = sum(grid[i, x] == 0 for x in range(j+1, cols))
    left = sum(grid[i, x] == 0 for x in range(j))
    down = sum(grid[x, j] == 0 for x in range(i+1, rows))
    up = sum(grid[x, j] == 0 for x in range(i))
    
    counts = [right, down, left, up]
    max_idx = counts.index(max(counts))
    return [(1, 0), (0, 1), (-1, 0), (0, -1)][max_idx]

def sort_triangles_for_pushing(triangles, direction):
    du, dv = direction
    
    if du > 0:
        return sorted(triangles, key=lambda t: t['uv_coord'][0], reverse=True)
    elif du < 0:
        return sorted(triangles, key=lambda t: t['uv_coord'][0])
    elif dv > 0:
        return sorted(triangles, key=lambda t: t['uv_coord'][1], reverse=True)
    else:
        return sorted(triangles, key=lambda t: t['uv_coord'][1])

def push_triangles_with_vertices(grid, vertex_data, i, j, direction, duplicate_limit=1):
    du, dv = direction
    rows, cols = grid.shape
    
    if grid[i, j] <= duplicate_limit:
        return
    
    key = f"{i},{j}"
    if key not in vertex_data:
        return
    
    triangles = vertex_data[key]
    sorted_triangles = sort_triangles_for_pushing(triangles, direction)
    
    vertex_data[key] = sorted_triangles[:duplicate_limit]
    triangles_to_push = sorted_triangles[duplicate_limit:]
    grid[i, j] = duplicate_limit
    
    if du != 0:
        step = 1 if du > 0 else -1
        current_col = j + step
        
        for triangle in triangles_to_push:
            pushed = False
            temp_col = current_col
            
            while temp_col >= 0 and temp_col < cols and not pushed:
                temp_key = f"{i},{temp_col}"
                
                if grid[i, temp_col] == 0:
                    grid[i, temp_col] = 1
                    if temp_key not in vertex_data:
                        vertex_data[temp_key] = []
                    vertex_data[temp_key].append(triangle)
                    pushed = True
                else:
                    if temp_key in vertex_data and len(vertex_data[temp_key]) > 0:
                        existing_triangle = vertex_data[temp_key].pop(0)
                        vertex_data[temp_key].append(triangle)
                        triangle = existing_triangle
                    temp_col += step
    else:
        step = 1 if dv > 0 else -1
        current_row = i + step
        
        for triangle in triangles_to_push:
            pushed = False
            temp_row = current_row
            
            while temp_row >= 0 and temp_row < rows and not pushed:
                temp_key = f"{temp_row},{j}"
                
                if grid[temp_row, j] == 0:
                    grid[temp_row, j] = 1
                    if temp_key not in vertex_data:
                        vertex_data[temp_key] = []
                    vertex_data[temp_key].append(triangle)
                    pushed = True
                else:
                    if temp_key in vertex_data and len(vertex_data[temp_key]) > 0:
                        existing_triangle = vertex_data[temp_key].pop(0)
                        vertex_data[temp_key].append(triangle)
                        triangle = existing_triangle
                    temp_row += step

def pushing_algorithm_with_vertices(grid, vertex_data, duplicate_limit=1):
    for i in (range(grid.shape[0])):
        for j in range(grid.shape[1]):
            if grid[i, j] > duplicate_limit:
                direction = return_push_direction(grid, i, j)
                push_triangles_with_vertices(grid, vertex_data, i, j, direction, duplicate_limit)
    
    return grid, vertex_data

def create_final_array(grid, vertex_data, grid_size=512):
    final_array = np.zeros((grid_size, grid_size, 9), dtype=np.float32)
    
    for i in range(grid_size):
        for j in range(grid_size):
            key = f"{i},{j}"
            if key in vertex_data and len(vertex_data[key]) > 0:
                triangle = vertex_data[key][0]
                final_array[i, j, :] = triangle['vertices_3d']
    
    return final_array

def reconstruct_mesh_from_array(array_512x512x9, output_path):
    """
    Reconstruct 3D mesh from 512x512x9 array and save as GLB.
    """
    print("Starting mesh reconstruction...")
    print(f"Array shape: {array_512x512x9.shape}")
    print(f"Array dtype: {array_512x512x9.dtype}")
    print(f"Non-zero elements in array: {np.count_nonzero(array_512x512x9)}")
    print(f"Array min: {np.min(array_512x512x9)}, max: {np.max(array_512x512x9)}")
    
    vertices = []
    faces = []
    
    non_zero_cells = 0
    for i in range(512):
        for j in range(512):
            if np.any(array_512x512x9[i, j, :] != 0):
                non_zero_cells += 1
                # Extract triangle vertices from 9-channel data
                triangle_verts = array_512x512x9[i, j, :].reshape(3, 3)
                
                # Add vertices to list
                start_idx = len(vertices)
                vertices.extend(triangle_verts)
                
                # Add face indices
                faces.append([start_idx, start_idx + 1, start_idx + 2])
    
    print(f"Found {non_zero_cells} non-zero cells")
    print(f"Generated {len(vertices)} vertices and {len(faces)} faces")
    
    if len(vertices) == 0:
        print("ERROR: No vertices generated! Array appears to be empty.")
        return 0, 0
    
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    print(f"Vertices array shape: {vertices.shape}")
    print(f"Faces array shape: {faces.shape}")
    
    # Validate data before creating mesh
    print("Validating mesh data...")
    print(f"Vertices min: {np.min(vertices, axis=0)}")
    print(f"Vertices max: {np.max(vertices, axis=0)}")
    print(f"Face indices min: {np.min(faces)}, max: {np.max(faces)}")
    
    # Check for invalid face indices
    if np.max(faces) >= len(vertices):
        print(f"ERROR: Face indices exceed vertex count! Max face index: {np.max(faces)}, Vertex count: {len(vertices)}")
        return 0, 0
    
    # Skip Open3D mesh creation and write files manually to avoid segfault
    print("Skipping Open3D mesh creation, writing files manually...")
    
    # Write OBJ file manually
    obj_path = output_path.replace('.glb', '.obj')
    try:
        print(f"Writing OBJ file manually: {obj_path}")
        with open(obj_path, 'w') as f:
            f.write("# OBJ file generated from UV grid reconstruction\n")
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        if Path(obj_path).exists():
            obj_size = Path(obj_path).stat().st_size
            print(f"OBJ saved successfully, size: {obj_size} bytes")
            return len(vertices), len(faces)
        else:
            print("OBJ file was not created")
            
    except Exception as e:
        print(f"Error writing OBJ manually: {e}")
        import traceback
        traceback.print_exc()
        

def process_glb_to_uv_grid(glb_path, grid_size=512, duplicate_limit=1):
    vertices, faces, uvs = load_glb_mesh(glb_path)
    original_triangle_count = len(faces)
    
    midpoints_uv, triangle_vertices_3d = calculate_triangle_midpoints_and_vertices(vertices, faces, uvs)
    grid, vertex_data, uv_bounds = create_uv_grid_mapping(midpoints_uv, triangle_vertices_3d, grid_size)
    
    final_grid, final_vertex_data = pushing_algorithm_with_vertices(grid, vertex_data, duplicate_limit)
    final_array = create_final_array(final_grid, final_vertex_data, grid_size)
    
    preserved_triangles = np.sum(final_grid > 0)
    preservation_rate = (preserved_triangles / original_triangle_count) * 100
    
    # print(f"Original: {original_triangle_count}, Preserved: {preserved_triangles}, Rate: {preservation_rate:.1f}%")
    
    return final_array, final_grid, final_vertex_data, uv_bounds

if __name__ == "__main__":
    glb_path = r"C:\Users\user\Desktop\ALIN\3D LLM Agent\mid_polys\model_0001_20b59782.glb"
    output_path = r"C:\Users\user\Desktop\ALIN\3D LLM Agent\mid_polys\model_0001_20b59782_recon.glb"
    
    # Process GLB to grid
    final_array, grid, vertex_data, uv_bounds = process_glb_to_uv_grid(glb_path)
    
    # Save array
    # np.save("uv_grid_array.npy", final_array)
    
    # Reconstruct mesh from array
    num_vertices, num_faces = reconstruct_mesh_from_array(final_array, output_path)
    print(f"Reconstructed mesh: {num_vertices} vertices, {num_faces} faces")