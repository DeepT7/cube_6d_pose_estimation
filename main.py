import cv2
import numpy as np 
import subprocess
import ast
from cube_vertices import detect_vertex
img = cv2.imread("data/frame10.png")

# Run the model
def run_dexined():
    dexined_path = '.\DexiNed'  
    command = f"python {dexined_path}\main.py"
    subprocess.run(command, shell=True, check=True)  # Execute and check output
    processed_image_path = 'result/BIPED2CLASSIC/fused/frame10.png'  # Output path of the model
    return processed_image_path

edge_img_path = run_dexined()
edge_img_path = 'result/BIPED2CLASSIC/fused/frame10.png'
# edge_img = cv2.imread(edge_img_path)

# Detect corners of the cube from the edges
detect_vertex(edge_img_path)

with open("points.txt", "r") as file:
    data = file.read()

# Convert the string to a list of tuples
array_2D_points = ast.literal_eval(data)

points_2D = np.array(array_2D_points[1:]
, dtype=np.float32)
X = (0.0, 0.0, 0.0)
A = (-2.8,-2.8, 2.8)
B = (-2.8, 2.8, 2.8)
C = (2.8, 2.8, 2.8)
D = (2.8,-2.8, 2.8)
E = (-2.8,-2.8,-2.8)
F = (-2.8, 2.8,-2.8)
G = (2.8, 2.8, -2.8)
H = (2.8,-2.8, -2.8)

points_3D = np.array([A, B, C, D, G, H])

dist_coeffs = np.array([[-9.14692714e-02],
                        [1.55892004], 
                        [4.77596533e-03], 
                        [-2.23396589e-03]])

# dist_coeffs = np.zeros((4, 1))
matrix_camera = np.array([[709.70715987, 0.0, 316.02753041],
                          [0.0, 710.15430705, 262.7973154 ],
                          [0.0, 0.0, 1.0]])

success, vector_rotation, vector_translation = cv2.solvePnP(points_3D, points_2D, matrix_camera, dist_coeffs, flags = 0)

cube_2D_center, jacobian = cv2.projectPoints(
    np.array([(0.0, 0.0, 50.0)]),
    vector_rotation,
    vector_translation,
    matrix_camera,
    dist_coeffs,
)

# for p in points_2D:
#     cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
# point1 = (int(points_2D[0][0]), int(points_2D[0][1]))
# point2 = (int(cube_2D_center[0][0][0]), int(cube_2D_center[0][0][1]))

# cv2.line(img, point1, point2, (255, 0, 0), 1)
# Create a 3x3 rotation matrix from the rotation vector
rotation_matrix, _ = cv2.Rodrigues(vector_rotation)

# Define the length of the coordinate axes
axis_length = 150

# Define the origin, center of the cube
ox = array_2D_points[0][0]
oy = array_2D_points[0][1]
origin = (ox, oy, 0)

# Define the X, Y, and Z axes in the world coordinate system
x_axis = axis_length * np.array([1, 0, 0])
y_axis = axis_length * np.array([0, 1, 0])
z_axis = axis_length * np.array([0, 0, 1])

# Rotate the axes using the rotation matrix
x_axis_rotated = np.dot(rotation_matrix, x_axis) + origin
y_axis_rotated = np.dot(rotation_matrix, y_axis) + origin
z_axis_rotated = np.dot(rotation_matrix, z_axis) + origin

# Create an image
img_size = (500, 500, 3)
image = np.ones(img_size, dtype=np.uint8) * 255

# Draw the rotated X, Y, and Z axes
origin = tuple(map(int, origin[:2]))
x_axis_end = tuple(map(int, x_axis_rotated[:2]))
y_axis_end = tuple(map(int, y_axis_rotated[:2]))
z_axis_end = tuple(map(int, z_axis_rotated[:2]))

cv2.line(img, origin, x_axis_end, (255, 0, 0), 2)  # Draw X-axis in blue
cv2.line(img, origin, y_axis_end, (0, 255, 0), 2)  # Draw Y-axis in green
cv2.line(img, origin, z_axis_end, (0, 0, 255), 2)  # Draw Z-axis in red

# Calculate distance 
tx = vector_translation[0][0]
ty = vector_translation[1][0]
tz = vector_translation[2][0]
distance = np.sqrt(tx**2 + ty**2 + tz**2)
distance_str = f"Distance: {distance:.2f}"
cv2.putText(img, distance_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

angle = np.linalg.norm(vector_rotation)
axis = vector_rotation / angle
angle_degrees = angle * 180 / np.pi
cos_x = axis[0]
cos_y = axis[1]
cos_z = axis[2]
angle_ox_rad = np.arccos(cos_x)
angle_oy_rad = np.arccos(cos_y)
angle_oz_rad = np.arccos(cos_z)
angle_ox_degrees = angle_ox_rad * 180 / np.pi
angle_oy_degrees = angle_oy_rad * 180 / np.pi
angle_oz_degrees = angle_oz_rad * 180 / np.pi
angles_str = f"""Ox: {angle_ox_degrees[0]:.2f} Oy: {angle_oy_degrees[0]:.2f} Oz: {angle_oz_degrees[0]:.2f}"""
cv2.putText(img, angles_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


cv2.imshow("Final", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(vector_rotation)
print(vector_translation)
print(distance)
print("Rotation angles (degrees):")
print("Ox:", angle_ox_degrees)
print("Oy:", angle_oy_degrees)
print("Oz:", angle_oz_degrees)