#!/usr/bin/env python

import SimpleITK as sitk
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def compute_axis_angle(ref_direction, img_direction):
    # Compute the rotation matrix R = img_direction * ref_direction.T
    R_matrix = img_direction @ ref_direction.T

    # If the rotations are close to identical, return zero difference
    if np.allclose(R_matrix, np.eye(3), atol=1e-6):
        # Return an arbitrary axis (here, the x-axis) and a 0 rotation angle
        return np.array([1.0, 0.0, 0.0]), 0.0

    # Convert R into axis-angle representation
    rotation = R.from_matrix(R_matrix)
    axis_angle = rotation.as_rotvec()

    # Compute the angle in degrees
    angle_degrees = np.linalg.norm(axis_angle) * (180.0 / np.pi)
    axis = axis_angle / np.linalg.norm(axis_angle)

    return axis, angle_degrees

def compute_origin_offset(ref_origin, img_origin):
    origin_shift = np.array(img_origin) - np.array(ref_origin)
    offset_distance = np.linalg.norm(origin_shift)
    return origin_shift, offset_distance

def process_images(reference_image_path, image_paths):
    reference_image = sitk.ReadImage(reference_image_path)

    if reference_image.GetDimension() == 4:
        direction_matrix_4d = np.array(reference_image.GetDirection()).reshape(4, 4)
        ref_direction = direction_matrix_4d[:3, :3]
        origin_4D = reference_image.GetOrigin()
        ref_origin = origin_4D[:3]
    else:
        ref_direction = np.array(reference_image.GetDirection()).reshape((3,3))
        ref_origin = reference_image.GetOrigin()

    print("Image,Rotation_axis_x,Rotation_axis_y,Rotation_axis_z,Rotation_angle_deg,offset_x,offset_y,offset_z,offset_distance")

    for path in image_paths:
        image = sitk.ReadImage(path)

        if image.GetDimension() == 4:
            direction_matrix_4d = np.array(image.GetDirection()).reshape(4, 4)
            img_direction = direction_matrix_4d[:3, :3]
            origin_4D = image.GetOrigin()
            img_origin = origin_4D[:3]
        else:
            img_direction = np.array(image.GetDirection()).reshape((3,3))
            img_origin = image.GetOrigin()

        axis, angle_degrees = compute_axis_angle(ref_direction, img_direction)

        origin_shift, offset_distance = compute_origin_offset(ref_origin, img_origin)

        print(f"{path},{axis[0]},{axis[1]},{axis[2]},{angle_degrees},{origin_shift[0]},{origin_shift[1]},{origin_shift[2]},{offset_distance}")


def main():
    parser = argparse.ArgumentParser(description="Compute rotation and origin offset between a reference image and a list of images.")
    parser.add_argument('reference_image_path', type=str, help='Path to the reference image')
    parser.add_argument('image_paths', nargs='+', help='Paths to the images to be processed')
    args = parser.parse_args()

    process_images(args.reference_image_path, args.image_paths)

if __name__ == "__main__":
    main()
