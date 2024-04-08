#!/usr/bin/env python

import ants
import SimpleITK as sitk
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def read_bvec(bvec_path):
    return np.loadtxt(bvec_path)

def write_bvec(bvecs, out_path):
    np.savetxt(out_path, bvecs, fmt='%0.6f')

def normalize_bvecs(bvecs):
    norms = np.linalg.norm(bvecs, axis=0)
    norms[norms == 0] = 1  # Avoid division by zero for zero vectors
    return bvecs / norms

def transform_bvecs(bvecs, R_ref, R_mov):

    # Compute R_r^{-1}
    R_ref_inv = np.linalg.inv(R_ref)

    # Perform the transformation for each b-vector
    transformed_bvecs = R_ref_inv @ R_mov @ bvecs

    # Renormalize the b-vectors to unit length, important for diffusion MRI analysis
    norms = np.linalg.norm(transformed_bvecs, axis=0)
    norms[norms == 0] = 1  # Prevent division by zero for zero vectors
    transformed_bvecs = transformed_bvecs / norms

    return transformed_bvecs

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

def resample_to_reference(ref_image_file, mov_image_file, output_prefix):

    ref_image = ants.image_read(ref_image_file)
    mov_image = ants.image_read(mov_image_file)

    first_ref_volume = ants.utils.slice_image(ref_image, axis=3, idx=0)

    identity = ants.create_ants_transform(precision='float', dimension=3)

    tmp_identity_file = '/tmp/identity.txt'

    ants.write_transform(identity, tmp_identity_file)

    output = ants.apply_transforms(fixed=first_ref_volume, moving=mov_image, transformlist=[tmp_identity_file],
                                   interpolator='linear', verbose=True, dimension=3, imagetype=3)

    ants.image_write(output, f"{output_prefix}.nii.gz")


def main(reference_image_path, moving_image_prefix, threshold, output_prefix):
    # Load images
    ref_image = sitk.ReadImage(reference_image_path)
    mov_image = sitk.ReadImage(moving_image_prefix + ".nii.gz")

    if ref_image.GetDimension() == 4:
        direction_matrix_4d = np.array(ref_image.GetDirection()).reshape(4, 4)
        ref_direction = direction_matrix_4d[:3, :3]
    else:
        ref_direction = np.array(ref_image.GetDirection()).reshape((3,3))

    if mov_image.GetDimension() != ref_image.GetDimension():
        raise ValueError("Reference and moving images have different dimensions")

    if mov_image.GetDimension() == 4:
        direction_matrix_4d = np.array(mov_image.GetDirection()).reshape(4, 4)
        mov_direction = direction_matrix_4d[:3, :3]
    else:
        mov_direction = np.array(mov_image.GetDirection()).reshape((3,3))

    # Compute rotation needed
    axis,angle = compute_axis_angle(ref_direction, mov_direction)

    print("Rotation axis: " + str(axis))
    print("Rotation angle in degrees: " + str(angle))

    # Decide action based on angle threshold
    if angle > threshold:
        # Resample moving image into reference image space
        resample_to_reference(reference_image_path, moving_image_prefix + ".nii.gz", output_prefix)
        # Rotate bvecs
        bvecs = read_bvec(moving_image_prefix + ".bvec")
        bvecs_rotated = transform_bvecs(bvecs, ref_direction, mov_direction)
        bvecs_out_path = f"{output_prefix}.bvec"
        write_bvec(bvecs_rotated, bvecs_out_path)
    else:
        # Set direction and origin of moving image to that of reference image
        mov_image.SetDirection(ref_image.GetDirection())
        mov_image.SetOrigin(ref_image.GetOrigin())
        sitk.WriteImage(mov_image, f"{output_prefix}.nii.gz")
        # bvecs unchanged
        bvecs = read_bvec(bvec_path)
        bvecs_out_path = f"{output_prefix}.bvec"
        write_bvec(bvecs, bvecs_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and bvecs based on rotation threshold.")
    parser.add_argument("reference_image_path", help="Path to the reference image")
    parser.add_argument("moving_image_prefix", help="Path to the moving image .nii.gz and .bvec")
    parser.add_argument("threshold", help="Angular threshold below which we don't resample bvecs", type=float)
    parser.add_argument("output_prefix", help="Output prefix for the resampled image and bvec file")

    args = parser.parse_args()

    main(args.reference_image_path, args.moving_image_prefix, args.threshold, args.output_prefix)
