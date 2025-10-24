#!/usr/bin/env python

import ants
import numpy as np
import argparse
import os
import shutil
import tempfile
import time



def read_bvec(path: str) -> np.ndarray:
    """Load 3xN or Nx3 bvecs; return 3xN."""
    B = np.loadtxt(path, dtype=float)
    if B.ndim != 2:
        raise ValueError('bvec must be 2D (3xN or Nx3).')
    if B.shape[0] == 3:
        return B
    if B.shape[1] == 3:
        return B.T
    raise ValueError('bvec must have one dimension of size 3.')


def write_bvec(B: np.ndarray, path: str) -> None:
    """Write 3xN bvecs to file"""
    np.savetxt(path, B, fmt='%.6f')


def get_dir3(img: ants.ANTsImage) -> np.ndarray:
    """Return 3x3 ITK direction cosine matrix from ANTsPy image."""
    D = np.array(img.direction, dtype=float).reshape(3, 3)
    return D


def normalize_bvecs(B: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(B, axis=0)
    norms[norms == 0.0] = 1.0  # keep zero dirs at zero
    return B / norms


def rotation_angle_deg_between_dirs(D_ref: np.ndarray, D_mov: np.ndarray) -> float:
    """
    Compute rotation angle between two direction frames using trace formula.
    R = D_mov @ D_ref.T
    angle = arccos((trace(R)-1)/2)
    """
    Rm = D_mov @ D_ref.T
    val = (np.trace(Rm) - 1.0) / 2.0
    # numeric clamp
    val = np.clip(val, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def rebase_bvecs(B_in: np.ndarray, D_ref: np.ndarray, D_mov: np.ndarray) -> np.ndarray:
    """
    Rebases the bvecs from the moving to the reference space. Bvecs are in a modified index space following
    FSL / BIDS conventions.
    Steps:
    1) If det(D_mov) > 0: flip x of input bvecs
    2) voxel->physical: D_mov @ B
    3) physical->ref-voxel: D_ref.T @ (...)
    4) If det(D_ref) > 0: flip x of result
    5) Normalize columns
    """
    if B_in.shape[0] != 3:
        raise ValueError('bvecs must be 3xN.')
    B = B_in.copy()

    if np.linalg.det(D_mov) > 0:
        B[0, :] *= -1.0

    B_phys = D_mov @ B
    B_ref_vox = D_ref.T @ B_phys

    if np.linalg.det(D_ref) > 0:
        B_ref_vox[0, :] *= -1.0

    return normalize_bvecs(B_ref_vox)


def resample_dwi(ref_dwi: ants.ANTsImage, moving_dwi: ants.ANTsImage) -> ants.ANTsImage:
    """
    Resample the moving 4D DWI to the reference 4D DWI space using ANTsPy.apply_transforms.

    Both images must be 4D, ie raw dwi images.

    Returns the resampled 4D DWI

    """
    if len(ref_dwi.shape) != 4 or len(moving_dwi.shape) != 4:
        raise ValueError('Both images must be 4D.')

    # check that spacing is equal or very close to equal
    if not np.allclose(ref_dwi.spacing, moving_dwi.spacing, atol=1e-5):
        raise ValueError('Resampling not supported for differing resolutions.')

    # Check grid size - but allow different number of volumes
    if ref_dwi.shape[0:3] != moving_dwi.shape[0:3]:
        raise ValueError('Resampling not supported for differing grid sizes.')

    # Get the first volume of the reference as 3D target
    ref_3d = ref_dwi.slice_image(axis=3, idx=0, collapse_strategy=0)

    # ANTsPy unfortunately does not allow Identity as a string, so make a matrix
    identity = ants.create_ants_transform(precision='float', dimension=3)

    tmp_identity_file = tempfile.NamedTemporaryFile(suffix='.mat', delete=True).name

    ants.write_transform(identity, tmp_identity_file)

    output_dwi = ants.apply_transforms(fixed=ref_3d, moving=moving_dwi, transformlist=[tmp_identity_file], imagetype=3,
                                       interpolator='lanczosWindowedSinc', singleprecision=True)

    return output_dwi

# ---------- main workflow ----------
def main(reference_image_path: str, moving_image_path: str, output_prefix: str) -> None:
    # Inputs
    ref4d = ants.image_read(reference_image_path)
    mov4d = ants.image_read(moving_image_path)

    moving_dirname = os.path.dirname(moving_image_path)
    moving_basename_prefix = os.path.basename(moving_image_path).replace('.nii.gz', '').replace('.nii', '')

    moving_image_prefix = os.path.join(moving_dirname, moving_basename_prefix)

    # Directions
    D_ref = get_dir3(ants.slice_image(ref4d, axis=3, idx=0) if len(ref4d.shape) == 4 else ref4d)
    D_mov = get_dir3(ants.slice_image(mov4d, axis=3, idx=0) if len(mov4d.shape) == 4 else mov4d)

    # Rotation summary
    angle_deg = rotation_angle_deg_between_dirs(D_ref, D_mov)
    print(f'Rotation angle (moving -> reference) [deg]: {angle_deg:.3f}')

    # Translation summary
    translation = np.array(ref4d.origin[:3]) - np.array(mov4d.origin[:3])
    print(f'Translation vector (moving -> reference origin) [mm]: {translation}')

    # Read bvecs
    bvec_in = moving_image_prefix + '.bvec'
    if not os.path.exists(bvec_in):
        raise FileNotFoundError(f'Missing required bvec: {bvec_in}')
    B = read_bvec(bvec_in)

    if angle_deg < 0.1 and np.linalg.norm(translation) < 0.1:
        print(f'Rotation and translation below thresholds; copying inputs to outputs.')

        # Copy image
        shutil.copyfile(moving_image_path, f'{output_prefix}.nii.gz')
        shutil.copyfile(bvec_in, f'{output_prefix}.bvec')
        shutil.copyfile(moving_image_prefix + '.bval', f'{output_prefix}.bval')
        shutil.copyfile(moving_image_prefix + '.json', f'{output_prefix}.json')

        return

    print(f'Resampling DWI to reference space')
    out_img = resample_dwi(ref4d, mov4d)

    B_out = rebase_bvecs(B, D_ref, D_mov)

    # Write outputs
    print(f'Writing output')
    ants.image_write(out_img, f'{output_prefix}.nii.gz')
    write_bvec(B_out, f'{output_prefix}.bvec')
    shutil.copyfile(moving_image_prefix + '.bval', f'{output_prefix}.bval')
    print(f'Wrote: {output_prefix}.nii.gz')
    print(f'Wrote: {output_prefix}.bvec')

    # Record what we did in the JSON
    sidecar_in = moving_image_prefix + '.json'
    sidecar_out = output_prefix + '.json'
    if os.path.exists(sidecar_in):
        import json
        with open(sidecar_in, 'r') as f_in:
            metadata = json.load(f_in)
        metadata['ManualEdits'] = True
        metadata['ManualEditDescription'] = 'Resliced to reference DWI space'
        # Get date and local time
        metadata['ManualEditTimestamp'] = time.strftime('%Y-%m-%d %H:%M:%S%z', time.localtime())
        metadata['ManualEditOperator'] = os.getenv('USER', 'unknown')
        with open(sidecar_out, 'w') as f_out:
            json.dump(metadata, f_out, indent=4, sort_keys=True)
        print(f'Wrote: {sidecar_out}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Reslice DWI and rebase bvecs to reference space. If the rotation is less than 0.1'
                                'deg and the translation is less than 0.1mm, the data is simply copied.')
    p.add_argument('reference_image_path', help='Path to 4D DWI reference (ref space); uses first volume as ref3d')
    p.add_argument('moving_image_path', help='Path to 4D DWI moving image to be resampled')
    p.add_argument('output_prefix', help='Output prefix, will write prefix [.nii.gz , .json, .bvec]')
    args = p.parse_args()
    main(args.reference_image_path, args.moving_image_path, args.output_prefix)
