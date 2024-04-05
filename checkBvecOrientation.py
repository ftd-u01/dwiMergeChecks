#!/usr/bin/env python

import argparse
import numpy as np
import sys

def calculate_angle(v1, v2):
    """
    Calculate the angle in degrees between two vectors.
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return np.degrees(angle)

def compare_bvecs(ref_bvec_file, ref_bval_file, *bvec_files):
    """
    Compare each bvec in the given files with the reference bvec file.
    Prints the file name followed by the angles between corresponding bvecs in degrees.
    """
    ref_bvecs = np.loadtxt(ref_bvec_file)
    ref_bvals = np.loadtxt(ref_bval_file)
    # read list of bvec files into a list of numpy arrays
    input_bvecs = [np.loadtxt(bvec_file) for bvec_file in bvec_files]

    for idx in range(len(input_bvecs)):
        if input_bvecs[idx].shape != ref_bvecs.shape:
            print(f"Error: {bvec_files[idx]} does not have the same number of bvecs as the reference.")
            sys.exit(1)

    # Print header row
    header = "Index\tref_bval"

    print("bvec files:")
    for idx,fn in enumerate(bvec_files):
        print(f"{idx}\t{fn}")
        header = header + f"\tbvec_file{idx}"

    print(header + "\n")

    # Iterate over each of the reference bvecs, print line by line
    for vec_index in range(ref_bvecs.shape[1]):
        line = [f"{vec_index}", f"{ref_bvals[vec_index]}"]
        for b_index in range(len(input_bvecs)):
            angle = calculate_angle(ref_bvecs[:,vec_index], input_bvecs[b_index][:,vec_index])
            # print angle to three decimal places
            line.append(f"{angle:.3f}")
        print("\t".join(line))

def main():
    parser = argparse.ArgumentParser(description="Compare bvec files against a reference bvec file.")
    parser.add_argument("ref_bvec_file", help="The reference bvec file.")
    parser.add_argument("ref_bval_file", help="The reference bval file")
    parser.add_argument("bvec_files", nargs="+", help="One or more bvec files to compare against the reference.")

    args = parser.parse_args()

    compare_bvecs(args.ref_bvec_file, args.ref_bval_file, *args.bvec_files)


if __name__ == "__main__":
    main()
