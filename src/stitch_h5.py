import h5py
import os


def stitch_h5_groups(input_folder, output_path):
    """
    Read all .h5 files in input_folder (sorted by name).  
    Each file must have this structure:
      /<episode_idx>/
          obs
          perturbed
          magnitudes
          indices
          properties

    We copy every episode group in order into output_path, renumbering
    episode groups sequentially from 0.
    """
    # 1) discover and sort input files
    files = sorted(f for f in os.listdir(input_folder) if f.endswith(".h5"))
    if not files:
        raise RuntimeError(f"No .h5 files in {input_folder!r}")

    # 2) open output once
    with h5py.File(output_path, "w") as fout:
        next_episode = 0

        for fname in files:
            in_path = os.path.join(input_folder, fname)
            print(f"Processing {fname} …", flush=True)

            with h5py.File(in_path, "r") as fin:
                # iterate each episode group in numeric order
                for grp_name in sorted(fin.keys(), key=lambda x: int(x)):
                    src_grp = fin[grp_name]
                    # create a new group under the next available index
                    dst_grp = fout.create_group(str(next_episode))

                    # copy each dataset inside this episode group
                    for ds_name, ds in src_grp.items():
                        # ds is a Dataset, so ds[...] reads it into memory,
                        # which is OK because these are relatively small per‐episode arrays.
                        data = ds[...]
                        dst_grp.create_dataset(
                            ds_name,
                            data=data,
                            compression="gzip",
                            compression_opts=4
                        )

                    next_episode += 1

            print(f"  done, total episodes so far = {next_episode}")

    print(f"\nAll files stitched into {output_path!r} ({next_episode} episodes).")


def print_h5_shapes(file_path):
    """
    Opens an HDF5 file and prints the shape of each dataset stored in it.

    Args:
        file_path (str): Path to the .h5 file.
    """
    with h5py.File(file_path, 'r') as f:
        print(f"Contents of '{file_path}':")
        def visit(name, node):
            if isinstance(node, h5py.Dataset):
                print(f"{name}: shape {node.shape}, dtype {node.dtype}")
        f.visititems(visit)

stitch_h5_groups("data/observations/separated", "data/observations/separated/stitched.h5")
print("done")