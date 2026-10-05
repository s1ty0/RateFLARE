# create_subsampled_dataset.py (Memory-Efficient + Final Ratio Check)
import os
import numpy as np

def main():
    original_root = "./myDataK"
    new_root = "./myDataK20"

    splits = {
        "train": 20000,
        "val": 2000,
        "test": 2000
    }

    for split, n_samples in splits.items():
        print(f"\nProcessing {split}...")
        label_path = os.path.join(original_root, split, "label_data.npy")

        # Step 1: Load labels (small)
        labels = np.load(label_path)
        N = len(labels)
        print(f"  Original size: {N}, positive ratio: {labels.mean():.4f}")

        if n_samples >= N:
            indices = np.arange(N)
        else:
            np.random.seed(42)
            indices = np.random.choice(N, size=n_samples, replace=False)

        # Step 2: Memory-mapped loading for large arrays
        lc_path = os.path.join(original_root, split, "lc_data.npy")
        mask_path = os.path.join(original_root, split, "mask_data.npy")

        lc_mmap = np.load(lc_path, mmap_mode='r')
        mask_mmap = np.load(mask_path, mmap_mode='r')

        # Step 3: Extract subset
        lc_sub = lc_mmap[indices]
        mask_sub = mask_mmap[indices]
        labels_sub = labels[indices]

        # Step 4: Save
        out_dir = os.path.join(new_root, split)
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "lc_data.npy"), lc_sub)
        np.save(os.path.join(out_dir, "mask_data.npy"), mask_sub)
        np.save(os.path.join(out_dir, "label_data.npy"), labels_sub)

        # ✅ Step 5: Verify final positive ratio
        final_positive_ratio = labels_sub.mean()
        print(f"  Subsampled size: {len(labels_sub)}")
        print(f"  Final positive ratio: {final_positive_ratio:.4f}")
        print(f"  Saved to {out_dir}")

    print("\n✅ Subsampling completed!")

    # Optional: Final summary
    print("\n📊 Final Dataset Statistics:")
    for split in splits:
        label_file = os.path.join(new_root, split, "label_data.npy")
        final_labels = np.load(label_file)
        ratio = final_labels.mean()
        print(f"  {split}: {len(final_labels)} samples, positive ratio = {ratio:.4f}")

if __name__ == "__main__":
    main()