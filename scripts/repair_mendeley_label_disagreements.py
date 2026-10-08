"""Repair four confirmed merged2 Mendeley class-ID mismatches, preserving boxes.

Raw labels use class 1 for river_vegetation (mapped to hyacinth). The four
expected class flips are guarded by exact split/stem/box index and geometry;
the raw annotations are never modified.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = PROC / "donors" / "mendeley" / "extracted" / "Floating Waste and River Vegetation Dataset"
FIXES = {
    ("val", "men_FOTO_0140"): (11, "FOTO_0140"),
    ("val", "men_IMG_6180_frame_00000"): (5, "IMG_6180_frame_00000"),
    ("test", "men_FOTO_0011"): (1, "FOTO_0011"),
    ("test", "men_IMG_6190_frame_00005"): (0, "IMG_6190_frame_00005"),
}


def main():
    for (split, stem), (index, raw_stem) in FIXES.items():
        raw_path = RAW / split / "labels" / f"{raw_stem}.txt"
        out_path = PROC / "merged2" / "labels" / split / f"{stem}.txt"
        raw_rows = [line.split() for line in raw_path.read_text().splitlines()]
        out_rows = [line.split() for line in out_path.read_text().splitlines()]
        if len(raw_rows) != len(out_rows) or len(raw_rows) <= index:
            raise RuntimeError(f"unexpected row count for {stem}")
        if raw_rows[index][0] != "1" or out_rows[index][0] != "0":
            raise RuntimeError(f"unexpected class IDs at {stem} box {index}; refusing to edit")
        if [round(float(value), 6) for value in raw_rows[index][1:]] != [
                round(float(value), 6) for value in out_rows[index][1:]]:
            raise RuntimeError(f"geometry differs at {stem} box {index}; refusing to edit")
        out_rows[index][0] = "1"
        out_path.write_text("\n".join(" ".join(row) for row in out_rows) + "\n", encoding="utf-8")
        print(f"fixed {split}/{stem} box {index}: litter -> hyacinth")


if __name__ == "__main__":
    main()
