import os
import json

# ======================
# CONFIG
BASE_DIR = r"E:\Dataset CVAT"


# ======================
# EXTRACT TIMESTAMP
# ======================
def get_timestamp(name):
    try:
        return int(name.split("_")[-1].split(".")[0])
    except:
        return None


# ======================
# MATCH RGB → DEPTH
# ======================
def match_depth(rgb_name, depth_files):
    ts_rgb = get_timestamp(rgb_name)

    best = None
    best_diff = float("inf")

    for d in depth_files:
        ts_d = get_timestamp(d)
        if ts_d is None:
            continue

        diff = abs(ts_rgb - ts_d)

        if diff < best_diff:
            best_diff = diff
            best = d

    return best


# ======================
# BUILD SAMPLES PER TASK ✅
# ======================
def build_samples():
    samples = []

    print("\n📌 Assign weight PER TASK (1 task = 1 cluster)\n")

    for task in os.listdir(BASE_DIR):

        task_path = os.path.join(BASE_DIR, task)

        if not os.path.isdir(task_path):
            continue

        img_dir = os.path.join(task_path, "images", "train")
        depth_dir = os.path.join(task_path, "depth")

        if not os.path.exists(img_dir) or not os.path.exists(depth_dir):
            continue

        rgb_files = [f for f in os.listdir(img_dir) if "Color" in f]
        depth_files = [f for f in os.listdir(depth_dir) if "Depth" in f]

        if len(rgb_files) < 2:
            continue

        print(f"\n🟢 TASK: {task}")
        print(f"Views: {len(rgb_files)}")
        print("Example:", os.path.join(img_dir, rgb_files[0]))

        user_input = input("Enter weight (g) or 'skip': ")

        if user_input.lower() == "skip":
            continue

        try:
            weight = float(user_input)
        except:
            print("❌ Invalid, skip")
            continue

        rgb_list = []
        depth_list = []

        for rgb_name in rgb_files:
            depth_name = match_depth(rgb_name, depth_files)

            if depth_name is None:
                continue

            rgb_path = os.path.join(img_dir, rgb_name)
            depth_path = os.path.join(depth_dir, depth_name)

            if not os.path.exists(depth_path):
                continue

            rgb_list.append(rgb_path)
            depth_list.append(depth_path)

        if len(rgb_list) == 0:
            continue

        samples.append({
            "rgb": rgb_list,
            "depth": depth_list,
            "weight": weight
        })

    print(f"\n✅ TOTAL SAMPLES: {len(samples)}")
    return samples


# ======================
# MAIN
# ======================
if __name__ == "__main__":

    samples = build_samples()

    with open("samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\n✅ Saved samples.json")