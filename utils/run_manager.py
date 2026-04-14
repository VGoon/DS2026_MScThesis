import os

def get_next_run_path(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    # find existing Run_X folders
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Run_")
    ]

    # extract numbers
    run_numbers = []
    for name in existing:
        try:
            num = int(name.split("_")[1])
            run_numbers.append(num)
        except:
            continue

    next_run = max(run_numbers, default=0) + 1

    run_path = os.path.join(base_dir, f"Run_{next_run}")
    os.makedirs(run_path)

    return run_path, f"Run_{next_run}/"