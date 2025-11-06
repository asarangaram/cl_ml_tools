from pathlib import Path

# specify your folder
folder = Path("./logs")

# iterate over all .log files
for log_file in folder.glob("*.log"):
    new_name = log_file.with_name(log_file.name + ".preserve")
    print(f"Renaming {log_file.name} → {new_name.name}")
    log_file.rename(new_name)
