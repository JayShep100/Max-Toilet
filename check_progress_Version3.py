import csv

# Count Stage 1 results
with open("shortlist.csv") as f:
    rows = list(csv.DictReader(f))
dog_clips = [r for r in rows if r["dog_detected"].strip().upper() == "TRUE"]
no_dog = [r for r in rows if r["dog_detected"].strip().upper() == "FALSE"]
print(f"Stage 1: {len(rows)} total, {len(dog_clips)} with dog, {len(no_dog)} without dog")

# Count Stage 2 results
try:
    with open("pad_clips1.csv") as f:
        pad_rows = list(csv.DictReader(f))
    on_pad = [r for r in pad_rows if r["on_pad"].strip() == "True"]
    off_pad = [r for r in pad_rows if r["on_pad"].strip() == "False"]
    print(f"Stage 2: {len(pad_rows)} processed, {len(on_pad)} ON pad, {len(off_pad)} OFF pad")
    print(f"Stage 2 MISSING: {len(dog_clips) - len(pad_rows)} dog clips not yet processed")
except FileNotFoundError:
    print("No pad_clips file found — Stage 2 not started")