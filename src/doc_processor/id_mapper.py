import csv
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

input_file = Path(os.getenv("PATH_DATA")) / "raw" / "master_ledger_2025.csv"
output_file = Path(os.getenv("PATH_DATA")) / "interleaved" / "accountTOscenario.csv"

seen = set()

with open(input_file, "r", encoding="utf-8-sig", newline="") as infile, \
     open(output_file, "w", encoding="utf-8", newline="") as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) < 3:
            continue

        first_column = row[0].split("-", 2)

        if len(first_column) >= 3:
            extracted = first_column[1].strip()
        else:
            continue

        data = (extracted, row[2])

        if data not in seen:
            seen.add(data)
            writer.writerow(data)

print(f"Done. Saved unique data to {output_file}")