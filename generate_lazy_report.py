import csv
from datetime import datetime
from collections import defaultdict
import os
import json


EMP_DIR = "employees"

employee_meta = {}  # name → {id, department}

for emp_name in os.listdir(EMP_DIR):
    info_path = os.path.join(EMP_DIR, emp_name, "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            data = json.load(f)
            employee_meta[emp_name] = {
                "id": data.get("employee_id", "NA"),
                "department": data.get("department", "NA")
            }



LOG_FILE = "employee_log.csv"
REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

today = datetime.now().strftime("%Y-%m-%d")
REPORT_FILE = os.path.join(REPORT_DIR, f"total_lazy_time_{today}.csv")

# employee -> total lazy seconds
lazy_seconds = defaultdict(float)

# employee -> lazy start timestamp
lazy_start = {}

with open(LOG_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        if len(row) != 4:
            continue  # safety

        ts, cam, employee, gesture = row
        time_obj = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")

        # LAZY START
        if employee == "Detecting..." or employee == "Unknown":
            continue

        if gesture == "Lazy":
            if employee not in lazy_start:
                lazy_start[employee] = time_obj


        # LAZY END
        elif gesture == "Perfect":
            if employee in lazy_start and employee not in ("Detecting...", "Unknown"):
                duration = (time_obj - lazy_start[employee]).total_seconds()
                lazy_seconds[employee] += duration
                del lazy_start[employee]





# HANDLE EMPLOYEES STILL LAZY AT END OF LOG
end_time = datetime.now()
for employee, start_time in lazy_start.items():
    duration = (end_time - start_time).total_seconds()
    lazy_seconds[employee] += duration

# WRITE REPORT
with open(REPORT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Employee ID","Employee Name","Department","Total Lazy Time (minutes)"])


    for employee, seconds in lazy_seconds.items():
        if employee in ("Detecting...", "Unknown"):
            continue

        minutes = round(seconds / 60, 2)

        emp_id = employee_meta.get(employee, {}).get("id", "NA")
        department = employee_meta.get(employee, {}).get("department", "NA")

        writer.writerow([emp_id,employee,department,minutes])




print(f"[OK] Total Lazy Time Report Generated → {REPORT_FILE}")
