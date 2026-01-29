import csv
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ---------------- CONFIG ----------------
REPORT_DIR = "reports"

# Auto-pick latest CSV
csv_files = [
    f for f in os.listdir(REPORT_DIR)
    if f.startswith("total_lazy_time_") and f.endswith(".csv")
]

if not csv_files:
    raise FileNotFoundError("No lazy report CSV found")

csv_files.sort(reverse=True)
CSV_FILE = os.path.join(REPORT_DIR, csv_files[0])

DATE = csv_files[0].replace("total_lazy_time_", "").replace(".csv", "")
PDF_FILE = f"reports/total_lazy_time_{DATE}.pdf"

EMP_MASTER = "employees/employee_master.csv"



# ---------------- LOAD EMPLOYEE MASTER ----------------
emp_info = {}

with open(EMP_MASTER, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        emp_info[row["Employee"]] = {
        "id": row["Employee ID"],
        "dept": row["Department"]
        }


# ---------------- CREATE PDF ----------------
c = canvas.Canvas(PDF_FILE, pagesize=A4)
width, height = A4

# Title
c.setFont("Helvetica-Bold", 16)
c.drawCentredString(width / 2, height - 50, "Employee Lazy Time Report")

# Date
c.setFont("Helvetica", 11)
c.drawCentredString(width / 2, height - 75, f"Date: {DATE}")

# Table Header
y = height - 120
c.setFont("Helvetica-Bold", 11)

c.drawString(40, y, "Employee ID")
c.drawString(120, y, "Employee Name")
c.drawString(260, y, "Department")
c.drawString(420, y, "Lazy Time ")

c.line(30, y - 5, width - 30, y - 5)
y -= 30

# Table Rows
c.setFont("Helvetica", 11)

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)

    for row in reader:
        emp_id, name, dept, minutes = row


        c.drawString(40, y, emp_id)
        c.drawString(120, y, name)
        c.drawString(260, y, dept)
        c.drawString(440, y, f"{float(minutes):.2f}")

        y -= 22
        if y < 80:
            c.showPage()
            y = height - 80

# Footer
c.setFont("Helvetica-Oblique", 9)
c.drawCentredString(
    width / 2,
    40,
    "Generated automatically by Employee Gesture Tracking System"
)

c.save()

print(f"[PDF GENERATED] {PDF_FILE}")
