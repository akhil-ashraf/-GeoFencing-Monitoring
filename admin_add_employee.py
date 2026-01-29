import os
import json
import shutil

EMP_DIR = "employees"
os.makedirs(EMP_DIR, exist_ok=True)

print("\n==============================")
print(" ADMIN – ADD NEW EMPLOYEE ")
print("==============================\n")

emp_id = input("Enter Employee ID (ex: EMP001): ").strip().upper()
emp_name = input("Enter Employee Name: ").strip()
emp_dept = input("Enter Employee Department: ").strip()
photo_path = input("Enter Employee Photo Path: ").strip()

if not os.path.exists(photo_path):
    print("❌ Photo file not found")
    exit()

# Create employee folder using EMPLOYEE NAME
emp_folder = os.path.join(EMP_DIR, emp_name)
os.makedirs(emp_folder, exist_ok=True)

# Copy photo
photo_dest = os.path.join(emp_folder, "photo.jpg")
shutil.copy(photo_path, photo_dest)

# Save employee info
info = {
    "employee_id": emp_id,
    "employee_name": emp_name,
    "department": emp_dept,
    "photo": "photo.jpg"
}

with open(os.path.join(emp_folder, "info.json"), "w") as f:
    json.dump(info, f, indent=4)

print("\n✅ Employee Added Successfully")
print(f"ID         : {emp_id}")
print(f"Name       : {emp_name}")
print(f"Department : {emp_dept}")
print("Photo saved\n")
