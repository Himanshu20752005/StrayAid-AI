import os

# Path to dataset
base_path = "."

# Loop through each class folder
for class_name in os.listdir(base_path):
    class_path = os.path.join(base_path, class_name)

    if os.path.isdir(class_path):
        count = 1

        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)

            # Skip non-image files
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            new_name = f"{class_name}_{count}.jpg"
            new_path = os.path.join(class_path, new_name)

            os.rename(file_path, new_path)
            count += 1

print("✅ Renaming complete!")