import os

BASE_DIR = r"C:\Users\Shailendra\Downloads\Final_Proj\Stray-Aid\training_Cat_decease_Detection_system\dataset"

def clean_name(name):
    return name.lower().replace(" ", "_")

def rename_all():
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        new_folder_name = clean_name(folder)
        new_folder_path = os.path.join(BASE_DIR, new_folder_name)

        if folder != new_folder_name:
            if not os.path.exists(new_folder_path):
                os.rename(folder_path, new_folder_path)
            folder_path = new_folder_path

        # Rename images safely
        count = 1
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(file)[1]
            new_name = f"{new_folder_name}_{count}{ext}"
            new_path = os.path.join(folder_path, new_name)

            # 🔥 Skip if already exists
            if os.path.exists(new_path):
                count += 1
                continue

            os.rename(file_path, new_path)
            count += 1

        print(f"Processed: {new_folder_name}")

if __name__ == "__main__":
    rename_all()
    print("\n✅ Safe renaming done!")