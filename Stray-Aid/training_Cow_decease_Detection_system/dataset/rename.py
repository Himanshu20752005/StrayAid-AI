import os

# 👉 CHANGE THIS PATH (IMPORTANT)
BASE_DIR = r"C:\Users\Shailendra\Downloads\Final_Proj\Stray-Aid\training_Cow_decease_Detection_system\dataset"

def clean_folder_name(name):
    return name.replace("-", "_").replace(" ", "_").lower()

def rename_files():
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        # Clean folder name
        new_folder_name = clean_folder_name(folder)
        new_folder_path = os.path.join(BASE_DIR, new_folder_name)

        if folder != new_folder_name:
            os.rename(folder_path, new_folder_path)
            print(f"Renamed folder: {folder} → {new_folder_name}")
        else:
            new_folder_path = folder_path

        # Rename images inside folder
        count = 1
        for file in os.listdir(new_folder_path):
            file_path = os.path.join(new_folder_path, file)

            if not os.path.isfile(file_path):
                continue

            ext = file.split('.')[-1]
            new_name = f"{new_folder_name}_{count}.{ext}"
            new_path = os.path.join(new_folder_path, new_name)

            os.rename(file_path, new_path)
            count += 1

        print(f"Processed: {new_folder_name}")

if __name__ == "__main__":
    rename_files()
    print("\n✅ Renaming completed!")