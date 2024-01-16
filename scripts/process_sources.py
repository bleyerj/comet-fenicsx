import os
import yaml
import subprocess
import zipfile
import fileinput


def check_source(directory):
    source_list_path = os.path.join(directory, "source_files.yml")

    if os.path.exists(source_list_path):
        with open(source_list_path, "r") as file:
            try:
                source_list = yaml.safe_load(file)
                source_zip = source_list.get("zip", None)
                convert_and_zip(directory, source_zip)
            except yaml.YAMLError as e:
                print(f"Error loading YAML file in {directory}: {e}")
    else:
        print(f"No source_files.yml file found in {directory}")
        convert_and_zip(directory, None)


def convert_and_zip(directory, source_zip=None):
    demo = os.path.split(directory)[-1]
    md_files = [f for f in os.listdir(directory) if f.endswith(".md")]

    if md_files:
        md_file = md_files[0]
        nb_file_name = demo + ".ipynb"
        py_file_name = demo + ".py"

        replace_download_placeholder(os.path.join(directory, md_file), demo, source_zip)

        # Use Jupytext to convert .md to .ipynb
        subprocess.run(
            [
                "jupytext",
                "--update",
                "--to",
                "notebook",
                os.path.join(directory, md_file),
            ]
        )
        # Use Black to format myst
        subprocess.run(
            [
                "jupytext",
                "--to",
                "myst",
                "--pipe",
                "black",
                os.path.join(directory, nb_file_name),
            ]
        )

        # Use Jupytext to convert .md to .py
        subprocess.run(["jupytext", "--to", "py", os.path.join(directory, md_file)])
        # Create a zip file with the notebook and source zip file
        with zipfile.ZipFile(os.path.join(directory, f"{demo}.zip"), "w") as zip_file:
            zip_file.write(os.path.join(directory, nb_file_name), nb_file_name)
            zip_file.write(os.path.join(directory, py_file_name), py_file_name)
            if source_zip is not None:
                if isinstance(source_zip, list):
                    for source in source_zip:
                        print("File to zip:", source)
                        zip_file.write(os.path.join(directory, source), source)
                else:
                    print("File to zip:", source_zip)
                    zip_file.write(os.path.join(directory, source_zip), source_zip)
            else:
                print("No other file to zip.")

        print(f"Conversion and zip completed for {directory}")
    else:
        print(f"No .md file found in {directory}")


def replace_download_placeholder(md_file_path, demo, source_zip=None):
    replacement_text = f"```{{admonition}} Download sources\n:class: download\n\n* {{Download}}`Python script<./{demo}.py>`\n* {{Download}}`Jupyter notebook<./{demo}.ipynb>`"
    if source_zip:
        replacement_text += f"\n* {{Download}}`Complete sources files<./{demo}.zip>`"
    replacement_text += "\n```"

    with fileinput.FileInput(md_file_path, inplace=True) as file:
        for line in file:
            print(line.replace("__DOWNLOAD_PLACEHOLDER__", replacement_text), end="")


def process_directories(root_directory, exclude_patterns):
    for root, dirs, files in os.walk(root_directory):
        if os.path.split(root)[-1] not in exclude_patterns and all(
            [os.path.split(f)[-1] not in exclude_patterns for f in files]
        ):
            check_source(root)


if __name__ == "__main__":
    directories_to_check = ["intro", "tours", "tips"]

    exclude_patterns = ["__pycache__", "contents.md"]

    for directory in directories_to_check:
        process_directories(directory, exclude_patterns)
