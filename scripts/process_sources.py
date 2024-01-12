import os
import yaml


def check_source(directory):
    source_list_path = os.path.join(directory, "source_list.yml")

    if os.path.exists(source_list_path):
        with open(source_list_path, "r") as file:
            try:
                source_list = yaml.safe_load(file)
                if "source" in source_list:
                    print(f"Source found in {directory}: {source_list['source']}")
                else:
                    print(f"No source found in {directory}")
            except yaml.YAMLError as e:
                print(f"Error loading YAML file in {directory}: {e}")
    else:
        print(f"No source_list.yml file found in {directory}")


def process_directories(root_directory):
    for root, dirs, files in os.walk(root_directory):
        check_source(root)


if __name__ == "__main__":
    directories_to_check = ["intro", "tours", "tips"]

    for directory in directories_to_check:
        process_directories(directory)
