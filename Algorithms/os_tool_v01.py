import os

def list_files_with_extension(folder_path, extension):
  """
  Lists all files with a given extension in the specified folder.

  Args:
    folder_path (str): The path to the folder.
    extension (str): The file extension (e.g., '.png', '.jpg').

  Returns:
    list: A list of file names with the specified extension.
  """
  if not os.path.isdir(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    return []

  file_list = []
  for file_name in os.listdir(folder_path):
    if file_name.endswith(extension):
      file_list.append(file_name)
  return file_list

# Example usage:
# folder = '.' # Current directory
# png_files = list_files_with_extension(folder, '.png')
# print(png_files)
