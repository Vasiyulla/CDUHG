import os

def list_files_in_directory(directory=r'C:\Users\Dell\Documents\Mini Project 3rd year\CDUHG\Example'):
    try:
        files = os.listdir(directory)
        print("Files in directory:", directory)
        for file in files:
            print(file)
    except Exception as e:
        print(f"An error occurred: {e}")
