import os
print("TEST SCRIPT RUNNING")

print("Current working directory:", os.getcwd())

open("test_output.txt", "w").write("hello")
print("File written")
