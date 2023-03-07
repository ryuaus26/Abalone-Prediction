import pandas as pd

# Read the abalone.csv file
df = pd.read_csv("./Data/abalone.csv")

# Get the "Type" column
type_column = df["Type"]

# Write the "Type" column to a new file
# with open("new_file.csv", mode="w") as file:
#     file.write("Type\n")  # Write the header row
#     for value in type_column:
#         file.write(str(value) + "\n")  # Write each value in a new row

X = df[["LongestShell","Diameter","Height","WholeWeight","VisceraWeight","ShellWeight","Rings"]]
        
X.to_csv("X.csv",index=False)

