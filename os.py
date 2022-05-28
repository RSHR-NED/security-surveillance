import os
s = os.listdir("./identified_faces")[0]
result = s.split(".")[0]
print(result)