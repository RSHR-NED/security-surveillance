# detect_&_recognize is the package


### Advantages
# Dont need to save the images of the identified databases
# for compare faces. only on the bases of the generated 260 int values
# Hog is uses due to which slight tilt face can be detect
# 

### Research Required

# which resizing technique is best for the upsampling or down-sampling
# Resizing in the way that small face can detect
# How to set image id of the running frame
# How to optimize the algorithm 
# path specify 


# Information
""" No. This is actually the norm in Python. In fact, Python only loads the module when it is imported in the first file and all subsequent files simply set the name to refer to the already loaded module. While it is possible to override this behavior so that each file has its own copy of the module, it is generally not recommended.
To see this in action, import the same module in two different files. Set a variable in the module from one of the files and have the other file retrieve that variable from the module.
Example using the interpreter: """

import os 
import os as p
p.g = 5
print(p.g,os.g)



