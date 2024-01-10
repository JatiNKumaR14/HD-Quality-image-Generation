import os 

#set the path to the respective directory 
path = os.chdir('Resized')

i = 1 #counter
index = 1
for file in os.listdir(path):
    new_file_name = str(index).zfill(6)+'.jpeg' #change the format if required  os.rename(filename, str(index).zfill(7)+'.png')
    os.rename(file,new_file_name)
    print("Done %d" %i )
    index = index + 1
    i = i + 1
    


