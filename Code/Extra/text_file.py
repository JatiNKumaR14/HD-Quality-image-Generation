f= open("100k.txt","w+")
for i in range(4000):
     f.write("%06i.jpeg\n" % (i+1))
f.close() 
