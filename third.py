import pandas as pd
import numpy as np
data=[]
n = int(input("Enter no of persons :"))
for i in range(n):
    print(f"\nEnter details of Student:")
    n=input("Name:")
    a=int(input("Age:"))
    m=int(input("Marks:"))
    at=int(input("Attendance:"))
    data.append({
        "Name": n,"Age": a,"Marks": m,"Attendance": at
    })
df = pd.DataFrame(data)
print(sum(data['Marks']))
for i in data['Marks']:
    if i >= 80:
        print(i)
