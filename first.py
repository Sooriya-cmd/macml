import numpy as np


Al= []
R1=int(input("enter row:"))
C1=int(input("enter column:"))

print("Enter elements for 2x2 matrix:")

for i in range(R1):
    row = []
    for j in range(C1):
        val = int(input(f"Enter element at position [{i}][{j}]: "))
        row.append(val)
    Al.append(row)

print("Matrix is:")
for row in Al:
    print(row)


    
Bl= []
R2=int(input("enter row:"))
C2=int(input("enter column:"))

print("Enter elements for 2x2 matrix:")

for i in range(R2):
    row = []
    for j in range(C2):
        val = int(input(f"Enter element at position [{i}][{j}]: "))
        row.append(val)
    Bl.append(row)

print("Matrix is:")
for row in Bl:
    print(row)
    
A =  np.array(Al)
B=np.array(Bl)
print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

addition = A + B
subtraction = A - B
multiplication = A * B          
matrix_mult = np.dot(A, B)      

print("\nMatrix Addition (A + B):")
print(addition)

print("\nMatrix Subtraction (A - B):")
print(subtraction)

print("\nElement-wise Multiplication (A * B):")
print(multiplication)

print("\nMatrix Multiplication (A . B):")
print(matrix_mult)

try:
    inverse_A = np.linalg.inv(A)
    print("\nInverse of Matrix A:")
    print(inverse_A)
except np.linalg.LinAlgError:
    print("\nMatrix A is singular, inverse does not exist.")

try:
    inverse_B = np.linalg.inv(B)
    print("\nInverse of Matrix B:")
    print(inverse_B)
except np.linalg.LinAlgError:
    print("\nMatrix B is singular, inverse does not exist.")

transpose_A = np.transpose(A)
transpose_B = np.transpose(B)

print("\nTranspose of Matrix A:")
print(transpose_A)

print("\nTranspose of Matrix B:")
print(transpose_B)
