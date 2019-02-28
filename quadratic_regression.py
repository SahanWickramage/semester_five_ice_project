import numpy as np

# Solving following system of linear equation
# 1x + 1y = 35
# 2x + 4y = 94

a = np.array([[1, 1],[2,4]])
b = np.array([35, 94])

#don't need to inverse matrix a
print(np.linalg.solve(a,b))

#quadratic regression
#y = a(x^2) + b(x) + c
#(sigma(x^4))a + (sigma(x^3))b + (sigma(x^2))c = (sigma((x^2)*y))
#(sigma(x^3))a + (sigma(x^2))b + (sigma(x^1))c = (sigma((x^1)*y))
#(sigma(x^2))a + (sigma(x^1))b + (sigma(x^0))c = (sigma((x^0)*y))
array_a = np.zeros((3,3))
array_b = np.zeros((3,1))

temporary_variable = 0
for var in range(1,3):
    temporary_variable += var**4

array_a[0][0] = temporary_variable

temporary_variable = 0
for var in range(1,3):
    temporary_variable += var**3

array_a[0][1] = temporary_variable
array_a[1][0] = temporary_variable

temporary_variable = 0
for var in range(1,3):
    temporary_variable += var**2

array_a[0][2] = temporary_variable
array_a[1][1] = temporary_variable
array_a[2][0] = temporary_variable

temporary_variable = 0
for var in range(1,3):
    temporary_variable += var

array_a[1][2] = temporary_variable
array_a[2][1] = temporary_variable

temporary_variable = 0
for var in range(1,3):
    temporary_variable += 1

array_a[2][2] = temporary_variable

print(array_a)


