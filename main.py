import streamlit as st
import numpy as np
from pathlib import Path
import base64
#Main Function
st.set_page_config(
     page_title='Numpy-Cheat-Sheet',
     page_icon='title.png',
     layout="wide",
     initial_sidebar_state="expanded",
)
#Image function Start
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
#Image function End
#Sidebar Start
def cs_sidebar():
    st.sidebar.markdown('''[<img src='data:image/png;base64,{}' class='img-fluid' width=300 height=150>](https://numpy.org/doc/stable/user/whatisnumpy.html)'''.format(img_to_bytes("side.png")), unsafe_allow_html=True)
    st.sidebar.title('Numpy cheat sheet')
    st.sidebar.markdown('''<small>Developed by <b>Jayanthan Senthilkumar<b></small>''', unsafe_allow_html=True)
    st.sidebar.markdown('__Install and import__')
    st.sidebar.code('$ pip install numpy')
    st.sidebar.code('''
# Import convention
>>> import numpy as np
''')
#Sidebar End
def cs_body():
    st.title("Numerical Python")
    print()
    a,b, = st.columns(2)
#A - Column Starts
    a.header("What is Numerical Python ???")
    a.markdown("The NumPy library is the core library for scientific computing inPython."
             " It provides a high-performance multidimensional array"
             "object, and tools for working with these arrays")
    a.markdown("---")
#Initial Placeholder Starts 
    a.title("Initial Placeholders")
    a.subheader("Create an array of zeros")
    a.code("np.zeros((4,4))")
    a.subheader("Create an array of ones")
    a.code("np.ones((2,3,4),dtype=np.float)")
    a.subheader("Create an array of evenly")
    a.code("np.arange(10,25,5)")
    a.subheader("Create an array of evenly spaced values")
    a.code("np.linspace(0,2,9) ")
    a.subheader("Create a constant array")
    a.code("np.full((2,2),7) ")
    a.subheader("Create a 2X2 identity matrix")
    a.code("np.eye(2) ")
    a.subheader("Create an array with random values")
    a.code("np.random.random((2,2)) ")
    a.subheader("Create an empty array")
    a.code("np.empty((3,2))")
    a.markdown("---")
#Initial Placeholder ends
#Data Types Starts
    a.title("Data Types")
    a.subheader("Signed 64-bit integer types")
    a.code("np.int64()")
    a.subheader("Standard double-precision floating point")
    a.code("np.float32()")
    a.subheader("Complex numbers represented by 128 floats")
    a.code("np.complex()")
    a.subheader("Boolean type storing TRUE and FALSE values")
    a.code("np.bool()")
    a.subheader("Python object type")
    a.code("np.object()")
    a.subheader("Fixed-length string type")
    a.code("np.string_() ")
    a.subheader("Fixed-length unicode type")
    a.code("np.unicode_()")
    a.markdown("---")
#Data types ends
#Array Attributes and Methods Starts
    a.title("Array Attributes and Methods")
    a.header("Here Consider an array :")
    a.code("a=[02,05,2005]")
    a.subheader("Array dimensions")
    a.code("a.shape()")
    a.subheader("Length of array")
    a.code("len(a)")
    a.subheader("Number of array dimensions")
    a.code("a.ndim()")
    a.subheader("Number of array elements")
    a.code("a.size()")
    a.subheader("Data type of array elements")
    a.code("a.dtype()")
    a.subheader("Name of data type")
    a.code("a.dtype.name()")
    a.subheader("Convert an array to a different type")
    a.code("a.astype(int)")
    a.markdown("---")
#Array Attributes and Methods Ends
#Comparision Starts
    a.title("Comparsion")
    cta="""
import numpy as np
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[1, 3, 3],
              [7, 5, 8]])
element_wise_comparison = a == b
print("Element-wise comparison:")
print(element_wise_comparison)

# Element-wise comparison with a scalar
element_wise_comparison_scalar = a < 2
print("\nElement-wise comparison with a scalar:")
print(element_wise_comparison_scalar)

# Array-wise comparison
array_wise_comparison = np.array_equal(a, b)
print("\nArray-wise comparison:")
print(array_wise_comparison)

"""
    a.code(cta)
    a.markdown("---")
#Comparision Ends
#Copying Array Starts
    a.title("Copying Arrays")
    a.subheader("Create a view of the array with the same data")
    a.code("a.view()")
    a.subheader("Create a copy of the array")
    a.code("np.copy(a)")
    a.subheader("Create a deep copy of the array")
    a.code("a.copy()")
    a.subheader("Example Program :")
    cpp="""
import numpy as np

# Define an array
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Create a view of the array with the same data
h = a.view()
print("View of the array with the same data:")
print(h)

# Create a copy of the array
copy_a = np.copy(a)
print("\nCopy of the array:")
print(copy_a)

# Alternatively, you can also use:
h = a.copy()
print("\nAnother way to create a copy of the array:")
print(h)
"""
    a.code(cpp)
    a.markdown("---")
#Copying Array Ends
#Sorted Array Starts
    a.title("Sorting Arrays")
    a.subheader("Sort an array")
    a.code("a.sort()")
    a.subheader("Sort the elements of an array's axis")
    a.code("a.sort(axis=0)")
    a.subheader("Example Program :")
    py="""
    import numpy as np

    # Define arrays
    a = np.array([[3, 2, 1],
                  [6, 5, 4]])

    c = np.array([[9, 8, 7],
                 [12, 11, 10]])

    # Sort array 'a'
    a.sort()
    print("Sorted array 'a':")
    print(a)

    # Sort along the specified axis for array 'c'
    c.sort(axis=0)
    print("\nSorted array 'c' along axis 0:")
    print(c)"""
    a.code(py)
    a.markdown("---")
#Sorted Array Ends
#Array Manipulation Starts
    a.title("Array Manipulation")
    a.subheader("Transposing Array")
    tk="""
import numpy as np
i=np.transpose(b)"""
    a.code(tk)
    a.markdown("---")
    a.title("Changing Array Shape")

    a.code("b.ravel()")
    a.write("Flatten the array")

    a.code("g.reshape(3,-2)")
    a.write("Reshape, but don’t change data")
    a.subheader("Example Program :")
    skk="""
import numpy as np

# Define arrays
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

g = np.array([[1, 2],
              [3, 4],
              [5, 6]])

# Flatten the array 'b'
flattened_b = b.ravel()
print("Flattened array 'b':", flattened_b)

# Reshape the array 'g' to have 3 rows and as many columns as needed
reshaped_g = g.reshape(3, -2)
print("Reshaped array 'g' with 3 rows and as many columns as needed:")
print(reshaped_g)
"""
    a.code(skk)
    a.title("Adding/Removing Elements")
    a.markdown("---")
    a.code("np.concatenate((a,b),axis=0) ")
    a.write("Concatenate arrays")
    a.code("np.vstack((a,b))")
    a.write("Stack arrays vertically (row-wise)")
    a.code("np.r_[a,b] ")
    a.write("Stack arrays vertically (row-wise)")
    a.code("np.hstack((a,b)) ")
    a.write("Stack arrays horizontally (column-wise)")
    a.code("np.column_stack((a,b))")
    a.write("Create stacked column-wise arrays")
    a.title("Splitting Arrays")
    a.code("np.hsplit(a,3)")
    a.write("Split the array horizontally at the 3rd index")
    a.code("np.vsplit(b,3)")
    a.write("Split the array vertically at the 2nd index")
    srk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3])
b = np.array([[[1.5, 2., 1.],
               [4., 5., 6.]],
              [[3., 2., 3.],
               [4., 5., 6.]],
              [[7., 8., 9.],
               [10., 11., 12.]]])

# Split the array 'a' horizontally at the 3rd index
hsplit_a = np.hsplit(a, 3)
print("Split array 'a' horizontally at the 3rd index:")
print(hsplit_a)

# Split the array 'b' vertically at the 2nd index
vsplit_b = np.vsplit(b, 2)
print("\nSplit array 'b' vertically at the 2nd index:")
print(vsplit_b)
"""
    a.code(srk)
    a.markdown("---")
#Array Manipulation Ends
#A - Column Ends

#B - Column Starts
#Numpy Array Session Start
    b.title("Numpy Array")
    b.image('numpy.png',caption="Numerical Python Array")
    b.markdown("The Introduction for Numerical Python and its usecase is enough to know and It has a three types of an Array like")
    b.markdown("Let we come to see how thw arrays are to created and executed using numpy")
    b.markdown("*One Dimensional Array")
    b.markdown("*Two Dimensional Array")
    b.markdown("*Three Dimensional Array")
    b.markdown("---")
    b.title("Creating Arrays")
    b.latex("1 Dimensional Array")
    b.code("np.array([1,2,3])")
    b.latex("2 Dimensional Array")
    b.code("np.array([[1,3,2,9],[0,1,0,5]],dtype=float32)")
    b.latex("3 Dimensional Array")
    b.code("np.array([[[1,3],[2,5],[2,8],[2,9]]],dtype=int64)")
    b.markdown("---")
#Numpy Array Sesssion End
##Numpy I/O Starts
    b.title("Input and Output")
    b.subheader("Saving & Loading On Disk")
    b.code("np.save('my_array', a)")
    b.code("np.savez('array.npz', a, b)")
    b.code(" np.load('my_array.npy')")
    b.markdown("---")
    b.subheader("Saving & Loading Text Files")
    b.code("np.loadtxt('myfile.txt')")
    b.code("np.genfromtxt('my_file.csv', delimiter=',')")
    b.code("np.savetxt('myarray.txt', a, delimiter='')")
    b.markdown("---")    
#Numpy I/O Sesssion End
#Numpy Arithmetic Operations Session Starts
    b.title("Numpy Operators")
    b.subheader("Addition")
    adb="""
import numpy as np
arr1=np.array([[1,2,3],[4,5,6]])
arr2=np.array([[7,8,9],[3,2,1]])
print(np.add(arr1,arr2))
"""
    b.code(adb)
    b.markdown("---")
#Subtraction
    b.subheader("Subtraction")
    sub2 = """
    import numpy as np
    arr1=np.array([[13,29,31],[32,25,28]])
    arr2=np.array([[5,1,8],[12,2,4]])
    print(np.subtract(arr1,arr2))
    """
    b.code(sub2)
    b.markdown("---")
#Multiplication
    b.subheader("Multiplication")
    mult2 = """
     import numpy as np
     arr1=np.array([[1,2,3],[4,5,6]])
     arr2=np.array([[7,8,9],[3,2,1]])
     print(np.multiply(arr1,arr2))
     """
    b.code(mult2)
    b.markdown("---")
#Division
    b.subheader("Division")
    div2 = """
    import numpy as np
    arr1=np.array([[1,2,3],[4,5,6]])
    arr2=np.array([[7,8,9],[3,2,1]])
    result=np.divide(arr1,arr2)
    print(result)
    """
    b.code(div2)
    b.markdown("---")
    b.subheader("Dot Product")
    dp="""
    import numpy as np
    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6])
    dot_product = np.dot(array1, array2)
    print(dot_product)"""
    b.code(dp)
    b.markdown("---")
#Exponential
    b.subheader("Exponentiation")
    b.code("np.exp(a)")
#Square root
    b.subheader("Square Root")
    b.code("np.sqrt(a)")
#Trigonometry Functions
    b.subheader("Sines of an array")
    b.code("np.sin(a)")
    b.subheader("Cosine of an array")
    b.code("np.cos(a)")
    b.subheader("Element-wise natural logarithm")
    b.code("np.log(a)")
    b.markdown("---")
#Numpy Arithmetic Operations Session Ends
#Aggregate Functions Starts
    b.title("Aggregate Functions")
    b.subheader("Array wise sum")
    b.code("a.sum()")
    b.subheader("Array-wise minimum value")
    b.code("a.min()")
    b.subheader("Maximum value of an array row")
    b.code(" b.max(axis=0)")
    b.subheader("Cumulative sum of the elements")
    b.code("b.cumsum(axis=1)")
    b.subheader("Mean")
    b.code("a.mean()")
    b.subheader("Median")
    b.code("a.median()")
    b.subheader("Correlation coefficient")
    b.code("a.corrcoef()")
    b.subheader("Standard Deviation")
    b.code("np.std(a)")
    b.subheader("Example Program :")
    aff="""
    import numpy as np

# Define arrays
a = np.array([[1, 2, 3],
              [4, 5, 6]])

b = np.array([[7, 8, 9],
              [10, 11, 12]])

# Array-wise sum
array_sum = a.sum()
print("Array-wise sum:", array_sum)

# Array-wise minimum value
array_min = a.min()
print("Array-wise minimum value:", array_min)

# Maximum value of each column
max_per_column = b.max(axis=0)
print("Maximum value of each column:", max_per_column)

# Cumulative sum of the elements along each row
cumulative_sum = b.cumsum(axis=1)
print("Cumulative sum of the elements along each row:")
print(cumulative_sum)

# Mean of all elements
array_mean = a.mean()
print("Mean of all elements:", array_mean)

# Standard deviation
array_std = np.std(b)
print("Standard deviation:", array_std)
"""
    b.code(aff)
    b.markdown("---")
#Aggregate Functions Ends
#String Concepts Starts
    b.title("String Concept in Numpy")
    b.subheader("Subsetting")
    b.code("a[3]")
    b.write("Select the element at the 2nd index")
    b.code("b[1,3]")
    b.write("Select the element at row 1 column 2 (equivalent to b[1][2])")
    b.subheader("Example Program :")
    kk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Select the element at the 2nd index of array 'a'
element_a = a[2]
print("Element at the 2nd index of array 'a':", element_a)

# Select the element at row 1, column 2 of array 'b'
element_b = b[1, 2]
print("Element at row 1, column 2 of array 'b':", element_b)
"""
    b.code(kk)
    b.markdown("---")
    b.subheader("Slicing")
    b.code("a[0:2] ")
    b.write("Select items at index 0 and 1")
    b.code("b[0:2,1]")
    b.write("Select items at rows 0 and 1 in column 1")
    b.code("a[ : :-1] ")
    b.write("Reversed array ")
    b.markdown("---")
    b.subheader("Boolean Indexing")
    b.code("a[a<2]")
    b.write("Select elements from a less than 2")
    b.markdown("---")
#String Concepts Ends

    b.title("Example Program for Adding/Removing Elements")
    kkk="""
import numpy as np

# Define arrays
a = np.array([1, 2, 3])
b = np.array([[10, 15, 20],
              [1, 0, 1]])
d = np.array([[7, 7],
              [7, 7]])

# Concatenate arrays 'a' and 'b' along axis 0
concatenated_ab_axis0 = np.concatenate((a, b), axis=0)
print("Concatenated arrays 'a' and 'b' along axis 0:")
print(concatenated_ab_axis0)

# Stack arrays 'a' and 'b' vertically (row-wise)
stacked_ab_vertically = np.vstack((a, b))
print("\nStacked arrays 'a' and 'b' vertically (row-wise):")
print(stacked_ab_vertically)

# Stack arrays 'a' and 'b' vertically (row-wise) using np.r_
stacked_ab_vertically_r = np.r_[a, b]
print("\nStacked arrays 'a' and 'b' vertically (row-wise) using np.r_:")
print(stacked_ab_vertically_r)

# Stack arrays 'a' and 'b' horizontally (column-wise)
stacked_ab_horizontally = np.hstack((a[:, np.newaxis], b))
print("\nStacked arrays 'a' and 'b' horizontally (column-wise):")
print(stacked_ab_horizontally)

# Create stacked column-wise arrays using np.column_stack()
stacked_column_wise = np.column_stack((a, d))
print("\nStacked column-wise arrays using np.column_stack():")
print(stacked_column_wise)

# Stack arrays 'a' and 'd' column-wise using np.c_
stacked_column_wise_c = np.c_[a, d]
print("\nStacked arrays 'a' and 'd' column-wise using np.c_:")
print(stacked_column_wise_c)
"""
    b.code(kkk)
#B - Column Ends
cs_sidebar()
cs_body()