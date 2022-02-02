#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra Review
# 
# **Basic Notation**
# 
# Linear algebra provides a way of compactly representing and operating on sets of linear equations. For example:
# 
# $$
# \begin{align}
# 4x_{1}-5x_{2} &= -13 \\
# -2x_{1}+3x_{2} &= 9
# \end{align}
# $$
# 
# <p style="text-align:center">The matrix notation of above equations is:</p>
# 
# $$Ax =b$$
# 
# $$\text{with } A = \begin{bmatrix}
#       4 & -5 \\
#       -2 & 3
#       \end{bmatrix}, \enspace b = \begin{bmatrix}
#       -13 \\
#       9
#       \end{bmatrix}$$
# 
# By $A \in \mathbb{R}^{m \times n}$, we denote a matrix with $m$ rows and $n$ columns. By $x \in \mathbb{R}^n$, we denote vector with $n$ entries.
# 
# $$\begin{align}A = \begin{bmatrix}
#       a_{11} & a_{12} & a_{13} & ... & a_{1n} \\
#       a_{21} & a_{22} & a_{23} & ... & a_{2n} \\
#       a_{31} & a_{32} & a_{33} & ... & a_{3n} \\
#       ... & ... & ... & ... & ... \\
#       a_{m1} & a_{m2} & a_{m3} & ... & a_{mn}
#       \end{bmatrix}, \enspace \enspace x = \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       x_{3} \\
#       .. \\
#       x_{n}
#       \end{bmatrix}
#       \end{align}$$
#       
# We denote the $j$th column of $A$ by $a^j$ or $A_{:,j}$:
# 
# $$A = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix}$$
#       
# We denote the $i$th row of $A$ by $a^T$ or $A_{i,:}$:
# 
# $$A = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix}$$
# **Matrix Multiplication**
# 
# The product of two matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$ is the matrix
# 
# $$
# C = AB \in \mathbb{R}^{m \times p} \enspace \text{where, } C_{ij} = \sum_{k=1}^{n}A_{ik}B_{kj}$$
# 
# Note that in order for the matrix product to exist, the number of columns in $A$ must equal the number of rows in $B$.
# 
# **Vector-Vector Multiplication**
# 
# Given two vectors $x,y \in \mathbb{R}^n$, the quantity $x^Ty$, sometimes called the <span class = 'high'>inner product or dot product</span> of the vectors, is a real number given by
# 
# $$\begin{align} x^Ty \in \mathbb{R} = \begin{matrix} [x_{1} & x_{2} & ... & x_{n}]\end{matrix} \begin{bmatrix}
#       y_{1} \\
#       y_{2} \\
#       .. \\
#       y_{n}
#       \end{bmatrix} = \sum_{i=1}^{n}x_{i}y_{i}\end{align}$$
#       
# Given vectors $x \in \mathbb{R}^m, y \in \mathbb{R}^n$ (not necessarily of the same size), $xy^T \in \mathbb{R}^{m \times n}$ is called the <span class = 'high'>outer product</span> of the vectors. It is a matrix, whose entries are given by $(xy^T)_{ij} = x_iy_j$:
# 
# $$xy^T \in \mathbb{R}^{m \times n} = \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       .. \\
#       x_{n}
#       \end{bmatrix}\begin{matrix} [y_{1} & y_{2} & ... & y_{n}]\end{matrix} = \begin{bmatrix}
#       x_{1}y_{1} & x_{1}y_{2} & ... & x_{1}y_{n} \\
#       x_{2}y_{1} & x_{2}y_{2} & ... & x_{2}y_{n} \\
#       ... & ... & ... & ... \\
#       x_{m}y_{1} & x_{m}y_{2} & ... & x_{m}y_{n} \\
#       \end{bmatrix}$$
#       
# **Matrix-Vector Products**
# 
# Given a matrix $A \in \mathbb{R}^{m \times n}$ and a vector $x \in \mathbb{R}^n$, their product is a vector $y = Ax \in \mathbb{R}^m$. There are a couple of ways of looking at matrix-vector multiplication, and we will look at each of them in turn.
# 
# If we write $A$ by rows, then we can express $Ax$ as,
# 
# $$y = Ax = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix}x = \begin{bmatrix}
#       a_{1}^Tx\\
#       a_{2}^Tx \\
#       .. \\
#       a_{m}^Tx 
#       \end{bmatrix}$$

# In other words, the $i$th entry of $y$ is equal to the inner product of the $i$th row of $A$ and $x$, $y=a_i^Tx$.
# 
# Alternatively, let's write $A$ in column form. In this case we see that,
# 
# $$y = Ax = \begin{bmatrix}
#       | & | & | & ... & | \\
#       a^1 & a^2 & a^3 & ... & a^n \\
#       | & | & | & ... &|
#       \end{bmatrix} \begin{bmatrix}
#       x_{1} \\
#       x_{2} \\
#       x_{3} \\
#       .. \\
#       x_{n}
#       \end{bmatrix} = \begin{matrix}
#       a^1
#       \end{matrix}x_1+\begin{matrix}
#       a^2
#       \end{matrix}x_2+...+\begin{matrix}
#       a^n
#       \end{matrix}x_n$$
#       
# In other words, $y$ is a **linear combination** of the columns of $A$, where the coefficients of the linear combination are given by the entries of $x$.
# 
# **Matrix-Matrix Products**
# 
# Using the above information, we can view matrix-matrix multiplication from various perspectives. One of them is to view the matrix-matrix multiplication as a set of vector-vector products. The most obvious viewpoint, which follows immediately from the definition, is that the $(i,j)$th entry of $C$ is equal to the inner product of the $i$th row of $A$ and the $j$th column of $B$. Symbolically, this looks like the following:
# 
# $$C = AB = \begin{bmatrix}
#       - a_{1}^T - \\
#       - a_{2}^T - \\
#       .. \\
#       - a_{m}^T - 
#       \end{bmatrix}\begin{bmatrix}
#       | & | & | & ... & | \\
#       b^1 & b^2 & b^3 & ... & b^n \\
#       | & | & | & ... &|
#       \end{bmatrix}=\begin{bmatrix}
#       a_1^Tb^1 & a_1^Tb^2 & ... & a_1^Tb^p \\
#       a_2^Tb^1 & a_2^Tb^2 & ... & a_2^Tb^p \\
#       .. & .. & ... & .. \\
#       a_m^Tb^1 & a_m^Tb^2 & ... & a_m^Tb^p
#       \end{bmatrix}$$
#       
# Remember that since $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}, a_i \in \mathbb{R}^n \text{ and } b^j \in \mathbb{R}^n$, so these inner products all make sense. This is the 'natural' representation when we represent $A$ by rows and $B$ by columns.

# **Symmetric Matrices**
# A square matrix $A \in \mathbb{R}^{n \times n}$ is **symmetric** if $A=A^T$. It is **anti-symmetric** if $A=-A^T$. It is easy to show that any matrix $A \in \mathbb{R}^{n \times n}$, the matrix $A+A^T$ is symmetric and the matrix $A-A^T$ is anti-symmetric. From this it follows that any square matrix $A \in \mathbb{R}^{n \times n}$ can be be represented as a sum of a symmetric matrix and an anti-symmetric matrix as shown below:
# 
# $$A = \frac{1}{2}(A+A^T)+\frac{1}{2}(A-A^T)$$

# **Norms**
# 
# A norm, $||x||$, of a vector is informally defined as the measure of 'length' of the vector. For example, we have the commonly used Eucledian or $l_{2}$ norm,
# 
# $$||x||_{2} = \sqrt{\sum_{i=1}^{n}x_{i}^2}$$
# 
# Note that $||x||_{2}^2 = x^Tx$
# More formally, norm is a function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ that satisfies $4$ properties:
# 1. For all $x \in \mathbb{R}^n, f(x) \geq 0$ (non-negativity).
# 2. $f(x) = 0$ if and only if $x=0$ (definiteness).
# 3. For all $x \in \mathbb{R}^n, t \in \mathbb{R}, f(tx) = |t|f(x)$ (homogeneity).
# 4. For all $x,y \in \mathbb{R}^n, f(x+y) \leq f(x) +f(y)$ (triangle inequality)
# 
# Other examples of norms are the $l_1$ norm,
# 
# $$||x||_1 = \sum_{i=1}^{n}|x_i|$$
# 
# and the $l_\infty$ norm,
# 
# $$||x||_\infty = max_i|x_i|$$

# 

# 

# 

# 
