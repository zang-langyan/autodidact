$$
J = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \vdots & \vdots & \vdots\\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$


For those $\R^n \to \R^m$, the Jacobian Matrix is like above;

Commonly seen functions in Machine Learning are $\R^n \to \R$ such that the Jacobian Matrix is as follows,

$$
J = \begin{pmatrix}
\frac{\partial L}{\partial x_1} & \frac{\partial L}{\partial x_2} & \cdots & \frac{\partial L}{\partial x_n} \\
\end{pmatrix}
$$


$$
\textit{vjp} \to J^T \vec v = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1} & \frac{\partial y_2}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_1} \\
\frac{\partial y_1}{\partial x_2} & \frac{\partial y_2}{\partial x_2} & \cdots & \frac{\partial y_m}{\partial x_2} \\
\vdots & \vdots & \vdots & \vdots\\
\frac{\partial y_1}{\partial x_n} & \frac{\partial y_2}{\partial x_n} & \cdots & \frac{\partial y_m}{\partial x_n}
\end{pmatrix} 
\begin{pmatrix}
\frac{\partial L}{\partial y_1} \\
\frac{\partial L}{\partial y_2} \\
\vdots \\
\frac{\partial L}{\partial y_m} \\
\end{pmatrix} 
= 
\begin{pmatrix}
\frac{\partial L}{\partial x_1} \\
\frac{\partial L}{\partial x_2} \\
\vdots \\
\frac{\partial L}{\partial x_m} \\
\end{pmatrix}
$$