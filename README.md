# dleeim-ML_CW3_Time_Series_on_Dynamical_System

To design a data-driven model predictive controller for a multistage extraction column system—a complex chemical process characterized by non-linear dynamics—I conducted an in-depth data analysis to extract meaningful insights from limited data under stringent computational constraints. I focused heavily on the data collection phase, designing various scenarios with step changes in the input variables, incorporating time lags to allow the dynamical system to approach steady-state conditions. This methodical approach enabled me to explore a wide range of system responses, ensuring the dataset captured the breadth of the system's behavior despite the restricted evaluation budget. My analysis involved several key steps: I applied linear regression with Lasso regularization to model the system, leveraging its ability to select relevant features and prevent overfitting in a data-scarce environment. I modeled the differences in system outputs rather than absolute values, enhancing the stationarity of the time-series data and uncovering dynamic trends more effectively. To ensure robust validation, I implemented k-fold cross-validation tailored for time-series data, structuring it to eliminate look-ahead bias and maintain the temporal integrity of the results. Through this process, I demonstrated my capability to design strategic data collection, apply advanced regression techniques, and adapt validation methods to complex datasets, ultimately laying a solid foundation for predictive modeling in real-world applications.

![image](https://github.com/user-attachments/assets/e5bb63f9-73c0-49ac-bb2e-8f26f5244a62)
![image](https://github.com/user-attachments/assets/5468caa8-f07b-402c-bc44-9ceae2637cda)


Source: "https://github.com/OptiMaL-PSE-Lab/DDMPC-Coursework"
