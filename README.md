
# Human vs AI written text.

This application will give you live prediction and comparison of various classifiers whether the given text is AI written or Human written. 


## How to have it up and running.
It is very simple application ready to predict in seconds. All you need is a python virtual environment. At the moment I am using  conda (Python 3.11).

Open the project in PyCharm.

Add Interpreter (Create Virtual environment)

Once you have the interpreter set, make sure your terminal is reflecting the same virtual environment. If not you can type in the following command:

```bash
Conda activate [Your Virtual Environment Name]
```
After activating virtual environment type the follwoing command to install all the required libraries for the application to run:

```bash
pip install -r .\requirements.txt
```
After all the dependencies have been successfully installed, application is all set to run. To run the application, type in the follwoing command:

```bash
streamlit run app.py
```


