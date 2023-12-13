# ishield.ai COMMUNICATION MESSAGES
This is the ```COMMUNICATION MESSAGES``` application for ```ishield.ai```. This service is made using python and is used to call OPENAI GPT-3 APIs etc. to generate content. 

## Getting Started
 **Steps to get started**

 *** Making it work locally ***
 

* Clone the repository 
 ```
 git clone git@gitlab.com:ishield1/communication-messages.git
 ```


* Enter the working directory
 ```
 cd communication-messages
 ```


* Create the virtual environment
 ```
 virtualenv venv --python=python3
 ```

* Activate the virtual environment
 ```
 source venv/bin/activate
 ```

* Install the requirements
 ```
 pip install -r requirements.txt
 ```

* Set up the env
 ```
 export ISHIELD_ENV=dev
 ```

* Run the project
 ```
 python main.py
 ```

* Run the project using gunicorn with 4 workers
 ```
 gunicorn --bind 0.0.0.0:5000 -w 4 "main:app"
 ```