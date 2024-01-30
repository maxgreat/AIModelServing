# AI Serving server

This simple flask app run several models on GPU.

Using Celery for task management.

Installation:

    pip install -r requirements.txt

Run the app :

    python main.py


## REQUEST

<address>/<modelname> will return a task ID

Get progress and result with <address>/task/<task-id>
