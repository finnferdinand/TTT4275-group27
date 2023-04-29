# TTT4275 Classification Project

This is the repository for project group 27 in the course TTT4275 Estimation, 
Detection and Classification.
It provides a solution to the projects regarding classification of Iris flowers 
and handwritten digits, using the Python programming language.

## Project structure

The project is separated into multiple Python modules for ease of development
and scalability.
The modules are structured as follows
```
root/
|-- classifiers/
|   |-- classifier.py    (general classifier python class)
|   |-- knn.py           (knn classifier python class)
|   |-- linear.py        (linear classifier python class)
|-- datasets/
|   |-- data/            (iris and digits datsets)
|   |-- irisdatset.py    (iris datset python class)
|   |-- mnistdatset.py   (MNist dataset python class)
|-- experiments/
|   |-- digits.py        (experiments done with digits dataset)
|   |-- iris.py          (experiments done with iris dataset)
|-- utilities/
|   |-- logger.py        (file logging class)
|-- main.py              (entrypoint)
|-- output.txt           (generated output file)
|-- figures/             (generated figures and plot)
```

## Running the program

### Prerequisites

You need a version of Python newer than 3.6 installed on your system 
(the project was developed using Python 3.8.9). 
Python can be downloaded [here](https://www.python.org/downloads/).

### Setup

Create a virtual python environment (this only needs to be done once)
```bash
$ python -m venv .venv
```
or
```bash
$ python3 -m venv .venv
```


Activate the virtual environment with
```bash
$ source .venv/bin/activate
```

Install requirements
```bash
$ pip install -r requirements.txt
```

### Running the program

Run the main program with
```bash
$ python main.py
```

### Output

Running the program may take a while, and the computation time will depend on 
your specific computer's specs and operating system.
The output of the program is collected in a file `output.txt` containing
all significant results.
All generated figures are saved in a folder `figures/`.

As running the program uses significant computing time, the program is run on
GitHub actions and the outputs are uploaded as artifacts. To find the results
from the most recent run, navigate to ["Actions"](https://github.com/finnferdinand/TTT4275-group27/actions),
select the most recent run and download the zipped file from "Artifacts".
This contains a report of results and all generated figures.
