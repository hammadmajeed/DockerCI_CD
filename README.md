## ML application automation using Github, Dockers and Jenkins 
In this tutorial, a use case is implemented where EEG dataset is provided to train three differenet ML models, namely, LDA, NN and RF using python and sklearn. 
In the next stage Jenkins is used to pull this code and after installing the requirements, run the train.py as a first job. On successful completion three trained models are created. These are then packaged into Docker image in the next job. On successful creation of the image, Docker image is pushed on to your cloud repository for the general public to use. A Flask version of the image is also hosted as a service. 

```python``` 