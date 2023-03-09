## ML application automation using Github, Dockers and Jenkins 
In this tutorial, a use case is implemented where EEG dataset is provided to train three differenet ML models, namely, LDA, NN and RF using python and sklearn. 
Once coded,  Jenkins is used to pull this code, install the requirements and run the train.py as a first job. On its successful completion, three trained models are created. These are then packaged into Docker image in the next stage. On successful creation of the image, in the delivery stage, Docker image is pushed on to your cloud registry for the general public to use. A Flask app wrapping this image is also hosted as a service. 

```python``` 