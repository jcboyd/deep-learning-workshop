# deep-learning-workshop

Introductory workshop on deep learning "Deep learning on handwritten digits" given at Institut Curie u900 retreat April, 2017. Inspired and adapted from Stanford Vision course cs231n and official TensorFlow tutorial training LeNet on  MNIST ((https://hub.docker.com/r/tensorflow/tensorflow/)[https://hub.docker.com/r/tensorflow/tensorflow/]).

Docker images (cpu/gpu) available from [https://hub.docker.com/r/jcboyd/deep-learning-workshop/](https://hub.docker.com/r/jcboyd/deep-learning-workshop/).

## NVIDIA-docker

Install NVIDIA-docker according to [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Build

<pre>
docker build -t --build-arg http_proxy=&lt;http_proxy&gt; --build-arg https_proxy=&lt;https_proxy&gt; jcboyd/deep-learning-workshop:[cpu|gpu] .
</pre>

## Pull
<pre>
docker pull jcboyd/deep-learning-workshop:[cpu|gpu]
</pre>

## Run
<pre>
nvidia-docker run -ti -e http_proxy=&lt;http_proxy&gt; -e https_proxy=&lt;https_proxy&gt; -p 8888:8888 deep-learning-workshop:[cpu|gpu]
</pre>

## References
Linear model inspired by [https://cs231n.github.io/linear-classify/](https://cs231n.github.io/linear-classify/)

Notebook arranged for Slideshow with [https://github.com/damianavila/RISE](https://github.com/damianavila/RISE)

Optionally install Jupyter themes with [https://github.com/dunovank/jupyter-themes](https://github.com/dunovank/jupyter-themes)
