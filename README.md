# deep-learning-workshop

## NVIDIA-docker

Install NVIDIA-docker according to [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Build

<pre>
docker build -t --build-arg http_proxy=&lt;http_proxy&gt; --build-arg https_proxy=&lt;https_proxy&gt; jcboyd/retreat-image .
</pre>

## Pull
<pre>
docker pull jcboyd/retreat-image
</pre>

## Run
<pre>
nvidia-docker run -ti -e http_proxy=&lt;http_proxy&gt; -e https_proxy=&lt;https_proxy&gt; -p 8888:8888 retreat-image
</pre>

## References
Linear model inspired by [https://cs231n.github.io/linear-classify/](https://cs231n.github.io/linear-classify/)

Notebook arranged for Slideshow with [https://github.com/damianavila/RISE](https://github.com/damianavila/RISE)

Optionally install Jupyter themes with [https://github.com/dunovank/jupyter-themes](https://github.com/dunovank/jupyter-themes)
