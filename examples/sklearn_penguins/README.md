# Penguin Classification Scikit-learn Example

Expanded the [TFX penguin example
pipeline](https://github.com/tensorflow/tfx/tree/master/tfx/examples/penguin)
with instructions for using [scikit-learn](https://scikit-learn.org/stable/)
to build and train the model.

## Instructions

Clone the tfx-addons repo and navigate to the penguin directory.

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx-addons.git
cd tfx-addons/examples/sklearn_penguins
</pre>

Next, create a Python virtual environment for this example, activate the
environment, and install dependencies. Make sure you are using a version of
python supported by TFX.

<pre class="devsite-terminal devsite-click-to-copy">
python -m venv venv
source ./penguin/bin/activate
pip install -r requirements.txt
</pre>

### Local Example
Execute the pipeline python file. Output can be found at `~/tfx`:

<pre class="devsite-terminal devsite-click-to-copy">
python penguin_pipeline_sklearn_local.py
</pre>

### GCP Example
This example uses a custom container image instead of the default TFX ones found
[here](gcr.io/tfx-oss-public/tfx). This custom container ensures the proper
version of scikit-learn is installed. Run the following commands to build this
image and upload it to Google Container Registry (GCR).

<pre class="devsite-terminal devsite-click-to-copy">
cd ~/penguin/experimental
gcloud auth configure-docker
docker build \
  --tag gcr.io/[PROJECT-ID]/tfx-example-sklearn \
  --build-arg TFX_VERSION=$(python -c 'import tfx; print(tfx.__version__)') \
  .
docker push gcr.io/[PROJECT-ID]/tfx-example-sklearn
</pre>

Note that the custom container extends an official TFX container image based on
the local TFX version. If an unreleased version of TFX is being used
(e.g. installing from HEAD), `Dockerfile` may need to be modified to install the
unreleased version.

Set the project id and bucket in `penguin_pipeline_sklearn_gcp.py`. Then, run
the following commands to copy the `~/penguin` directory to GCS and execute the
pipeline python file. Output can be found at `[BUCKET]/tfx`.

<pre class="devsite-terminal devsite-click-to-copy">
vi penguin_pipeline_sklearn_gcp.py
gsutil -m cp -r ~/penguin/data/* gs://[BUCKET]/penguin/data/
gsutil -m cp ~/penguin/experimental/\*.py gs://[BUCKET]/penguin/experimental/

tfx pipeline create \
  --engine kubeflow \
  --pipeline-path penguin_pipeline_sklearn_gcp.py \
  --endpoint [MY-GCP-ENDPOINT.PIPELINES.GOOGLEUSERCONTENT.COM]
</pre>

Note that
`gsutil -m cp ~/penguin/experimental/*.py gs://[BUCKET]/penguin/experimental`
will need to be run every time updates are made to the GCP example.
Additionally, subsequent pipeline deployments should use `tfx pipeline update`
instead of `tfx pipeline create`.
