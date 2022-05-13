# Fraud feast Example

Expanded [Feast Fraud tutorial](https://github.com/feast-dev/feast-fraud-tutorial/blob/4acf205dfbb3615d2f3e913adf1c28c5f2655f4c/notebooks/Fraud_Detection_Tutorial.ipynb) to use TFX-Addons [FeastExampleGen](/tfx_addons/feast_examplegen/README.md)

## Instructions

Clone the tfx-addons repo and navigate to the fraud_feast directory.

<pre class="devsite-terminal devsite-click-to-copy">
git clone https://github.com/tensorflow/tfx-addons.git
cd tfx-addons/examples/fraud_feast
</pre>

Next, create a Python virtual environment for this example, activate the
environment, and install dependencies. Make sure you are using a version of
python supported by TFX.

<pre class="devsite-terminal devsite-click-to-copy">
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
</pre>

### Local Example
Initialize Feast repository and run local file.

<pre class="devsite-terminal devsite-click-to-copy">
cd repo && feast apply && cd ..
python feast_pipeline_local.py
</pre>
