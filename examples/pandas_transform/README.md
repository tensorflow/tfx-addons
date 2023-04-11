# PandasTransform
## TL;DR
PandasTransform is a TFX component which can be used instead of the standard Transform component, and allows you to work with Pandas dataframes for your feature engineering.  Processing is distributed using Beam for scalability.  Operations which require a full pass over the dataset are not currently supported.  Statistics such as the standard deviation, which are required for operations such as z-score normalization, are supplied using the statistics which are captured by StatisticsGen.

## This Example
This example notebook shows how to use the PandasTransform component in a TFX pipeline.  Notice in particular the way that StatisticsGen is used to create statistics for both the raw dataset and the transformed dataset.

Note that although this example does use a TensorFlow model, since PandasTransform does not create a Transform graph the feature engineering which is done in PandasTransform will need to be applied separately during serving.

## Project Team
Robert Crowe (rcrowe-google) robertcrowe--at--google--dot--com
