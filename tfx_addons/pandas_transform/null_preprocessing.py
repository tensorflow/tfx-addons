# Null preprocessing, for minimal testing
from absl import logging


def preprocessing_fn(df, schema, statistics):    
  logging.info('Running null preprocessing')
  return df
