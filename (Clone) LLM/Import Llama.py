# Databricks notebook source
!rmdir /dbfs/user/chris.crawford@databricks.com/src/llama

# COMMAND ----------

!touch /dbfs/user/chris.crawford@databricks.com/file.txt

# COMMAND ----------

!git clone https://github.com/facebookresearch/llama.git /dbfs/user/chris.crawford@databricks.com/src/

# COMMAND ----------

!ls /dbfs/user/chris.crawford@databricks.com/

# COMMAND ----------

!cd llama

# COMMAND ----------

!ls 

# COMMAND ----------

!chmod 744 llama/download.sh

# COMMAND ----------

!echo -e "https://download.llamameta.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiemJvNHJ1cnYyYW5wd2s0bWhpbmdtb2FuIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZG93bmxvYWQubGxhbWFtZXRhLm5ldFwvKiIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTY5Mjg4NDU2NH19fV19&Signature=kH6-H99j589EVzrRT-zp3ztoKcwmPsjhBpK%7Eq-eKlBBAnAeXV2jDLT9v-8hQtAiwe7pgjZyyQ2DigXZq-qJmI0N0bWAb99pXx5rtbUfMtPezthSYpkL6qDjxY6Zt4KvVVbftA6PW8OqJsZfrJtLoLSYB0%7EsNKbXDqn4Kd%7Emf6EAvjh72AGupHe3N1Od2Qa-ObnZATaBk8KrwP2v4aUNKjI%7Enle2l4dLeHwncvyNlGbjcH3ydTJMZ7jj412UaVe%7E4G0U5vcytCZ6o0Uioq9XtLNgQCeye2QQsBnteVigUhx3gFfs98cLb7qepwSCplR1w7taJCOmDenDqHySPXYx5Fw__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1283076235679667" | llama/download.sh

# COMMAND ----------

!ls /databricks/driver/

# COMMAND ----------


