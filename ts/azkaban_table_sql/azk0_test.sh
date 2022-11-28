#!/bin/bash

echo ${INFRA_CLIENT}

${INFRA_CLIENT}/bin/spark-submit \
--cluster zjyprc-hadoop-spark2.3 \
--master yarn-cluster \
--name jinzhao1_test \
--conf spark.yarn.job.owners=s_info_algo_app \
--queue root.production.info_group.xdata.s_info_algo_app \
--conf spark.sql.autoBroadcastJoinThreshold=-1 \
--driver-memory=1g \
--executor-memory=1g \
--num-executors=1 \
--executor-cores=1 \
--conf spark.rpc.message.maxSize=256 \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.sql.codegen.wholeStage=true \
--conf spark.default.parallemlism=400 \
--conf spark.sql.shuffle.partitions=400 \
--conf spark.scheduler.mode=FAIR \
--conf spark.sql.windowExec.buffesr.spill.threshold=51200 \
--conf spark.yarn.executor.memoryOverhead=4g \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.driver.maxResultSize=4g \
--conf spark.network.timeout=36000 \
--conf spark.disable.stdout=false \
--conf spark.files.localize=hdfs://zjyprc-hadoop/spark/zjyprc-hadoop-spark2.3/cache/hive-site.xml \
--conf spark.pypsark.python=pyenv/xiaomi/bin/python3 \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=pyenv/xiaomi/bin/python3 \
--conf spark.yarn.dist.archives=hdfs://zjyprc-hadoop/user/s_sales/smart_cargo/xiaomi_pyenv.zip#pyenv \
azk0_test.py
