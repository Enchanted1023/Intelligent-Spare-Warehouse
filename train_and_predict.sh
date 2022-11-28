#!/bin/bash
base_dir=$(cd `dirname ../`; pwd)
echo "work directory(base_dir):"$base_dir
echo "—————————————————————打印本地ip start———————————————————————"
echo "$(ifconfig)"
echo "—————————————————————打印本地ip end———————————————————————"


cluster=$1
env=$2
predict_date=$3
mihome=$4

echo $cluster
echo "env——————————————"${env}
echo "predict_date——————————————"${predict_date}
echo "mihome——————————————"${mihome}
echo ${INFRA_CLIENT}

# 文件变为可执行
# 必须有这个否则报错
chmod 777 ../../time_series_forecasting

echo "————————————————————————————————————————"${base_dir}
echo "————————————————————————————————————————"$cluster
echo "————————————————————————————————————————"${INFRA_CLIENT}

${INFRA_CLIENT}/bin/spark-submit \
--cluster $cluster \
--master yarn-cluster \
--name smart_full_truckload_model \
--conf spark.yarn.job.owners=s_info_algo_app \
--queue root.production.info_group.xdata.s_info_algo_app \
--driver-memory=10g \
--executor-memory=10g \
--num-executors=25 \
--executor-cores=1 \
--conf spark.sql.autoBroadcastJoinThreshold=-1 \
--conf spark.rpc.message.maxSize=256 \
--conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
--conf spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT=1 \
--conf spark.sql.codegen.wholeStage=true \
--conf spark.default.parallemlism=400 \
--conf spark.sql.shuffle.partitions=400 \
--conf spark.scheduler.mode=FAIR \
--conf spark.sql.windowExec.buffesr.spill.threshold=51200 \
--conf spark.executor.memoryOverhead=4g \
--conf spark.dynamicAllocation.enabled=true \
--conf spark.driver.maxResultSize=4g \
--conf spark.network.timeout=36000 \
--conf spark.disable.stdout=false \
--conf spark.files.localize=hdfs://zjyprc-hadoop/spark/zjyprc-hadoop-spark2.3/cache/hive-site.xml \
--conf spark.yarn.dist.archives=hdfs://zjyprc-hadoop/user/s_info_algo_app/python_env/xiaomi_ortools93_lgb332.tar.gz#pyenv \
--conf spark.pypsark.python=pyenv/xiaomi/bin/python3 \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=pyenv/xiaomi/bin/python3 \
--py-files ../src.zip \
../src/pyspark/2_train_and_predict.py --env ${env} --predict_date ${predict_date}

spark_task_status=$?

if [ $spark_task_status -ne 0 ];then
    echo "spark task failed"
    exit 1
fi

echo "finish"