# create hackathon schema and put data into HDFS
create schema if not exists hackathon;

hdfs dfs -put ../data/features.csv /tmp
hdfs dfs -put ../data/target.csv /tmp
