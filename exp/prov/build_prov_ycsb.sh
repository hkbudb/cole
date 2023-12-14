~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./prov-data-10000k > ./prov-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./prov-data-10000k > ./prov-run.txt
cat ./prov-load.txt ./prov-run.txt > ./prov-data.txt
rm ./prov-load.txt ./prov-run.txt
