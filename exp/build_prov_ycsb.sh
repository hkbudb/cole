~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./prov/prov-data-10000k > ./prov/prov-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./prov/prov-data-10000k > ./prov/prov-run.txt
cat ./prov/prov-load.txt ./prov/prov-run.txt > ./prov/prov-data.txt
rm ./prov/prov-load.txt ./prov/prov-run.txt
