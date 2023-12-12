~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readonly/readonly10000k > ./readonly/readonly-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readonly/readonly10000k > ./readonly/readonly-run.txt
cat ./readonly/readonly-load.txt ./readonly/readonly-run.txt > ./readonly/readonly-data.txt
rm ./readonly/readonly-load.txt ./readonly/readonly-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./readwriteeven/readwriteeven10000k > ./readwriteeven/readwriteeven-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./readwriteeven/readwriteeven10000k > ./readwriteeven/readwriteeven-run.txt
cat ./readwriteeven/readwriteeven-load.txt ./readwriteeven/readwriteeven-run.txt > ./readwriteeven/readwriteeven-data.txt
rm ./readwriteeven/readwriteeven-load.txt ./readwriteeven/readwriteeven-run.txt

~/ycsb-0.17.0/bin/ycsb.sh load basic -P ./writeonly/writeonly10000k > ./writeonly/writeonly-load.txt
~/ycsb-0.17.0/bin/ycsb.sh run basic -P ./writeonly/writeonly10000k > ./writeonly/writeonly-run.txt
cat ./writeonly/writeonly-load.txt ./writeonly/writeonly-run.txt > ./writeonly/writeonly-data.txt
rm ./writeonly/writeonly-load.txt ./writeonly/writeonly-run.txt