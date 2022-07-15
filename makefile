run-aq-semihard:
	python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 linenet.py --cfg configs/config_aq_semihard.yaml --data-path datasets/AirQualityUCI/AirQualityUCI_charts --batch-size 2048 --output datasets/AirQualityUCI --amp-opt-level O0         
run-traffic-semihard:
	python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 linenet.py --cfg configs/config_traffic_semihard.yaml --data-path datasets/TrafficVolume/TrafficVolume_charts --batch-size 2048 --output datasets/TrafficVolume --amp-opt-level O0     
run-eeg-semihard:
	python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 linenet.py --cfg configs/config_eeg_semihard.yaml --data-path datasets/EEG/EEG_charts --batch-size 2048 --output datasets/EEG --amp-opt-level O0     
run-stocks-semihard:
	python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 linenet.py --cfg configs/config_stocks_semihard.yaml --data-path datasets/Stocks/Stocks_charts --batch-size 2048 --output datasets/Stocks --amp-opt-level O0        