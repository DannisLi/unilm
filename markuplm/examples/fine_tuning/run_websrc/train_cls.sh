CUDA_VISIBLE_DEVICES=0,1,2,3 python run_cls.py \
	--train_file /home/v-lizimeng/data/WebSRC/websrc1.0_train_.json \
	--predict_file /home/v-lizimeng/data/WebSRC/websrc1.0_dev_.json \
	--root_dir /home/v-lizimeng/data/WebSRC \
	--model_name_or_path microsoft/markuplm-base \
	--output_dir /home/v-lizimeng/unilm/markuplm/examples/fine_tuning/run_websrc/results_cls \
	--do_train \
	--save_steps 2000 \
	--max_query_length 42 \
	--max_seq_length 512 \
	--per_gpu_train_batch_size 4 \
	--per_gpu_eval_batch_size 8 \
	--num_node_spans_per_case 384 \
	--warmup_ratio 0.1 \
	--num_train_epochs 5