python run_train_bilstm.py \
  --model_type='bilstm-crf' \
  --window_size=70 \
  --expend_size=10 \
  --lstm_units='128,128' \
  --emb_matrix_output_path='./saved_model_files/emb_matrix.pkl' \
  --stop_patience=2 \
  --batch_size=64 \
  --epoch=50 \
