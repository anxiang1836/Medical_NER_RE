python run_split_build_w2idx_emb.py \
  --root_path='./data/round1/train/' \
  --test_size=0.15 \
  --val_size=0.2 \
  --word2idx_output_path='./saved_model_files/word2idx.pkl' \
  --emb_matrix_output_path='./saved_model_files/emb_matrix.pkl' \
  --w2v_output_path='./saved_model_files/char2vec.model'