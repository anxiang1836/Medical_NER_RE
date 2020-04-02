python run_train_w2v.py \
  --root_path='./data/round1/train/' \
  --w2v_file_path='./saved_model_files/char2vec_prepareData.txt' \
  --w2v_output_path='./saved_model_files/char2vec.model' \
  --emb_size=256 \
  --window_size=5 \
  --sg=0 \
  --hs=0 \
  --negative=3