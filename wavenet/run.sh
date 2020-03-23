folder='images'
[ ! -d $folder ] && mkdir $folder

python model.py wavenet "${folder}/image"
python model.py rnn "${folder}/image_rnn"
