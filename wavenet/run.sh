folder='images'
[ ! -d $folder ] && mkdir $folder

python model.py "${folder}/image"
