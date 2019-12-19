folder='images'
[ ! -d $folder ] && mkdir $folder

for ((target_label = 0; target_label <= 9; target_label++))
do
    echo "Training on digit $target_label"
    python model.py $target_label "${folder}/digit_${target_label}.png"
done
