CURRENT_DIR=$(pwd)

mkdir data
cd data

# download datasets and splits
wget -c http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip -O mitstates.zip
wget -c https://www.senthilpurushwalkam.com/publication/compositional/compositional_split_natural.tar.gz -O compositional_split_natural.tar.gz
wget -c https://s3.mlcloud.uni-tuebingen.de/czsl/cgqa-updated.zip -O cgqa.zip
wget -c https://drive.google.com/file/d/1_J-oiLnj0P0WB9OFqTsxLqHMv8wdOL_a/view?usp=drive_link -O vaw-czsl.zip


# MIT-States
unzip mitstates.zip 'release_dataset/images/*' -d mit-states/
mv mit-states/release_dataset/images mit-states/images/
rm -r mit-states/release_dataset
rename "s/ /_/g" mit-states/images/*

# C-GQA
unzip cgqa.zip -d cgqa/

# VAW-CZSL
unzip vaw-czsl.zip -d vaw-czsl/

# Download new splits for Purushwalkam et. al
tar -zxvf compositional_split_natural.tar.gz

cd $CURRENT_DIR
