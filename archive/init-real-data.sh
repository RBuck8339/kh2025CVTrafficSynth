# Source: https://math-ml-x.github.io/TrafficCAM/
echo "Segmentation Init. Going to installing real-world data"

# data dir
mkdir data
cd data

# download fully annotated (saves as Fully_annotate.zip)
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h4oUqECDF05vSYMkYgz0aUc_RRzHlh3e' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h4oUqECDF05vSYMkYgz0aUc_RRzHlh3e" -O Fully_annotate.zip && rm -rf /tmp/cookies.txt
echo "installing data"

if [[ -z "$(which gdown)" ]]; then
    echo "please install gdown with pip3 install --user gdown"
    exit 1
fi

# install fully annotated
zipfile="TrafficCAM-fully-annotated"
fileid="1h4oUqECDF05vSYMkYgz0aUc_RRzHlh3e" 

gdown "$fileid" -O "$zipfile"

# unzip
unzip "$zipfile"

# clean up
rm -rf __MACOSX/
rm "$zipfile"

mv Fully_annotate "$zipfile"
