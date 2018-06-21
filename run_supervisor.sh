unset CUDA_VISIBLE_DEVICES
export PYTHONPATH=./:$PYTHONPATH
../opt/anaconda2/bin/python ../opt/anaconda2/bin/supervisord -c supervisord.conf
